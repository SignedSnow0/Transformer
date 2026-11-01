//! Transformer building blocks backed by the local autograd graph.
//!
//! The implementation operates on sequence-major tensors with shape
//! `[sequence_length, model_dimension]`. A [`TransformerBlock`] adds sinusoidal
//! positional encodings, computes scaled causal self-attention, and applies a
//! two-layer feed-forward network with residual connections.

use crate::autograd;
use numr::{
    runtime::cpu::{CpuDevice, CpuRuntime},
    tensor::Tensor,
};
use rand::RngExt;

/// Parameters and graph-node references for one transformer block.
///
/// The projection matrices have these shapes:
///
/// - `w_q`, `w_k`, and `w_v`: `[d_model, d_model]`
/// - `w_1`: `[d_model, d_ff]`
/// - `w_2`: `[d_ff, d_model]`
pub struct TransformerBlock {
    /// Query projection matrix.
    pub w_q: autograd::NodeId,
    /// Key projection matrix.
    pub w_k: autograd::NodeId,
    /// Value projection matrix.
    pub w_v: autograd::NodeId,
    /// Input-to-hidden feed-forward projection matrix.
    pub w_1: autograd::NodeId,
    /// Hidden-to-output feed-forward projection matrix.
    pub w_2: autograd::NodeId,
}

impl TransformerBlock {
    /// Creates a transformer block and initializes all projection matrices.
    ///
    /// Weights use Kaiming-normal initialization based on their input width.
    /// The tensors are added to `graph`, and the returned node IDs can be used
    /// by [`Self::apply_forward_block`].
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph that owns the parameters.
    /// * `device` - CPU device on which the parameter tensors are allocated.
    /// * `d_model` - Embedding and attention width.
    /// * `d_ff` - Hidden width of the feed-forward network.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter tensor cannot be created or registered.
    pub fn new(
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let w_q = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;

        let w_k = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;

        let w_v = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;

        let w_1 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_ff, d_model),
            &[d_model, d_ff],
            device,
        )?)?;

        let w_2 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_ff * d_model, d_ff),
            &[d_ff, d_model],
            device,
        )?)?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_1,
            w_2,
        })
    }

    /// Builds the forward computation for this transformer block.
    ///
    /// The computation is:
    ///
    /// 1. Add sinusoidal positional encodings to the input.
    /// 2. Compute query, key, and value projections.
    /// 3. Compute scaled dot-product attention with a causal `-inf` mask.
    /// 4. Add the attention output through a residual connection.
    /// 5. Apply the ReLU feed-forward network and a second residual connection.
    ///
    /// The input and output tensors have shape `[sequence_length, d_model]`.
    /// This method only builds the graph; call [`autograd::Graph::backward`]
    /// afterward to compute gradients.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph receiving the computation nodes.
    /// * `device` - CPU device for positional encodings and scaling values.
    /// * `input_embeddings` - Node containing the input embeddings.
    /// * `d_model` - Embedding and attention width.
    ///
    /// # Errors
    ///
    /// Returns an error if any graph operation or tensor allocation fails.
    pub fn apply_forward_block(
        &self,
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        input_embeddings: autograd::NodeId,
        d_model: usize,
    ) -> Result<autograd::NodeId, Box<dyn std::error::Error>> {
        let seq_len = graph.nodes[input_embeddings].data.shape()[0];

        let pe_tensor = positional_encoding(seq_len, d_model, device)?;
        let pe_node = graph.variable(pe_tensor)?;

        let x = graph.sum(input_embeddings, pe_node)?;

        let q = graph.matmul(x, self.w_q)?;
        let k = graph.matmul(x, self.w_k)?;
        let v = graph.matmul(x, self.w_v)?;

        let k_t = graph.transpose(k)?;
        let raw_scores = graph.matmul(q, k_t)?;

        let scale_factor = 1.0 / (d_model as f32).sqrt();
        let scale_tensor = Tensor::from_slice(&[scale_factor], &[1], device)?;
        let scale_node = graph.variable(scale_tensor)?;
        let scaled_scores = graph.mul(raw_scores, scale_node)?;

        let masked_scores = graph.causal_mask(scaled_scores)?;
        let attn_weights = graph.softmax(masked_scores)?;
        let attn_out = graph.matmul(attn_weights, v)?;

        let x_2 = graph.sum(x, attn_out)?;

        let ffn_hidden = graph.matmul(x_2, self.w_1)?;
        let ffn_activated = graph.relu(ffn_hidden)?;
        let ffn_out = graph.matmul(ffn_activated, self.w_2)?;

        let output = graph.sum(x_2, ffn_out)?;

        Ok(output)
    }
}

/// Generates `size` Kaiming-normal values for a layer with `fan_in` inputs.
fn kaiming_init(size: usize, fan_in: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let std = (2.0 / fan_in as f32).sqrt();

    (0..size)
        .map(|_| {
            let u1: f32 = rng.random();
            let u2: f32 = rng.random();
            let normal_std = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

            normal_std * std
        })
        .collect()
}

/// Creates sinusoidal positional encodings with shape
/// `[seq_len, d_model]`.
///
/// Even feature positions use sine and the following odd positions use cosine.
/// For an odd `d_model`, the final feature contains only the sine component.
fn positional_encoding(
    seq_len: usize,
    d_model: usize,
    device: &CpuDevice,
) -> Result<Tensor<CpuRuntime>, Box<dyn std::error::Error>> {
    let mut pe = vec![0.0_f32; seq_len * d_model];

    for pos in 0..seq_len {
        for i in (0..d_model).step_by(2) {
            let denominator = 10000.0_f32.powf(i as f32 / d_model as f32);
            let angle = (pos as f32) / denominator;

            pe[pos * d_model + i] = angle.sin();
            if i + 1 < d_model {
                pe[pos * d_model + i + 1] = angle.cos();
            }
        }
    }

    Ok(Tensor::from_slice(&pe, &[seq_len, d_model], device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_transformer_block() -> Result<(), Box<dyn std::error::Error>> {
        let device = CpuDevice::new();
        let mut graph = autograd::Graph::new();

        let seq_len = 3;
        let d_model = 4;
        let d_ff = 16;

        // 1. Create Input Embeddings (e.g., batch of 3 token vectors)
        let dummy_embeddings = vec![0.5_f32; seq_len * d_model];
        let embeddings = graph.variable(Tensor::from_slice(
            &dummy_embeddings,
            &[seq_len, d_model],
            &device,
        )?)?;

        // 2. Initialize Weights
        let weights = TransformerBlock::new(&mut graph, &device, d_model, d_ff)?;

        // 3. Execute Transformer Block Forward Pass
        let output = weights.apply_forward_block(&mut graph, &device, embeddings, d_model)?;

        // 4. Backward Pass (Compute Gradients across entire model)
        graph.backward(output)?;

        // Verify output shape matches [seq_len, d_model]
        assert_eq!(graph.nodes[output].data.shape(), &[seq_len, d_model]);

        // Verify non-zero gradients flowed back to projection weights and embeddings
        assert!(
            graph.nodes[weights.w_q]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&g| g != 0.0)
        );
        assert!(
            graph.nodes[weights.w_1]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&g| g != 0.0)
        );
        assert!(
            graph.nodes[embeddings]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&g| g != 0.0)
        );

        Ok(())
    }
}
