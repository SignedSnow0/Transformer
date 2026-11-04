//! Transformer building blocks backed by the local autograd graph.
//!
//! This module implements encoder and decoder components for transformer architectures,
//! operating on sequence-major tensors with shape `[sequence_length, model_dimension]`.
//! The implementation features:
//!
//! - Encoder block with bidirectional self-attention and feed-forward networks
//! - Decoder block  with causal self-attention, cross-attention, and feed-forward networks
use crate::autograd::{self};
use numr::{error::Result, runtime::cpu::CpuDevice, tensor::Tensor};
use rand::RngExt;

/// Encoder layer for transformer architecture.
///
/// The encoder implements bidirectional self-attention followed by a position-wise
/// feed-forward network. Both sub-layers use Pre-LayerNorm with residual connections.
///
/// # Architecture
///
/// ```text
/// Input -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> Output
/// ```
///
/// # Parameters
///
/// - Self-attention projections: `w_q`, `b_q`, `w_k`, `b_k`, `w_v`, `b_v`
/// - Layer norms: `gamma_1`, `beta_1` (post-attention), `gamma_2`, `beta_2` (post-FFN)
/// - FFN projections: `w_1`, `b_1`, `w_2`, `b_2`
///
/// All weights are initialized using Kaiming normal initialization,
/// biases are zero-initialized, and LayerNorm parameters are initialized
/// with gamma=1.0 and beta=0.0.
pub struct Encoder {
    pub w_q: autograd::NodeId,
    pub b_q: autograd::NodeId,
    pub w_k: autograd::NodeId,
    pub b_k: autograd::NodeId,
    pub w_v: autograd::NodeId,
    pub b_v: autograd::NodeId,
    pub gamma_1: autograd::NodeId,
    pub beta_1: autograd::NodeId,

    pub w_1: autograd::NodeId,
    pub b_1: autograd::NodeId,
    pub w_2: autograd::NodeId,
    pub b_2: autograd::NodeId,
    pub gamma_2: autograd::NodeId,
    pub beta_2: autograd::NodeId,
}

/// Decoder layer for transformer architecture.
///
/// The decoder implements causal self-attention followed by cross-attention
/// (with encoder outputs) and a position-wise feed-forward network.
/// Both attention sub-layers and the FFN use Pre-LayerNorm with residual connections.
///
/// # Architecture
///
/// ```text
/// Input -> Causal Self-Attention -> Add & Norm
///            ↓
///        Cross-Attention -> Add & Norm -> FFN -> Add & Norm -> Output
///            ↑
///     Enc. Output
/// ```
///
/// # Parameters
///
/// - Self-attention: `w_q`, `b_q`, `w_k`, `b_k`, `w_v`, `b_v`
/// - Cross-attention: `w_q_cross`, `b_q_cross`, `w_k_cross`, `b_k_cross`, `w_v_cross`, `b_v_cross`
/// - Layer norms: `gamma_1`, `beta_1` (post-self-attn),
///               `gamma_2`, `beta_2` (post-cross-attn),
///               `gamma_3`, `beta_3` (post-FFN)
/// - FFN: `w_1`, `b_1`, `w_2`, `b_2`
///
/// The decoder uses causal masking to prevent attending to future positions
/// in the self-attention layer.
pub struct Decoder {
    pub w_q: autograd::NodeId,
    pub b_q: autograd::NodeId,
    pub w_k: autograd::NodeId,
    pub b_k: autograd::NodeId,
    pub w_v: autograd::NodeId,
    pub b_v: autograd::NodeId,
    pub gamma_1: autograd::NodeId,
    pub beta_1: autograd::NodeId,

    pub w_q_cross: autograd::NodeId,
    pub b_q_cross: autograd::NodeId,
    pub w_k_cross: autograd::NodeId,
    pub b_k_cross: autograd::NodeId,
    pub w_v_cross: autograd::NodeId,
    pub b_v_cross: autograd::NodeId,
    pub gamma_2: autograd::NodeId,
    pub beta_2: autograd::NodeId,

    pub w_1: autograd::NodeId,
    pub b_1: autograd::NodeId,
    pub w_2: autograd::NodeId,
    pub b_2: autograd::NodeId,
    pub gamma_3: autograd::NodeId,
    pub beta_3: autograd::NodeId,
}

impl Encoder {
    /// Creates a new encoder layer with initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - CPU device for tensor allocation
    /// * `d_model` - Model dimension (embedding size)
    /// * `d_ff` - Feed-forward hidden dimension
    ///
    /// # Returns
    ///
    /// An `Encoder` instance with all parameters registered in the graph.
    /// Weights use Kaiming normal initialization, biases are zero,
    /// and LayerNorm parameters are initialized to gamma=1, beta=0.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor allocation fails.
    pub fn new(
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self> {
        let w_q = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_q = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_k = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_k = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_v = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_v = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let gamma_1 = graph.variable(Tensor::ones(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;
        let beta_1 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_1 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_ff, d_model),
            &[d_model, d_ff],
            device,
        )?)?;
        let b_1 = graph.variable(Tensor::zeros(&[1, d_ff], numr::dtype::DType::F32, device)?)?;

        let w_2 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_ff * d_model, d_ff),
            &[d_ff, d_model],
            device,
        )?)?;
        let b_2 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let gamma_2 = graph.variable(Tensor::ones(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;
        let beta_2 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        Ok(Self {
            w_q,
            b_q,
            w_k,
            b_k,
            w_v,
            b_v,
            gamma_1,
            beta_1,
            w_1,
            b_1,
            w_2,
            b_2,
            gamma_2,
            beta_2,
        })
    }

    /// Forward pass through the encoder layer.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build computation in
    /// * `device` - CPU device (used for temporary tensors)
    /// * `x` - Input node ID with shape `[seq_len, d_model]`
    /// * `d_model` - Model dimension for scaling attention scores
    ///
    /// # Returns
    ///
    /// Node ID of output tensor with shape `[seq_len, d_model]`
    ///
    /// # Computation
    ///
    /// 1. Bidirectional self-attention: Q, K, V projections with scaled dot-product attention
    /// 2. Residual connection and Pre-LayerNorm
    /// 3. Position-wise FFN: Linear -> ReLU -> Linear
    /// 4. Residual connection and Pre-LayerNorm
    pub fn forward(
        &self,
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        x: autograd::NodeId,
        d_model: usize,
    ) -> Result<autograd::NodeId> {
        let attn_out = self.bidirectional_self_attention(graph, device, x, d_model)?;
        let x_1 = graph.sum(x, attn_out)?;
        let x_1_norm = graph.layer_norm(x_1, self.gamma_1, self.beta_1, 1e-5)?;

        let ffn_hidden = linear(graph, x_1_norm, self.w_1, self.b_1)?;
        let ffn_activated = graph.relu(ffn_hidden)?;
        let ffn_out = linear(graph, ffn_activated, self.w_2, self.b_2)?;

        let x_2 = graph.sum(x_1_norm, ffn_out)?;
        graph.layer_norm(x_2, self.gamma_2, self.beta_2, 1e-5)
    }

    /// Computes bidirectional self-attention without causal masking.
    ///
    /// Each position can attend to all positions in the sequence.
    /// Attention scores are scaled by `1 / sqrt(d_model)` before softmax.
    fn bidirectional_self_attention(
        &self,
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        x: autograd::NodeId,
        d_model: usize,
    ) -> Result<autograd::NodeId> {
        let q = linear(graph, x, self.w_q, self.b_q)?;
        let k = linear(graph, x, self.w_k, self.b_k)?;
        let v = linear(graph, x, self.w_v, self.b_v)?;

        let k_t = graph.transpose(k)?;
        let raw_scores = graph.matmul(q, k_t)?;

        let scale_factor = 1.0 / (d_model as f32).sqrt();
        let scale_node = graph.variable(Tensor::from_slice(&[scale_factor], &[1], device)?)?;
        let scaled_scores = graph.mul(raw_scores, scale_node)?;

        let attn_probs = graph.softmax(scaled_scores)?;
        graph.matmul(attn_probs, v)
    }
}

impl Decoder {
    /// Creates a new decoder layer with initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - CPU device for tensor allocation
    /// * `d_model` - Model dimension (embedding size)
    /// * `d_ff` - Feed-forward hidden dimension
    ///
    /// # Returns
    ///
    /// A `Decoder` instance with all parameters registered in the graph.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor allocation fails.
    pub fn new(
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self> {
        let w_q = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_q = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_k = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_k = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_v = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_v = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let gamma_1 = graph.variable(Tensor::ones(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;
        let beta_1 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_q_cross = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_q_cross = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_k_cross = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_k_cross = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_v_cross = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_v_cross = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let gamma_2 = graph.variable(Tensor::ones(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;
        let beta_2 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let w_1 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_ff, d_model),
            &[d_model, d_ff],
            device,
        )?)?;
        let b_1 = graph.variable(Tensor::zeros(&[1, d_ff], numr::dtype::DType::F32, device)?)?;

        let w_2 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_ff * d_model, d_ff),
            &[d_ff, d_model],
            device,
        )?)?;
        let b_2 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        let gamma_3 = graph.variable(Tensor::ones(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;
        let beta_3 = graph.variable(Tensor::zeros(
            &[1, d_model],
            numr::dtype::DType::F32,
            device,
        )?)?;

        Ok(Self {
            w_q,
            b_q,
            w_k,
            b_k,
            w_v,
            b_v,
            gamma_1,
            beta_1,
            w_q_cross,
            b_q_cross,
            w_k_cross,
            b_k_cross,
            w_v_cross,
            b_v_cross,
            gamma_2,
            beta_2,
            w_1,
            b_1,
            w_2,
            b_2,
            gamma_3,
            beta_3,
        })
    }

    /// Forward pass through the decoder layer.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build computation in
    /// * `device` - CPU device (used for temporary tensors)
    /// * `x` - Decoder input node ID with shape `[seq_len, d_model]`
    /// * `enc_output` - Encoder output node ID with shape `[seq_len, d_model]`
    /// * `d_model` - Model dimension for scaling attention scores
    ///
    /// # Returns
    ///
    /// Node ID of output tensor with shape `[seq_len, d_model]`
    ///
    /// # Computation
    ///
    /// 1. Causal self-attention with triangular masking
    /// 2. Residual connection and Pre-LayerNorm
    /// 3. Cross-attention with encoder outputs
    /// 4. Residual connection and Pre-LayerNorm
    /// 5. Position-wise FFN: Linear -> ReLU -> Linear
    /// 6. Residual connection and Pre-LayerNorm
    pub fn forward(
        &self,
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        x: autograd::NodeId,
        enc_output: autograd::NodeId,
        d_model: usize,
    ) -> Result<autograd::NodeId> {
        let attn_out = self.causal_self_attention(graph, device, x, d_model)?;
        let x_1 = graph.sum(x, attn_out)?;
        let x_1_norm = graph.layer_norm(x_1, self.gamma_1, self.beta_1, 1e-5)?;

        let cross_attn_out = self.cross_attention(graph, device, x_1_norm, enc_output, d_model)?;
        let x_2 = graph.sum(x_1_norm, cross_attn_out)?;
        let x_2_norm = graph.layer_norm(x_2, self.gamma_2, self.beta_2, 1e-5)?;

        let ffn_hidden = linear(graph, x_2_norm, self.w_1, self.b_1)?;
        let ffn_activated = graph.relu(ffn_hidden)?;
        let ffn_out = linear(graph, ffn_activated, self.w_2, self.b_2)?;

        let x_3 = graph.sum(x_2_norm, ffn_out)?;
        graph.layer_norm(x_3, self.gamma_3, self.beta_3, 1e-5)
    }

    /// Computes causal self-attention with triangular masking.
    ///
    /// Each position can only attend to previous positions (including itself).
    /// Upper triangular positions are masked with -inf before softmax.
    fn causal_self_attention(
        &self,
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        x: autograd::NodeId,
        d_model: usize,
    ) -> Result<autograd::NodeId> {
        let q = linear(graph, x, self.w_q, self.b_q)?;
        let k = linear(graph, x, self.w_k, self.b_k)?;
        let v = linear(graph, x, self.w_v, self.b_v)?;

        let k_t = graph.transpose(k)?;
        let raw_scores = graph.matmul(q, k_t)?;

        let scale_factor = 1.0 / (d_model as f32).sqrt();
        let scale_node = graph.variable(Tensor::from_slice(&[scale_factor], &[1], device)?)?;
        let scaled_scores = graph.mul(raw_scores, scale_node)?;

        let masked_scores = graph.causal_mask(scaled_scores)?;
        let attn_probs = graph.softmax(masked_scores)?;
        graph.matmul(attn_probs, v)
    }

    /// Computes cross-attention between decoder inputs and encoder outputs.
    ///
    /// Decoder queries attend to encoder keys and values.
    /// No causal masking is applied (bidirectional attention to encoder).
    fn cross_attention(
        &self,
        graph: &mut autograd::Graph,
        device: &CpuDevice,
        x: autograd::NodeId,
        encoder_output: autograd::NodeId,
        d_model: usize,
    ) -> Result<autograd::NodeId> {
        let q = linear(graph, x, self.w_q_cross, self.b_q_cross)?;
        let k = linear(graph, encoder_output, self.w_k_cross, self.b_k_cross)?;
        let v = linear(graph, encoder_output, self.w_v_cross, self.b_v_cross)?;

        let k_t = graph.transpose(k)?;
        let raw_scores = graph.matmul(q, k_t)?;

        let scale_factor = 1.0 / (d_model as f32).sqrt();
        let scale_node = graph.variable(Tensor::from_slice(&[scale_factor], &[1], device)?)?;
        let scaled_scores = graph.mul(raw_scores, scale_node)?;

        let attn_probs = graph.softmax(scaled_scores)?;
        graph.matmul(attn_probs, v)
    }
}

/// Applies linear transformation: `output = input @ weight + bias`
fn linear(
    graph: &mut autograd::Graph,
    x: autograd::NodeId,
    w: autograd::NodeId,
    b: autograd::NodeId,
) -> Result<autograd::NodeId> {
    let wx = graph.matmul(x, w)?;
    graph.sum(wx, b)
}

/// Generates Kaiming normal initialization values.
/// Standard deviation is `sqrt(2 / fan_in)`.
///
/// Used for weight initialization in linear layers.
/// Based on He et al. 2015: "Delving Deep into Rectifiers".
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Graph;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, Graph) {
        (CpuDevice::new(), Graph::new())
    }

    #[test]
    fn test_encoder_forward_shape() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let seq_len = 4;

        let encoder = Encoder::new(&mut graph, &device, d_model, d_ff)?;
        let input = graph.variable(Tensor::zeros(
            &[seq_len, d_model],
            numr::dtype::DType::F32,
            &device,
        )?)?;
        let out = encoder.forward(&mut graph, &device, input, d_model)?;

        assert_eq!(graph.nodes[out].data.shape(), &[seq_len, d_model]);

        Ok(())
    }

    #[test]
    fn test_decoder_forward_shape() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let seq_len = 4;

        let decoder = Decoder::new(&mut graph, &device, d_model, d_ff)?;
        let x = graph.variable(Tensor::zeros(
            &[seq_len, d_model],
            numr::dtype::DType::F32,
            &device,
        )?)?;
        let enc = graph.variable(Tensor::zeros(
            &[seq_len, d_model],
            numr::dtype::DType::F32,
            &device,
        )?)?;
        let out = decoder.forward(&mut graph, &device, x, enc, d_model)?;

        assert_eq!(graph.nodes[out].data.shape(), &[seq_len, d_model]);

        Ok(())
    }

    #[test]
    fn test_encoder_backward_grads() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let seq_len = 3;

        let encoder = Encoder::new(&mut graph, &device, d_model, d_ff)?;
        let input_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let input = graph.variable(Tensor::from_slice(
            &input_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let out = encoder.forward(&mut graph, &device, input, d_model)?;

        let target_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let target = graph.variable(Tensor::from_slice(
            &target_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let loss = graph.mul(out, target)?;

        graph.backward(loss)?;

        // Check gradients flowed to parameters
        let params = [
            ("w_q", &encoder.w_q),
            ("b_q", &encoder.b_q),
            ("w_1", &encoder.w_1),
            ("gamma_1", &encoder.gamma_1),
        ];
        for (name, p) in params {
            let grad = &graph.nodes[*p].grad;
            assert!(
                grad.to_vec::<f32>().iter().any(|&g| g.abs() > 1e-8),
                "No grad for {}",
                name
            );
        }

        Ok(())
    }

    #[test]
    fn test_decoder_backward_grads() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let seq_len = 4; // Use seq_len = 4 so positions 1, 2, 3 have non-trivial attention

        let decoder = Decoder::new(&mut graph, &device, d_model, d_ff)?;
        let x_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let x = graph.variable(Tensor::from_slice(&x_data, &[seq_len, d_model], &device)?)?;
        let enc_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.2)
            .collect();
        let enc = graph.variable(Tensor::from_slice(&enc_data, &[seq_len, d_model], &device)?)?;
        let out = decoder.forward(&mut graph, &device, x, enc, d_model)?;

        let target_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let target = graph.variable(Tensor::from_slice(
            &target_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let loss = graph.mul(out, target)?;

        graph.backward(loss)?;

        let params = [
            ("w_q", &decoder.w_q),
            ("b_q", &decoder.b_q),
            ("w_q_cross", &decoder.w_q_cross),
            ("gamma_1", &decoder.gamma_1),
        ];
        for (name, p) in params {
            let grad = &graph.nodes[*p].grad;
            let grad_vec = grad.to_vec::<f32>();
            let max_g = grad_vec.iter().map(|g| g.abs()).fold(0.0f32, f32::max);
            println!("{}: max abs grad = {}", name, max_g);
            assert!(max_g > 1e-8, "No grad for {}", name);
        }

        Ok(())
    }

    #[test]
    fn test_encoder_self_attention_causal_property() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 4;
        let d_ff = 16;
        let seq_len = 2;

        let input = graph.variable(Tensor::from_slice(
            &vec![1.0_f32; seq_len * d_model],
            &[seq_len, d_model],
            &device,
        )?)?;

        // Test bidirectional attention without causal mask
        let q = graph.variable(Tensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            &[seq_len, d_model],
            &device,
        )?)?;
        let k = graph.variable(Tensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            &[seq_len, d_model],
            &device,
        )?)?;

        let k_t = graph.transpose(k)?;
        let raw_scores = graph.matmul(q, k_t)?;
        let scale_node = graph.variable(Tensor::from_slice(
            &[1.0 / (d_model as f32).sqrt()],
            &[1],
            &device,
        )?)?;
        let scaled = graph.mul(raw_scores, scale_node)?;
        let probs = graph.softmax(scaled)?;

        let probs_vec = graph.nodes[probs].data.to_vec::<f32>();
        // Each row should sum to 1
        assert!((probs_vec[0] + probs_vec[1] - 1.0).abs() < 1e-5);
        assert!((probs_vec[2] + probs_vec[3] - 1.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_layer_norm_output_normalized() -> Result<()> {
        let (device, mut graph) = setup();

        let x = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[1, 4],
            &device,
        )?)?;
        let gamma = graph.variable(Tensor::ones(&[1, 4], numr::dtype::DType::F32, &device)?)?;
        let beta = graph.variable(Tensor::zeros(&[1, 4], numr::dtype::DType::F32, &device)?)?;

        let out = graph.layer_norm(x, gamma, beta, 1e-5)?;

        let out_data = graph.nodes[out].data.to_vec::<f32>();
        // Mean should be ~0
        let mean: f32 = out_data.iter().sum::<f32>() / out_data.len() as f32;
        assert!(mean.abs() < 1e-5);

        // Std should be ~1
        let var: f32 =
            out_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out_data.len() as f32;
        assert!((var.sqrt() - 1.0).abs() < 1e-5);

        Ok(())
    }
}
