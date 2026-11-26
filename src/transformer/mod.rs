//! Transformer building blocks backed by the local autograd graph.
//!
//! This module implements encoder and decoder components for transformer architectures,
//! operating on sequence-major tensors with shape `[sequence_length, model_dimension]`.
//! The encoder layers are shared with the BERT demos; the decoder, vocabulary
//! projection, and teacher-forced wrapper are retained for unit tests.
//! The implementation features:
//!
//! - Learned token embeddings
//! - Fixed sinusoidal positional encoding
//! - Multi-head attention with a configurable number of heads
//! - Encoder block with bidirectional self-attention and feed-forward networks
//! - Decoder block with causal self-attention, cross-attention, and feed-forward networks
//! - Vocabulary projection for teacher-forced logits
//! - Pre-LayerNorm residual blocks
mod embedding;
#[cfg(test)]
mod model;
#[cfg(test)]
mod output;
#[cfg(test)]
mod positional;

pub use embedding::TokenEmbedding;

use crate::autograd;
use numr::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::Mutex;

const LAYER_NORM_EPS: f32 = 1e-5;

/// Seed used when parameter initialization is not explicitly seeded.
const DEFAULT_INIT_SEED: u64 = 0x5EED_5EED;

/// Process-wide random source for parameter initialization.
static INIT_RNG: Mutex<Option<StdRng>> = Mutex::new(None);

/// Multi-head attention over sequence-major tensors `[seq_len, d_model]`.
///
/// Each of the `n_heads` heads projects into a `d_head = d_model / n_heads`
/// subspace, computes scaled dot-product attention independently, and the
/// head outputs are concatenated along the feature dimension and projected by
/// `w_o`.
///
/// Pass `key_value = None` for self-attention, or `Some(encoder_output)` for
/// cross-attention. Set `causal = true` for decoder self-attention.
///
/// # Parameters
///
/// - Per-head query/key/value projections: `w_q`, `b_q`, `w_k`, `b_k`, `w_v`, `b_v`
/// - Output projection: `w_o`, `b_o`
pub struct MultiHeadAttention {
    pub n_heads: usize,
    pub d_head: usize,
    pub w_q: Vec<autograd::NodeId>,
    pub b_q: Vec<autograd::NodeId>,
    pub w_k: Vec<autograd::NodeId>,
    pub b_k: Vec<autograd::NodeId>,
    pub w_v: Vec<autograd::NodeId>,
    pub b_v: Vec<autograd::NodeId>,
    pub w_o: autograd::NodeId,
    pub b_o: autograd::NodeId,
}

impl MultiHeadAttention {
    /// Creates a multi-head attention layer with initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - Device for tensor allocation
    /// * `d_model` - Model dimension (embedding size)
    /// * `n_heads` - Number of attention heads; must divide `d_model`
    ///
    /// # Errors
    ///
    /// Returns an error if `n_heads` is zero, does not divide `d_model`, or
    /// tensor allocation fails.
    pub fn new<R>(
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        d_model: usize,
        n_heads: usize,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let d_head = d_model / n_heads;
        let mut w_q = Vec::with_capacity(n_heads);
        let mut b_q = Vec::with_capacity(n_heads);
        let mut w_k = Vec::with_capacity(n_heads);
        let mut b_k = Vec::with_capacity(n_heads);
        let mut w_v = Vec::with_capacity(n_heads);
        let mut b_v = Vec::with_capacity(n_heads);

        for _ in 0..n_heads {
            w_q.push(graph.variable(Tensor::from_slice(
                &kaiming_init(d_model * d_head, d_model),
                &[d_model, d_head],
                device,
            )?)?);
            b_q.push(graph.variable(Tensor::zeros(&[1, d_head], DType::F32, device)?)?);

            w_k.push(graph.variable(Tensor::from_slice(
                &kaiming_init(d_model * d_head, d_model),
                &[d_model, d_head],
                device,
            )?)?);
            b_k.push(graph.variable(Tensor::zeros(&[1, d_head], DType::F32, device)?)?);

            w_v.push(graph.variable(Tensor::from_slice(
                &kaiming_init(d_model * d_head, d_model),
                &[d_model, d_head],
                device,
            )?)?);
            b_v.push(graph.variable(Tensor::zeros(&[1, d_head], DType::F32, device)?)?);
        }

        let w_o = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let b_o = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        Ok(Self {
            n_heads,
            d_head,
            w_q,
            b_q,
            w_k,
            b_k,
            w_v,
            b_v,
            w_o,
            b_o,
        })
    }

    /// Applies multi-head attention.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build computation in
    /// * `device` - Device (used for temporary tensors)
    /// * `query` - Query input with shape `[seq_len, d_model]`
    /// * `key_value` - Optional key/value input with shape `[kv_len, d_model]`.
    ///   `None` uses `query` for self-attention.
    /// * `causal` - Whether to apply a causal mask to the attention scores
    ///
    /// # Returns
    ///
    /// Node ID of the output tensor with shape `[seq_len, d_model]`.
    ///
    /// # Errors
    ///
    /// Returns an error if causal masking is requested with different query
    /// and key/value sequence lengths, or if a tensor operation fails.
    pub fn forward<R>(
        &self,
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        query: autograd::NodeId,
        key_value: Option<autograd::NodeId>,
        causal: bool,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let key_value = key_value.unwrap_or(query);

        let scale = 1.0 / (self.d_head as f32).sqrt();
        let scale_node = graph.variable(Tensor::from_slice(&[scale], &[1], device)?)?;

        let mut heads = Vec::with_capacity(self.n_heads);
        for head in 0..self.n_heads {
            let q = linear(graph, query, self.w_q[head], self.b_q[head])?;
            let k = linear(graph, key_value, self.w_k[head], self.b_k[head])?;
            let v = linear(graph, key_value, self.w_v[head], self.b_v[head])?;

            let k_t = graph.transpose(k)?;
            let raw_scores = graph.matmul(q, k_t)?;
            let scaled_scores = graph.mul(raw_scores, scale_node)?;
            let masked_scores = if causal {
                graph.causal_mask(scaled_scores)?
            } else {
                scaled_scores
            };

            let attn_probs = graph.softmax(masked_scores)?;
            heads.push(graph.matmul(attn_probs, v)?);
        }

        let concatenated = graph.concat(&heads, 1)?;
        linear(graph, concatenated, self.w_o, self.b_o)
    }

    /// Returns the node IDs of every trainable parameter in this layer.
    pub fn parameters(&self) -> Vec<autograd::NodeId> {
        let mut params = Vec::with_capacity(6 * self.n_heads + 2);
        for head in 0..self.n_heads {
            params.extend([
                self.w_q[head],
                self.b_q[head],
                self.w_k[head],
                self.b_k[head],
                self.w_v[head],
                self.b_v[head],
            ]);
        }
        params.push(self.w_o);
        params.push(self.b_o);
        params
    }
}

/// Encoder layer for transformer architecture.
///
/// The encoder implements multi-head bidirectional self-attention followed by
/// a position-wise feed-forward network. Both sub-layers use Pre-LayerNorm
/// with residual connections:
///
/// ```text
/// x1 = x  + MHA(LayerNorm(x))
/// x2 = x1 + FFN(LayerNorm(x1))
/// ```
///
/// # Parameters
///
/// - Multi-head self-attention
/// - Layer norms: `gamma_1`, `beta_1` (pre-attention), `gamma_2`, `beta_2` (pre-FFN)
/// - FFN projections: `w_1`, `b_1`, `w_2`, `b_2`
///
/// All weights are initialized using Kaiming normal initialization,
/// biases are zero-initialized, and LayerNorm parameters are initialized
/// with gamma=1.0 and beta=0.0.
pub struct Encoder {
    pub attn: MultiHeadAttention,
    pub gamma_1: autograd::NodeId,
    pub beta_1: autograd::NodeId,

    pub w_1: autograd::NodeId,
    pub b_1: autograd::NodeId,
    pub w_2: autograd::NodeId,
    pub b_2: autograd::NodeId,
    pub gamma_2: autograd::NodeId,
    pub beta_2: autograd::NodeId,
}

impl Encoder {
    /// Creates a new encoder layer with initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - Device for tensor allocation
    /// * `d_model` - Model dimension (embedding size)
    /// * `d_ff` - Feed-forward hidden dimension
    /// * `n_heads` - Number of attention heads; must divide `d_model`
    ///
    /// # Errors
    ///
    /// Returns an error if the attention configuration is invalid or tensor
    /// allocation fails.
    pub fn new<R>(
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let attn = MultiHeadAttention::new(graph, device, d_model, n_heads)?;

        let gamma_1 = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta_1 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        let w_1 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_ff, d_model),
            &[d_model, d_ff],
            device,
        )?)?;
        let b_1 = graph.variable(Tensor::zeros(&[1, d_ff], DType::F32, device)?)?;

        let w_2 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_ff * d_model, d_ff),
            &[d_ff, d_model],
            device,
        )?)?;
        let b_2 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        let gamma_2 = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta_2 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        Ok(Self {
            attn,
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
    /// * `device` - Device (used for temporary tensors)
    /// * `x` - Input node ID with shape `[seq_len, d_model]`
    ///
    /// # Returns
    ///
    /// Node ID of output tensor with shape `[seq_len, d_model]`
    ///
    /// # Computation
    ///
    /// 1. Pre-LayerNorm, then bidirectional multi-head self-attention
    /// 2. Residual connection
    /// 3. Pre-LayerNorm, then position-wise FFN: Linear -> GELU -> Linear
    /// 4. Residual connection
    #[allow(dead_code)]
    pub fn forward<R>(
        &self,
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        x: autograd::NodeId,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let mut unused_rng = StdRng::seed_from_u64(0);
        self.forward_with_dropout(graph, device, x, 0.0, false, &mut unused_rng)
    }

    /// Forward pass with optional dropout on the residual branches.
    ///
    /// `rng` is only sampled when `training` is true and `dropout_probability`
    /// is greater than zero, so evaluation passes are deterministic.
    pub fn forward_with_dropout<R>(
        &self,
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        x: autograd::NodeId,
        dropout_probability: f32,
        training: bool,
        rng: &mut impl RngExt,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let attn_input = graph.layer_norm(x, self.gamma_1, self.beta_1, LAYER_NORM_EPS)?;
        let attn_out = self.attn.forward(graph, device, attn_input, None, false)?;
        let attn_out = if training {
            graph.dropout(attn_out, dropout_probability, rng)?
        } else {
            attn_out
        };
        let x_1 = graph.sum(x, attn_out)?;

        let ffn_input = graph.layer_norm(x_1, self.gamma_2, self.beta_2, LAYER_NORM_EPS)?;
        let ffn_hidden = linear(graph, ffn_input, self.w_1, self.b_1)?;
        let ffn_activated = graph.gelu(ffn_hidden)?;
        let ffn_out = linear(graph, ffn_activated, self.w_2, self.b_2)?;
        let ffn_out = if training {
            graph.dropout(ffn_out, dropout_probability, rng)?
        } else {
            ffn_out
        };

        graph.sum(x_1, ffn_out)
    }

    /// Returns the node IDs of every trainable parameter in this layer.
    pub fn parameters(&self) -> Vec<autograd::NodeId> {
        let mut params = self.attn.parameters();
        params.extend([
            self.gamma_1,
            self.beta_1,
            self.w_1,
            self.b_1,
            self.w_2,
            self.b_2,
            self.gamma_2,
            self.beta_2,
        ]);
        params
    }
}

/// Decoder layer for transformer architecture.
///
/// The decoder implements causal multi-head self-attention, cross-attention
/// with encoder outputs, and a position-wise feed-forward network. All
/// sub-layers use Pre-LayerNorm with residual connections:
///
/// ```text
/// x1 = x  + CausalMHA(LayerNorm(x))
/// x2 = x1 + CrossMHA(LayerNorm(x1), enc_output)
/// x3 = x2 + FFN(LayerNorm(x2))
/// ```
///
/// # Parameters
///
/// - Causal self-attention
/// - Cross-attention over encoder outputs
/// - Layer norms: `gamma_1`, `beta_1` (pre-self-attention),
///   `gamma_2`, `beta_2` (pre-cross-attention), `gamma_3`, `beta_3` (pre-FFN)
/// - FFN: `w_1`, `b_1`, `w_2`, `b_2`
#[cfg(test)]
pub struct Decoder {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: MultiHeadAttention,
    pub gamma_1: autograd::NodeId,
    pub beta_1: autograd::NodeId,

    pub gamma_2: autograd::NodeId,
    pub beta_2: autograd::NodeId,

    pub w_1: autograd::NodeId,
    pub b_1: autograd::NodeId,
    pub w_2: autograd::NodeId,
    pub b_2: autograd::NodeId,
    pub gamma_3: autograd::NodeId,
    pub beta_3: autograd::NodeId,
}

#[cfg(test)]
impl Decoder {
    /// Creates a new decoder layer with initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - Device for tensor allocation
    /// * `d_model` - Model dimension (embedding size)
    /// * `d_ff` - Feed-forward hidden dimension
    /// * `n_heads` - Number of attention heads; must divide `d_model`
    ///
    /// # Errors
    ///
    /// Returns an error if the attention configuration is invalid or tensor
    /// allocation fails.
    pub fn new<R>(
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let self_attn = MultiHeadAttention::new(graph, device, d_model, n_heads)?;
        let cross_attn = MultiHeadAttention::new(graph, device, d_model, n_heads)?;

        let gamma_1 = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta_1 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;
        let gamma_2 = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta_2 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        let w_1 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_ff, d_model),
            &[d_model, d_ff],
            device,
        )?)?;
        let b_1 = graph.variable(Tensor::zeros(&[1, d_ff], DType::F32, device)?)?;

        let w_2 = graph.variable(Tensor::from_slice(
            &kaiming_init(d_ff * d_model, d_ff),
            &[d_ff, d_model],
            device,
        )?)?;
        let b_2 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        let gamma_3 = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta_3 = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        Ok(Self {
            self_attn,
            cross_attn,
            gamma_1,
            beta_1,
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
    /// * `device` - Device (used for temporary tensors)
    /// * `x` - Decoder input node ID with shape `[seq_len, d_model]`
    /// * `enc_output` - Encoder output node ID with shape `[enc_len, d_model]`
    ///
    /// # Returns
    ///
    /// Node ID of output tensor with shape `[seq_len, d_model]`
    ///
    /// # Computation
    ///
    /// 1. Pre-LayerNorm, then causal multi-head self-attention
    /// 2. Residual connection
    /// 3. Pre-LayerNorm, then cross-attention with encoder outputs
    /// 4. Residual connection
    /// 5. Pre-LayerNorm, then position-wise FFN: Linear -> ReLU -> Linear
    /// 6. Residual connection
    pub fn forward<R>(
        &self,
        graph: &mut autograd::Graph<R>,
        device: &R::Device,
        x: autograd::NodeId,
        enc_output: autograd::NodeId,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let self_attn_input = graph.layer_norm(x, self.gamma_1, self.beta_1, LAYER_NORM_EPS)?;
        let self_attn_out = self
            .self_attn
            .forward(graph, device, self_attn_input, None, true)?;
        let x_1 = graph.sum(x, self_attn_out)?;

        let cross_attn_input = graph.layer_norm(x_1, self.gamma_2, self.beta_2, LAYER_NORM_EPS)?;
        let cross_attn_out =
            self.cross_attn
                .forward(graph, device, cross_attn_input, Some(enc_output), false)?;
        let x_2 = graph.sum(x_1, cross_attn_out)?;

        let ffn_input = graph.layer_norm(x_2, self.gamma_3, self.beta_3, LAYER_NORM_EPS)?;
        let ffn_hidden = linear(graph, ffn_input, self.w_1, self.b_1)?;
        let ffn_activated = graph.gelu(ffn_hidden)?;
        let ffn_out = linear(graph, ffn_activated, self.w_2, self.b_2)?;

        graph.sum(x_2, ffn_out)
    }
}

/// Applies linear transformation: `output = input @ weight + bias`
pub(crate) fn linear<R>(
    graph: &mut autograd::Graph<R>,
    x: autograd::NodeId,
    w: autograd::NodeId,
    b: autograd::NodeId,
) -> Result<autograd::NodeId>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    let wx = graph.matmul(x, w)?;
    graph.sum(wx, b)
}

/// Generates zero-mean normal initialization values with the given standard
/// deviation using the Box-Muller transform.
fn normal_init(size: usize, std: f32) -> Vec<f32> {
    let mut guard = INIT_RNG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let rng = guard.get_or_insert_with(|| StdRng::seed_from_u64(DEFAULT_INIT_SEED));

    (0..size)
        .map(|_| {
            // Clamp to avoid `ln(0)` if the RNG ever returns exactly zero.
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random();
            let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            normal * std
        })
        .collect()
}

/// Generates Kaiming normal initialization values.
/// Standard deviation is `sqrt(2 / fan_in)`.
///
/// Used for weight initialization in linear layers.
/// Based on He et al. 2015: "Delving Deep into Rectifiers".
pub(crate) fn kaiming_init(size: usize, fan_in: usize) -> Vec<f32> {
    normal_init(size, (2.0 / fan_in as f32).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Graph;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, Graph) {
        (
            CpuDevice::new(),
            Graph::<numr::runtime::cpu::CpuRuntime>::new(),
        )
    }

    #[test]
    fn test_mha_forward_shape() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let n_heads = 2;
        let seq_len = 4;

        let attn = MultiHeadAttention::new(&mut graph, &device, d_model, n_heads)?;
        let input = graph.variable(Tensor::zeros(&[seq_len, d_model], DType::F32, &device)?)?;
        let out = attn.forward(&mut graph, &device, input, None, false)?;

        assert_eq!(graph.nodes[out].data.shape(), &[seq_len, d_model]);

        Ok(())
    }

    #[test]
    fn test_mha_cross_attention_shape() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let n_heads = 2;
        let query_len = 2;
        let key_value_len = 5;

        let attn = MultiHeadAttention::new(&mut graph, &device, d_model, n_heads)?;
        let query = graph.variable(Tensor::zeros(&[query_len, d_model], DType::F32, &device)?)?;
        let key_value = graph.variable(Tensor::zeros(
            &[key_value_len, d_model],
            DType::F32,
            &device,
        )?)?;
        let out = attn.forward(&mut graph, &device, query, Some(key_value), false)?;

        assert_eq!(graph.nodes[out].data.shape(), &[query_len, d_model]);

        Ok(())
    }

    #[test]
    fn test_mha_causal_property() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let n_heads = 2;
        let seq_len = 3;

        let attn = MultiHeadAttention::new(&mut graph, &device, d_model, n_heads)?;
        let input_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let mut changed_data = input_data.clone();
        for value in changed_data.iter_mut().skip((seq_len - 1) * d_model) {
            *value += 1.0;
        }

        let unchanged = graph.variable(Tensor::from_slice(
            &input_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let changed = graph.variable(Tensor::from_slice(
            &changed_data,
            &[seq_len, d_model],
            &device,
        )?)?;

        let out_unchanged = attn.forward(&mut graph, &device, unchanged, None, true)?;
        let out_changed = attn.forward(&mut graph, &device, changed, None, true)?;

        let unchanged_values = graph.nodes[out_unchanged].data.to_vec::<f32>();
        let changed_values = graph.nodes[out_changed].data.to_vec::<f32>();

        for row in 0..(seq_len - 1) {
            for column in 0..d_model {
                let index = row * d_model + column;
                assert!(
                    (unchanged_values[index] - changed_values[index]).abs() < 1e-6,
                    "causal leak at row {} column {}",
                    row,
                    column
                );
            }
        }

        let last_row_start = (seq_len - 1) * d_model;
        assert!(
            (last_row_start..(seq_len * d_model))
                .any(|i| (unchanged_values[i] - changed_values[i]).abs() > 1e-6),
            "changing the last token should change its output"
        );

        Ok(())
    }

    #[test]
    fn test_mha_backward_grads() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let n_heads = 2;
        let seq_len = 3;

        let attn = MultiHeadAttention::new(&mut graph, &device, d_model, n_heads)?;
        let input_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let input = graph.variable(Tensor::from_slice(
            &input_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let out = attn.forward(&mut graph, &device, input, None, true)?;

        let target_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.05)
            .collect();
        let target = graph.variable(Tensor::from_slice(
            &target_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let loss = graph.mul(out, target)?;

        graph.backward_with_grad(
            loss,
            Tensor::ones(&[seq_len, d_model], DType::F32, &device)?,
        )?;

        let params = [
            ("w_q_0", attn.w_q[0]),
            ("b_q_0", attn.b_q[0]),
            ("w_o", attn.w_o),
            ("b_o", attn.b_o),
        ];
        for (name, param) in params {
            let grad = &graph.nodes[param].grad;
            assert!(
                grad.to_vec::<f32>().iter().any(|&g| g.abs() > 1e-8),
                "No grad for {}",
                name
            );
        }

        assert_eq!(
            graph.nodes[attn.b_q[0]].grad.shape(),
            &[1, d_model / n_heads]
        );

        Ok(())
    }

    #[test]
    fn test_encoder_forward_shape() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let n_heads = 2;
        let seq_len = 4;

        let encoder = Encoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
        let input = graph.variable(Tensor::zeros(&[seq_len, d_model], DType::F32, &device)?)?;
        let out = encoder.forward(&mut graph, &device, input)?;

        assert_eq!(graph.nodes[out].data.shape(), &[seq_len, d_model]);

        Ok(())
    }

    #[test]
    fn test_decoder_forward_shape() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let n_heads = 2;
        let seq_len = 4;

        let decoder = Decoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
        let x = graph.variable(Tensor::zeros(&[seq_len, d_model], DType::F32, &device)?)?;
        let enc = graph.variable(Tensor::zeros(&[seq_len, d_model], DType::F32, &device)?)?;
        let out = decoder.forward(&mut graph, &device, x, enc)?;

        assert_eq!(graph.nodes[out].data.shape(), &[seq_len, d_model]);

        Ok(())
    }

    #[test]
    fn test_encoder_backward_grads() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let n_heads = 2;
        let seq_len = 3;

        let encoder = Encoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
        let input_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let input = graph.variable(Tensor::from_slice(
            &input_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let out = encoder.forward(&mut graph, &device, input)?;

        let target_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let target = graph.variable(Tensor::from_slice(
            &target_data,
            &[seq_len, d_model],
            &device,
        )?)?;
        let loss = graph.mul(out, target)?;

        graph.backward_with_grad(
            loss,
            Tensor::ones(&[seq_len, d_model], DType::F32, &device)?,
        )?;

        let params = [
            ("w_v_0", encoder.attn.w_v[0]),
            ("w_o", encoder.attn.w_o),
            ("b_o", encoder.attn.b_o),
            ("w_1", encoder.w_1),
            ("gamma_1", encoder.gamma_1),
        ];
        for (name, param) in params {
            let grad = &graph.nodes[param].grad;
            assert!(
                grad.to_vec::<f32>().iter().any(|&g| g.abs() > 1e-8),
                "No grad for {}",
                name
            );
        }

        assert_eq!(
            graph.nodes[encoder.attn.b_q[0]].grad.shape(),
            &[1, d_model / n_heads]
        );

        Ok(())
    }

    #[test]
    fn test_decoder_backward_grads() -> Result<()> {
        let (device, mut graph) = setup();
        let d_model = 8;
        let d_ff = 32;
        let n_heads = 2;
        let seq_len = 4;

        let decoder = Decoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
        let x_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        let x = graph.variable(Tensor::from_slice(&x_data, &[seq_len, d_model], &device)?)?;
        let enc_data: Vec<f32> = (0..(seq_len * d_model))
            .map(|i| (i + 1) as f32 * 0.2)
            .collect();
        let enc = graph.variable(Tensor::from_slice(&enc_data, &[seq_len, d_model], &device)?)?;
        let out = decoder.forward(&mut graph, &device, x, enc)?;

        let loss = graph.mul(out, out)?;

        graph.backward_with_grad(
            loss,
            Tensor::ones(&[seq_len, d_model], DType::F32, &device)?,
        )?;

        let params = [
            ("w_v_self_0", decoder.self_attn.w_v[0]),
            ("w_o_self", decoder.self_attn.w_o),
            ("w_v_cross_0", decoder.cross_attn.w_v[0]),
            ("w_o_cross", decoder.cross_attn.w_o),
            ("w_1", decoder.w_1),
            ("gamma_1", decoder.gamma_1),
        ];
        for (name, param) in params {
            let grad = &graph.nodes[param].grad;
            let grad_vec = grad.to_vec::<f32>();
            let max_g = grad_vec.iter().map(|g| g.abs()).fold(0.0f32, f32::max);
            println!("{}: max abs grad = {}", name, max_g);
            assert!(max_g > 1e-8, "No grad for {}", name);
        }

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
        let gamma = graph.variable(Tensor::ones(&[1, 4], DType::F32, &device)?)?;
        let beta = graph.variable(Tensor::zeros(&[1, 4], DType::F32, &device)?)?;

        let out = graph.layer_norm(x, gamma, beta, LAYER_NORM_EPS)?;

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
