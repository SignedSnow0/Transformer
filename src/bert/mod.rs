//! BERT-style encoder-only model with a masked-language-model (MLM) head.
//!
//! The model follows the main ingredients of BERT:
//!
//! - learned token embeddings,
//! - learned position embeddings,
//! - learned segment/token-type embeddings,
//! - a stack of bidirectional encoder blocks,
//! - a vocabulary projection trained with masked-language modeling,
//! - optional weight tying between the token embedding table and the MLM
//!   output projection.
//!
//! This implementation deliberately reuses the project's existing
//! [`Encoder`] blocks and autograd primitives. It is therefore a compact
//! educational BERT rather than an exact reproduction of the paper:
//!
//! - the encoder blocks use the project's pre-LayerNorm residual structure,
//!   while the original BERT uses post-LayerNorm blocks,
//! - the feed-forward networks and MLM head use a tanh-approximated GELU,
//! - dropout is applied to residual branches and the MLM dense head while
//!   training, and disabled automatically for evaluation,
//! - next-sentence prediction is not implemented; training uses MLM only.
//!
//! BERT is encoder-only. It is useful for masked-token prediction and for
//! producing contextual representations, not for autoregressive translation.
//! [`encoding::EncodingStrategy`] provides both word-level and learned BPE
//! vocabularies; the MLM example uses BPE. Task-specific data loaders, heads,
//! and training loops live in [`examples`], while this module contains only the
//! reusable model.

pub mod encoding;
pub mod examples;

use crate::autograd::{self, Graph};
use crate::transformer::{Encoder, TokenEmbedding, kaiming_init, linear};
use numr::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// LayerNorm epsilon used by BERT embeddings and the MLM head.
pub const BERT_LAYER_NORM_EPS: f32 = 1e-5;

/// Hyperparameters for [`Bert`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Hidden/model dimension.
    pub d_model: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Number of attention heads; must divide `d_model`.
    pub n_heads: usize,
    /// Number of encoder layers.
    pub n_layers: usize,
    /// Maximum sequence length supported by the position embeddings.
    pub max_len: usize,
    /// Inverted dropout probability applied during training. Zero disables it.
    pub dropout: f32,
    /// Whether the MLM output projection shares the token embedding table.
    pub tie_word_embeddings: bool,
}

impl BertConfig {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return invalid("vocab_size", "must be greater than zero");
        }
        if self.d_model == 0 {
            return invalid("d_model", "must be greater than zero");
        }
        if self.d_ff == 0 {
            return invalid("d_ff", "must be greater than zero");
        }
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return invalid(
                "n_heads",
                format!(
                    "must be greater than zero and divide d_model ({})",
                    self.d_model
                ),
            );
        }
        if self.n_layers == 0 {
            return invalid("n_layers", "must be greater than zero");
        }
        if self.max_len == 0 {
            return invalid("max_len", "must be greater than zero");
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return invalid("dropout", "must be in [0, 1)");
        }
        Ok(())
    }
}

/// BERT input embeddings: token + position + segment, then LayerNorm.
pub struct BertEmbeddings {
    /// Learned token embedding table of shape `[vocab_size, d_model]`.
    pub token: TokenEmbedding,
    /// Learned position embedding table of shape `[max_len, d_model]`.
    pub position: TokenEmbedding,
    /// Learned segment embedding table with two rows (segment 0 and 1).
    pub token_type: TokenEmbedding,
    /// LayerNorm scale.
    pub gamma: autograd::NodeId,
    /// LayerNorm shift.
    pub beta: autograd::NodeId,
}

impl BertEmbeddings {
    /// Creates a randomly initialized BERT embedding block.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or tensor allocation fails.
    pub fn new<R>(
        graph: &mut Graph<R>,
        device: &R::Device,
        vocab_size: usize,
        d_model: usize,
        max_len: usize,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        if vocab_size == 0 || d_model == 0 || max_len == 0 {
            return invalid("embedding dimensions", "must all be greater than zero");
        }

        let token = TokenEmbedding::new(graph, device, vocab_size, d_model)?;
        let position = TokenEmbedding::new(graph, device, max_len, d_model)?;
        let token_type = TokenEmbedding::new(graph, device, 2, d_model)?;
        let gamma = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        Ok(Self {
            token,
            position,
            token_type,
            gamma,
            beta,
        })
    }

    /// Applies token, position, and segment embeddings followed by LayerNorm.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is empty, too long, has mismatched token
    /// type IDs, or if a tensor operation fails.
    pub fn forward<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
        token_type_ids: &[usize],
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        if token_ids.is_empty() {
            return invalid("token_ids", "must not be empty");
        }
        if token_ids.len() > self.position.vocab_size {
            return invalid(
                "token_ids",
                format!(
                    "sequence length {} exceeds max_len {}",
                    token_ids.len(),
                    self.position.vocab_size
                ),
            );
        }
        if token_type_ids.len() != token_ids.len() {
            return invalid("token_type_ids", "must have the same length as token_ids");
        }
        if token_type_ids.iter().any(|&id| id > 1) {
            return invalid("token_type_ids", "entries must be 0 or 1");
        }
        if token_ids.iter().any(|&id| id >= self.token.vocab_size) {
            return invalid("token_ids", "contains an ID outside the vocabulary");
        }

        let positions: Vec<usize> = (0..token_ids.len()).collect();
        let token = self.token.forward(graph, device, token_ids)?;
        let position = self.position.forward(graph, device, &positions)?;
        let token_type = self.token_type.forward(graph, device, token_type_ids)?;
        let token_and_position = graph.sum(token, position)?;
        let combined = graph.sum(token_and_position, token_type)?;

        graph.layer_norm(combined, self.gamma, self.beta, BERT_LAYER_NORM_EPS)
    }

    /// Returns the trainable embedding parameters.
    pub fn parameters(&self) -> Vec<autograd::NodeId> {
        vec![
            self.token.table,
            self.position.table,
            self.token_type.table,
            self.gamma,
            self.beta,
        ]
    }
}

/// Masked-language-model head.
///
/// The head applies a learned dense projection, GELU, LayerNorm, and a
/// vocabulary projection. When weight tying is enabled, the vocabulary
/// projection uses the transpose of the token embedding table.
pub struct BertMlmHead {
    /// Dense projection weight with shape `[d_model, d_model]`.
    pub dense_w: autograd::NodeId,
    /// Dense projection bias with shape `[1, d_model]`.
    pub dense_b: autograd::NodeId,
    /// LayerNorm scale.
    pub gamma: autograd::NodeId,
    /// LayerNorm shift.
    pub beta: autograd::NodeId,
    /// Independent vocabulary projection, or `None` when weights are tied.
    pub vocab_w: Option<autograd::NodeId>,
    /// Vocabulary projection bias with shape `[1, vocab_size]`.
    pub vocab_b: autograd::NodeId,
}

impl BertMlmHead {
    /// Creates an MLM head.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or tensor allocation fails.
    pub fn new<R>(
        graph: &mut Graph<R>,
        device: &R::Device,
        d_model: usize,
        vocab_size: usize,
        tie_word_embeddings: bool,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        if d_model == 0 || vocab_size == 0 {
            return invalid("MLM dimensions", "must all be greater than zero");
        }

        let dense_w = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * d_model, d_model),
            &[d_model, d_model],
            device,
        )?)?;
        let dense_b = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;
        let gamma = graph.variable(Tensor::ones(&[1, d_model], DType::F32, device)?)?;
        let beta = graph.variable(Tensor::zeros(&[1, d_model], DType::F32, device)?)?;

        let vocab_w = if tie_word_embeddings {
            None
        } else {
            Some(graph.variable(Tensor::from_slice(
                &kaiming_init(d_model * vocab_size, d_model),
                &[d_model, vocab_size],
                device,
            )?)?)
        };
        let vocab_b = graph.variable(Tensor::zeros(&[1, vocab_size], DType::F32, device)?)?;

        Ok(Self {
            dense_w,
            dense_b,
            gamma,
            beta,
            vocab_w,
            vocab_b,
        })
    }

    /// Produces vocabulary logits from encoder hidden states.
    ///
    /// `tied_token_embedding` must be the `[vocab_size, d_model]` token table
    /// when [`Self::vocab_w`] is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the head has no vocabulary projection available or
    /// if a tensor operation fails.
    #[allow(dead_code)]
    pub fn forward<R>(
        &self,
        graph: &mut Graph<R>,
        hidden: autograd::NodeId,
        tied_token_embedding: Option<autograd::NodeId>,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let mut unused_rng = StdRng::seed_from_u64(0);
        self.forward_with_dropout(
            graph,
            hidden,
            tied_token_embedding,
            0.0,
            false,
            &mut unused_rng,
        )
    }

    /// Produces logits with optional dropout after the dense projection.
    pub fn forward_with_dropout<R>(
        &self,
        graph: &mut Graph<R>,
        hidden: autograd::NodeId,
        tied_token_embedding: Option<autograd::NodeId>,
        dropout_probability: f32,
        training: bool,
        rng: &mut impl RngExt,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let dense = linear(graph, hidden, self.dense_w, self.dense_b)?;
        let activated = graph.gelu(dense)?;
        let activated = if training {
            graph.dropout(activated, dropout_probability, rng)?
        } else {
            activated
        };
        let normalized = graph.layer_norm(activated, self.gamma, self.beta, BERT_LAYER_NORM_EPS)?;

        let vocab_w = match (self.vocab_w, tied_token_embedding) {
            (Some(weight), _) => weight,
            (None, Some(table)) => graph.transpose(table)?,
            (None, None) => {
                return invalid(
                    "tied_token_embedding",
                    "is required when the MLM head uses tied embeddings",
                );
            }
        };

        let logits = graph.matmul(normalized, vocab_w)?;
        graph.sum(logits, self.vocab_b)
    }

    /// Returns the trainable MLM-head parameters.
    pub fn parameters(&self) -> Vec<autograd::NodeId> {
        let mut parameters = vec![self.dense_w, self.dense_b, self.gamma, self.beta];
        if let Some(vocab_w) = self.vocab_w {
            parameters.push(vocab_w);
        }
        parameters.push(self.vocab_b);
        parameters
    }
}

/// Output of a full BERT forward pass.
#[derive(Debug, Clone, Copy)]
pub struct BertOutput {
    /// Masked-language-model logits with shape `[seq_len, vocab_size]`.
    pub logits: autograd::NodeId,
}

/// Encoder-only BERT model.
pub struct Bert {
    /// Model hyperparameters.
    pub config: BertConfig,
    /// Token, position, and segment embeddings.
    pub embeddings: BertEmbeddings,
    /// Bidirectional encoder layers.
    pub layers: Vec<Encoder>,
    /// Masked-language-model head.
    pub mlm: BertMlmHead,
}

impl Bert {
    /// Builds a BERT model with randomly initialized parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or tensor allocation
    /// fails.
    pub fn new<R>(graph: &mut Graph<R>, device: &R::Device, config: BertConfig) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        config.validate()?;

        let embeddings = BertEmbeddings::new(
            graph,
            device,
            config.vocab_size,
            config.d_model,
            config.max_len,
        )?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(Encoder::new(
                graph,
                device,
                config.d_model,
                config.d_ff,
                config.n_heads,
            )?);
        }

        let mlm = BertMlmHead::new(
            graph,
            device,
            config.d_model,
            config.vocab_size,
            config.tie_word_embeddings,
        )?;

        Ok(Self {
            config,
            embeddings,
            layers,
            mlm,
        })
    }

    /// Runs the embedding stack and all encoder layers.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid or a tensor operation fails.
    pub fn encode<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
        token_type_ids: &[usize],
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let mut unused_rng = StdRng::seed_from_u64(0);
        self.encode_with_dropout(
            graph,
            device,
            token_ids,
            token_type_ids,
            false,
            &mut unused_rng,
        )
    }

    /// Runs the encoder with dropout enabled only when `training` is true.
    pub fn encode_with_dropout<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
        token_type_ids: &[usize],
        training: bool,
        rng: &mut impl RngExt,
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let mut hidden = self
            .embeddings
            .forward(graph, device, token_ids, token_type_ids)?;

        for layer in &self.layers {
            hidden = layer.forward_with_dropout(
                graph,
                device,
                hidden,
                self.config.dropout,
                training,
                rng,
            )?;
        }

        Ok(hidden)
    }

    /// Runs the full encoder and masked-language-model head.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid or a tensor operation fails.
    pub fn forward<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
        token_type_ids: &[usize],
    ) -> Result<BertOutput>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let mut unused_rng = StdRng::seed_from_u64(0);
        self.forward_with_dropout(
            graph,
            device,
            token_ids,
            token_type_ids,
            false,
            &mut unused_rng,
        )
    }

    /// Runs the full model with dropout controlled by `training`.
    pub fn forward_with_dropout<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
        token_type_ids: &[usize],
        training: bool,
        rng: &mut impl RngExt,
    ) -> Result<BertOutput>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let hidden =
            self.encode_with_dropout(graph, device, token_ids, token_type_ids, training, rng)?;
        let tied = if self.config.tie_word_embeddings {
            Some(self.embeddings.token.table)
        } else {
            None
        };
        let logits = self.mlm.forward_with_dropout(
            graph,
            hidden,
            tied,
            self.config.dropout,
            training,
            rng,
        )?;

        Ok(BertOutput { logits })
    }

    /// Returns every trainable parameter in deterministic order.
    ///
    /// Parameters are registered before any forward pass, so they occupy a
    /// contiguous prefix of the graph and can be preserved by
    /// [`Graph::truncate`](crate::autograd::Graph::truncate).
    pub fn parameters(&self) -> Vec<autograd::NodeId> {
        let mut parameters = self.embeddings.parameters();
        for layer in &self.layers {
            parameters.extend(layer.parameters());
        }
        parameters.extend(self.mlm.parameters());
        parameters
    }

    /// Returns the number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

fn invalid<T>(argument: &'static str, reason: impl Into<String>) -> Result<T> {
    Err(Error::InvalidArgument {
        arg: argument,
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use std::collections::HashSet;

    fn config(tie_word_embeddings: bool) -> BertConfig {
        BertConfig {
            vocab_size: 32,
            d_model: 8,
            d_ff: 16,
            n_heads: 2,
            n_layers: 2,
            max_len: 10,
            dropout: 0.0,
            tie_word_embeddings,
        }
    }

    fn build(tie_word_embeddings: bool) -> Result<(CpuDevice, Graph, Bert)> {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let model = Bert::new(&mut graph, &device, config(tie_word_embeddings))?;
        Ok((device, graph, model))
    }

    #[test]
    fn test_forward_shapes() -> Result<()> {
        let (device, mut graph, model) = build(true)?;
        let token_ids = [2, 5, 6, 3, 7, 3];
        let token_type_ids = [0, 0, 0, 0, 1, 1];

        let output = model.forward(&mut graph, &device, &token_ids, &token_type_ids)?;

        assert_eq!(graph.nodes[output.logits].data.shape(), &[6, 32]);
        Ok(())
    }

    #[test]
    fn test_tied_head_updates_token_embedding() -> Result<()> {
        let (device, mut graph, model) = build(true)?;
        assert!(model.mlm.vocab_w.is_none());

        let output = model.forward(&mut graph, &device, &[2, 5, 6, 3], &[0, 0, 0, 0])?;
        graph.backward_with_grad(output.logits, Tensor::ones(&[4, 32], DType::F32, &device)?)?;

        let gradient = graph.nodes[model.embeddings.token.table]
            .grad
            .to_vec::<f32>();
        assert!(
            gradient.iter().any(|value| value.abs() > 1e-8),
            "tied token embedding should receive a gradient"
        );
        Ok(())
    }

    #[test]
    fn test_untied_head_has_own_projection() -> Result<()> {
        let (_device, _graph, model) = build(false)?;
        assert!(model.mlm.vocab_w.is_some());
        let parameters = model.parameters();
        let unique: HashSet<_> = parameters.iter().copied().collect();
        assert_eq!(parameters.len(), unique.len());
        Ok(())
    }

    #[test]
    fn test_invalid_head_count_is_rejected() {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let mut invalid_config = config(true);
        invalid_config.n_heads = 3;

        assert!(Bert::new(&mut graph, &device, invalid_config).is_err());
    }

    #[test]
    fn test_sequence_longer_than_max_len_is_rejected() -> Result<()> {
        let (device, mut graph, model) = build(true)?;
        let token_ids = vec![5; 11];
        let token_type_ids = vec![0; 11];

        assert!(
            model
                .forward(&mut graph, &device, &token_ids, &token_type_ids)
                .is_err()
        );
        Ok(())
    }
}
