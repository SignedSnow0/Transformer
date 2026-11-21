//! End-to-end transformer model wrapper.

use super::output::OutputProjection;
use super::positional::PositionalEncoding;
use super::{Decoder, Encoder, TokenEmbedding};
use crate::autograd::{self, Graph};
use numr::prelude::*;
use numr::runtime::cpu::CpuRuntime;

/// Hyperparameters for the toy transformer.
#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Model/embedding dimension.
    pub d_model: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Number of attention heads; must divide `d_model`.
    pub n_heads: usize,
    /// Maximum sequence length supported by the positional encoding.
    pub max_len: usize,
}

/// Encoder-decoder transformer assembled from the project's layers.
///
/// The wrapper owns one encoder layer and one decoder layer, plus the shared
/// token embedding, sinusoidal positional encoding, and vocabulary projection.
/// It exposes teacher-forced logits for training.
pub struct Transformer<R: Runtime = CpuRuntime> {
    /// Shared token embedding table.
    pub embedding: TokenEmbedding,
    /// Fixed sinusoidal positional encoding.
    pub positional: PositionalEncoding<R>,
    /// Encoder layer.
    pub encoder: Encoder,
    /// Decoder layer.
    pub decoder: Decoder,
    /// Vocabulary projection.
    pub output: OutputProjection,
}

impl<R> Transformer<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    /// Builds a complete transformer with randomly initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - Device of the selected runtime for tensor allocation
    /// * `config` - Model dimensions and vocabulary size
    ///
    /// # Errors
    ///
    /// Returns an error if any layer configuration is invalid or tensor
    /// allocation fails.
    pub fn new(
        graph: &mut Graph<R>,
        device: &R::Device,
        config: TransformerConfig,
    ) -> Result<Self> {
        let embedding = TokenEmbedding::new(graph, device, config.vocab_size, config.d_model)?;
        let positional = PositionalEncoding::<R>::new(device, config.max_len, config.d_model)?;
        let encoder = Encoder::new(graph, device, config.d_model, config.d_ff, config.n_heads)?;
        let decoder = Decoder::new(graph, device, config.d_model, config.d_ff, config.n_heads)?;
        let output = OutputProjection::new(graph, device, config.d_model, config.vocab_size)?;

        Ok(Self {
            embedding,
            positional,
            encoder,
            decoder,
            output,
        })
    }

    /// Encodes token IDs into contextual representations.
    ///
    /// # Returns
    ///
    /// Node ID of encoder output with shape `[seq_len, d_model]`.
    pub fn encode(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
    ) -> Result<autograd::NodeId> {
        let embedded = self.embedding.forward(graph, device, token_ids)?;
        let positioned = self.positional.forward(graph, embedded)?;
        self.encoder.forward(graph, device, positioned)
    }

    /// Computes next-token logits for a decoder prefix.
    ///
    /// This is the teacher-forced path: pass the shifted target sequence and
    /// apply a cross-entropy loss to the returned logits.
    ///
    /// # Returns
    ///
    /// Node ID of logits with shape `[decoder_len, vocab_size]`.
    pub fn decode_logits(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        decoder_token_ids: &[usize],
        encoder_output: autograd::NodeId,
    ) -> Result<autograd::NodeId> {
        let embedded = self.embedding.forward(graph, device, decoder_token_ids)?;
        let positioned = self.positional.forward(graph, embedded)?;
        let hidden = self
            .decoder
            .forward(graph, device, positioned, encoder_output)?;
        self.output.forward(graph, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 10,
            d_model: 8,
            d_ff: 16,
            n_heads: 2,
            max_len: 8,
        }
    }

    fn build_model() -> Result<(CpuDevice, Graph, Transformer)> {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let model = Transformer::new(&mut graph, &device, config())?;
        Ok((device, graph, model))
    }

    #[test]
    fn test_encode_and_decode_logits_shapes() -> Result<()> {
        let (device, mut graph, model) = build_model()?;

        let encoded = model.encode(&mut graph, &device, &[3, 4, 5, 6])?;
        assert_eq!(graph.nodes[encoded].data.shape(), &[4, 8]);

        let logits = model.decode_logits(&mut graph, &device, &[1, 3, 4], encoded)?;
        assert_eq!(graph.nodes[logits].data.shape(), &[3, 10]);

        Ok(())
    }

    #[test]
    fn test_backward_reaches_embedding_and_output_head() -> Result<()> {
        let (device, mut graph, model) = build_model()?;

        let encoded = model.encode(&mut graph, &device, &[3, 4, 5])?;
        let logits = model.decode_logits(&mut graph, &device, &[1, 3, 4], encoded)?;
        let loss = graph.mul(logits, logits)?;
        graph.backward_with_grad(loss, Tensor::ones(&[3, 10], DType::F32, &device)?)?;

        assert!(
            graph.nodes[model.embedding.table]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&gradient| gradient.abs() > 1e-8)
        );
        assert!(
            graph.nodes[model.output.w]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&gradient| gradient.abs() > 1e-8)
        );

        Ok(())
    }
}
