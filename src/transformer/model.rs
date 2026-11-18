//! End-to-end transformer model wrapper.

use super::decoding::{DecodingConfig, SamplingStrategy, select_token};
use super::{Decoder, Encoder, OutputProjection, PositionalEncoding, TokenEmbedding};
use crate::autograd::{self, Graph};
use numr::error::Result;
use numr::runtime::cpu::CpuDevice;
use rand::RngExt;

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
    /// Start-of-sequence token used to seed generation.
    pub sos_token: usize,
    /// End-of-sequence token that stops generation.
    pub eos_token: usize,
}

/// Encoder-decoder transformer assembled from the project's layers.
///
/// The wrapper owns one encoder layer and one decoder layer, plus the shared
/// token embedding, sinusoidal positional encoding, and vocabulary projection.
/// It exposes teacher-forced logits for training and autoregressive generation
/// for inference.
pub struct Transformer {
    /// Shared token embedding table.
    pub embedding: TokenEmbedding,
    /// Fixed sinusoidal positional encoding.
    pub positional: PositionalEncoding,
    /// Encoder layer.
    pub encoder: Encoder,
    /// Decoder layer.
    pub decoder: Decoder,
    /// Vocabulary projection.
    pub output: OutputProjection,
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Start-of-sequence token used to seed generation.
    pub sos_token: usize,
    /// End-of-sequence token that stops generation.
    pub eos_token: usize,
}

impl Transformer {
    /// Builds a complete transformer with randomly initialized parameters.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - CPU device for tensor allocation
    /// * `config` - Model dimensions and vocabulary size
    ///
    /// # Errors
    ///
    /// Returns an error if any layer configuration is invalid, if a special
    /// token is outside the vocabulary, or if tensor allocation fails.
    pub fn new(graph: &mut Graph, device: &CpuDevice, config: TransformerConfig) -> Result<Self> {
        let embedding = TokenEmbedding::new(graph, device, config.vocab_size, config.d_model)?;
        let positional = PositionalEncoding::new(device, config.max_len, config.d_model)?;
        let encoder = Encoder::new(graph, device, config.d_model, config.d_ff, config.n_heads)?;
        let decoder = Decoder::new(graph, device, config.d_model, config.d_ff, config.n_heads)?;
        let output = OutputProjection::new(graph, device, config.d_model, config.vocab_size)?;

        Ok(Self {
            embedding,
            positional,
            encoder,
            decoder,
            output,
            vocab_size: config.vocab_size,
            sos_token: config.sos_token,
            eos_token: config.eos_token,
        })
    }

    /// Returns every trainable parameter node, ordered deterministically.
    ///
    /// The parameters are registered when the model is constructed, so they
    /// occupy a contiguous prefix of the graph. [`Graph::truncate`] can
    /// therefore free per-step intermediates by keeping only
    /// `graph.nodes.len()` nodes captured immediately after construction.
    pub fn parameters(&self) -> Vec<autograd::NodeId> {
        let mut params = self.embedding.parameters();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params.extend(self.output.parameters());
        params
    }

    /// Encodes token IDs into contextual representations.
    ///
    /// # Returns
    ///
    /// Node ID of encoder output with shape `[seq_len, d_model]`.
    pub fn encode(
        &self,
        graph: &mut Graph,
        device: &CpuDevice,
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
        graph: &mut Graph,
        device: &CpuDevice,
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

    /// Generates decoder tokens autoregressively.
    ///
    /// Generation starts from the model's `sos_token` and repeatedly runs the
    /// decoder on the growing prefix, selecting the next token with the
    /// configured strategy. Generation stops when `eos_token` is selected or
    /// when `max_new_tokens` or the positional-encoding capacity is reached.
    ///
    /// The returned vector contains content tokens only; `sos_token` and
    /// `eos_token` are not included. The decoder is rerun from scratch for each
    /// step, so no key/value cache is used. This is intentionally simple for
    /// the toy examples.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoding strategy is invalid or if an
    /// encoder/decoder operation fails.
    pub fn generate(
        &self,
        graph: &mut Graph,
        device: &CpuDevice,
        encoder_token_ids: &[usize],
        config: &DecodingConfig,
        rng: &mut impl RngExt,
    ) -> Result<Vec<usize>> {
        let encoder_output = self.encode(graph, device, encoder_token_ids)?;
        let mut decoder_tokens = vec![self.sos_token];
        let mut generated = Vec::new();

        for _ in 0..config.max_new_tokens {
            let logits = self.decode_logits(graph, device, &decoder_tokens, encoder_output)?;

            let shape = graph.nodes[logits].data.shape();

            let values = graph.nodes[logits].data.to_vec::<f32>();
            let start = (shape[0] - 1) * self.vocab_size;
            let last_logits = &values[start..start + self.vocab_size];

            let uniform = match config.strategy {
                SamplingStrategy::Greedy => 0.0,
                SamplingStrategy::TopP { .. } => rng.random::<f32>(),
            };
            let next_token = select_token(last_logits, config.strategy, uniform)?;

            if next_token == self.eos_token {
                break;
            }

            generated.push(next_token);
            decoder_tokens.push(next_token);

            // The next step would exceed the positional encoding capacity.
            if decoder_tokens.len() > self.positional.max_len {
                break;
            }
        }

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 10,
            d_model: 8,
            d_ff: 16,
            n_heads: 2,
            max_len: 8,
            sos_token: 1,
            eos_token: 2,
        }
    }

    fn build_model() -> Result<(CpuDevice, Graph, Transformer)> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
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
    fn test_generate_greedy_and_top_p() -> Result<()> {
        let (device, mut graph, model) = build_model()?;
        let mut rng = StdRng::seed_from_u64(7);

        let greedy = model.generate(
            &mut graph,
            &device,
            &[3, 4, 5, 6],
            &DecodingConfig {
                max_new_tokens: 4,
                strategy: SamplingStrategy::Greedy,
            },
            &mut rng,
        )?;
        assert!(greedy.len() <= 4);
        assert!(greedy.iter().all(|&token| token < model.vocab_size));

        let top_p = model.generate(
            &mut graph,
            &device,
            &[3, 4, 5, 6],
            &DecodingConfig {
                max_new_tokens: 4,
                strategy: SamplingStrategy::TopP {
                    p: 0.9,
                    temperature: 0.8,
                },
            },
            &mut rng,
        )?;
        assert!(top_p.len() <= 4);
        assert!(top_p.iter().all(|&token| token < model.vocab_size));

        Ok(())
    }

    #[test]
    fn test_generate_is_bounded_by_positional_capacity() -> Result<()> {
        let (device, mut graph, model) = build_model()?;
        let mut rng = StdRng::seed_from_u64(11);

        let generated = model.generate(
            &mut graph,
            &device,
            &[3, 4],
            &DecodingConfig {
                max_new_tokens: 100,
                strategy: SamplingStrategy::Greedy,
            },
            &mut rng,
        )?;

        assert!(generated.len() <= model.positional.max_len);

        Ok(())
    }

    #[test]
    fn test_backward_reaches_embedding_and_output_head() -> Result<()> {
        let (device, mut graph, model) = build_model()?;

        let encoded = model.encode(&mut graph, &device, &[3, 4, 5])?;
        let logits = model.decode_logits(&mut graph, &device, &[1, 3, 4], encoded)?;
        let loss = graph.mul(logits, logits)?;
        graph.backward(loss)?;

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
