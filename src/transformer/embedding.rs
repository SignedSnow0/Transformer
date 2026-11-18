//! Learned token embeddings.
//!
//! Token IDs are discrete inputs, so they carry no gradient of their own. The
//! embedding table is the trainable parameter: every lookup selects one row of
//! that table, and the backward pass accumulates the upstream gradient into the
//! selected rows.

use super::normal_init;
use crate::autograd::{self, Graph};
use numr::{
    error::{Error, Result},
    runtime::cpu::CpuDevice,
    tensor::Tensor,
};

/// Learned embedding table mapping integer token IDs to dense vectors.
///
/// The table has shape `[vocab_size, d_model]` and is a trainable graph
/// variable. A lookup of `n` token IDs returns a `[n, d_model]` tensor.
///
/// Lookups are implemented as a one-hot matrix multiplication against the
/// table. This keeps the operation inside the existing autograd primitives:
/// the matrix-multiplication backward pass already computes the correct table
/// gradient, including summing the gradients of repeated token IDs into the
/// same row. For the small vocabularies used by the toy examples this is
/// inexpensive; a dedicated lookup/scatter-add operation would be preferable
/// for large vocabularies.
pub struct TokenEmbedding {
    /// Trainable table node with shape `[vocab_size, d_model]`.
    pub table: autograd::NodeId,
    /// Number of rows in the embedding table.
    pub vocab_size: usize,
    /// Width of each embedding vector.
    pub d_model: usize,
}

impl TokenEmbedding {
    /// Creates a randomly initialized embedding table.
    ///
    /// Rows are initialized from a normal distribution with standard deviation
    /// `1 / sqrt(d_model)`.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register the table in
    /// * `device` - CPU device for tensor allocation
    /// * `vocab_size` - Number of tokens in the vocabulary
    /// * `d_model` - Embedding dimension
    ///
    /// # Errors
    ///
    /// Returns an error if `vocab_size` or `d_model` is zero, if `vocab_size`
    /// exceeds `i32::MAX`, or if tensor allocation fails.
    pub fn new(
        graph: &mut Graph,
        device: &CpuDevice,
        vocab_size: usize,
        d_model: usize,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "vocab_size",
                reason: "vocab_size must be greater than zero".to_string(),
            });
        }
        if vocab_size > i32::MAX as usize {
            return Err(Error::InvalidArgument {
                arg: "vocab_size",
                reason: format!("vocab_size must fit in an i32, got {vocab_size}"),
            });
        }
        if d_model == 0 {
            return Err(Error::InvalidArgument {
                arg: "d_model",
                reason: "d_model must be greater than zero".to_string(),
            });
        }

        let std = 1.0 / (d_model as f32).sqrt();
        let table = graph.variable(Tensor::from_slice(
            &normal_init(vocab_size * d_model, std),
            &[vocab_size, d_model],
            device,
        )?)?;

        Ok(Self {
            table,
            vocab_size,
            d_model,
        })
    }

    /// Looks up token IDs and returns their embeddings.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build the lookup in
    /// * `device` - CPU device for the temporary one-hot tensor
    /// * `token_ids` - Token indices, each in `0..vocab_size`
    ///
    /// # Returns
    ///
    /// Node ID of a tensor with shape `[token_ids.len(), d_model]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `token_ids` is empty, if any ID is out of range, or
    /// if a tensor operation fails.
    pub fn forward(
        &self,
        graph: &mut Graph,
        device: &CpuDevice,
        token_ids: &[usize],
    ) -> Result<autograd::NodeId> {
        if token_ids.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: "embedding lookup requires at least one token".to_string(),
            });
        }

        let table_shape = graph.nodes[self.table].data.shape();
        if table_shape != [self.vocab_size, self.d_model] {
            return Err(Error::InvalidArgument {
                arg: "table",
                reason: format!(
                    "expected embedding table shape [{}, {}], got {:?}",
                    self.vocab_size, self.d_model, table_shape
                ),
            });
        }

        for (position, &id) in token_ids.iter().enumerate() {
            if id >= self.vocab_size {
                return Err(Error::InvalidArgument {
                    arg: "token_ids",
                    reason: format!(
                        "token id {id} at position {position} is out of range for vocabulary size {}",
                        self.vocab_size
                    ),
                });
            }
        }

        let indices: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let indices = Tensor::from_slice(&indices, &[token_ids.len()], device)?;
        let one_hot = indices.one_hot(self.vocab_size)?;
        let one_hot = graph.variable(one_hot)?;

        graph.matmul(one_hot, self.table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::Encoder;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_embedding_output_shape() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let embedding = TokenEmbedding::new(&mut graph, &device, 5, 4)?;

        let out = embedding.forward(&mut graph, &device, &[0, 2, 4])?;

        assert_eq!(graph.nodes[out].data.shape(), &[3, 4]);

        Ok(())
    }

    #[test]
    fn test_embedding_matches_table_rows() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let d_model = 4;
        let embedding = TokenEmbedding::new(&mut graph, &device, 5, d_model)?;

        let token_ids = [2_usize, 0, 2];
        let out = embedding.forward(&mut graph, &device, &token_ids)?;

        let table = graph.nodes[embedding.table].data.to_vec::<f32>();
        let embedded = graph.nodes[out].data.to_vec::<f32>();

        for (row, &id) in token_ids.iter().enumerate() {
            for column in 0..d_model {
                let expected = table[id * d_model + column];
                let actual = embedded[row * d_model + column];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "mismatch at row {row}, column {column}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_embedding_accumulates_gradient_for_repeated_tokens() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let embedding = TokenEmbedding::new(&mut graph, &device, 3, 2)?;

        // Token 1 appears twice, so its row receives two units of gradient.
        let out = embedding.forward(&mut graph, &device, &[1, 1])?;
        graph.backward(out)?;

        let grad = graph.nodes[embedding.table].grad.to_vec::<f32>();
        assert_eq!(grad, vec![0.0, 0.0, 2.0, 2.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_embedding_rejects_invalid_token() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let embedding = TokenEmbedding::new(&mut graph, &device, 5, 4)?;

        assert!(embedding.forward(&mut graph, &device, &[0, 5]).is_err());

        Ok(())
    }

    #[test]
    fn test_embedding_rejects_empty_sequence() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let embedding = TokenEmbedding::new(&mut graph, &device, 5, 4)?;

        assert!(embedding.forward(&mut graph, &device, &[]).is_err());

        Ok(())
    }

    #[test]
    fn test_embedding_feeds_encoder_and_receives_gradient() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let d_model = 8;
        let embedding = TokenEmbedding::new(&mut graph, &device, 10, d_model)?;
        let encoder = Encoder::new(&mut graph, &device, d_model, 16, 2)?;

        let token_ids: Vec<usize> = (0..4).collect();
        let embedded = embedding.forward(&mut graph, &device, &token_ids)?;
        let encoded = encoder.forward(&mut graph, &device, embedded)?;
        let loss = graph.mul(encoded, encoded)?;

        graph.backward(loss)?;

        assert_eq!(graph.nodes[encoded].data.shape(), &[4, d_model]);
        assert!(
            graph.nodes[embedding.table]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&gradient| gradient.abs() > 1e-8),
            "embedding table did not receive a gradient through the encoder"
        );

        Ok(())
    }
}
