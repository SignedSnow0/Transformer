//! Learned token embeddings.
//!
//! Token IDs are discrete inputs, so they carry no gradient of their own. The
//! embedding table is the trainable parameter: every lookup selects one row of
//! that table, and the backward pass accumulates the upstream gradient into the
//! selected rows.

use super::normal_init;
use crate::autograd::{self, Graph};
use numr::prelude::*;

/// Learned embedding table mapping integer token IDs to dense vectors.
///
/// The table has shape `[vocab_size, d_model]` and is a trainable graph
/// variable. A lookup of `n` token IDs returns a `[n, d_model]` tensor.
///
/// Lookups use the backend's embedding-gather operation. The backward pass
/// scatters gradients into the selected rows and sums repeated IDs, avoiding a
/// dense `[n, vocab_size]` one-hot matrix on every forward pass.
pub struct TokenEmbedding {
    /// Trainable table node with shape `[vocab_size, d_model]`.
    pub table: autograd::NodeId,
    /// Number of rows in the embedding table.
    pub vocab_size: usize,
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
    /// * `device` - Device for tensor allocation
    /// * `vocab_size` - Number of tokens in the vocabulary
    /// * `d_model` - Embedding dimension
    ///
    /// # Errors
    ///
    /// Returns an error if `vocab_size` or `d_model` is zero, if `vocab_size`
    /// exceeds `i32::MAX`, or if tensor allocation fails.
    pub fn new<R>(
        graph: &mut Graph<R>,
        device: &R::Device,
        vocab_size: usize,
        d_model: usize,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let std = 1.0 / (d_model as f32).sqrt();
        let table = graph.variable(Tensor::from_slice(
            &normal_init(vocab_size * d_model, std),
            &[vocab_size, d_model],
            device,
        )?)?;

        Ok(Self { table, vocab_size })
    }

    /// Looks up token IDs and returns their embeddings.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build the lookup in
    /// * `device` - Device for the temporary index tensors
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
    pub fn forward<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        token_ids: &[usize],
    ) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        if token_ids.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: "must not be empty".to_string(),
            });
        }
        if token_ids.iter().any(|&id| id >= self.vocab_size) {
            return Err(Error::InvalidArgument {
                arg: "token_ids",
                reason: "contains an ID outside the vocabulary".to_string(),
            });
        }

        let d_model = graph.nodes[self.table].data.shape()[1];
        let indices_1d: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
        let indices_2d: Vec<i64> = token_ids
            .iter()
            .flat_map(|&id| std::iter::repeat_n(id as i64, d_model))
            .collect();

        let indices_1d =
            graph.variable(Tensor::from_slice(&indices_1d, &[token_ids.len()], device)?)?;
        let indices_2d = graph.variable(Tensor::from_slice(
            &indices_2d,
            &[token_ids.len(), d_model],
            device,
        )?)?;

        graph.embedding_lookup(self.table, indices_1d, indices_2d)
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
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let embedding = TokenEmbedding::new(&mut graph, &device, 5, 4)?;

        let out = embedding.forward(&mut graph, &device, &[0, 2, 4])?;

        assert_eq!(graph.nodes[out].data.shape(), &[3, 4]);

        Ok(())
    }

    #[test]
    fn test_embedding_matches_table_rows() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
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
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let embedding = TokenEmbedding::new(&mut graph, &device, 3, 2)?;

        // Token 1 appears twice, so its row receives two units of gradient.
        let out = embedding.forward(&mut graph, &device, &[1, 1])?;
        graph.backward_with_grad(out, Tensor::ones(&[2, 2], DType::F32, &device)?)?;

        let grad = graph.nodes[embedding.table].grad.to_vec::<f32>();
        assert_eq!(grad, vec![0.0, 0.0, 2.0, 2.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_embedding_feeds_encoder_and_receives_gradient() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let d_model = 8;
        let embedding = TokenEmbedding::new(&mut graph, &device, 10, d_model)?;
        let encoder = Encoder::new(&mut graph, &device, d_model, 16, 2)?;

        let token_ids: Vec<usize> = (0..4).collect();
        let embedded = embedding.forward(&mut graph, &device, &token_ids)?;
        let encoded = encoder.forward(&mut graph, &device, embedded)?;
        let loss = graph.mul(encoded, encoded)?;

        graph.backward_with_grad(loss, Tensor::ones(&[4, d_model], DType::F32, &device)?)?;

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
