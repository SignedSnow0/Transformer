//! Fixed sinusoidal positional encoding.
//!
//! Self-attention is permutation-invariant, so token order must be injected
//! explicitly. This module implements the sinusoidal encoding from Vaswani et
//! al. (2017), section 3.5:
//!
//! ```text
//! PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
//! PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
//! ```
//!
//! The table is precomputed once for the maximum sequence length and is not
//! trainable.

use crate::autograd::{self, Graph};
use numr::{
    error::{Error, Result},
    runtime::cpu::{CpuDevice, CpuRuntime},
    tensor::Tensor,
};

/// Sinusoidal positional encoding for sequence-major inputs.
///
/// The encoding table has shape `[max_len, d_model]`. [`Self::forward`] adds
/// the first `seq_len` rows to an input of shape `[seq_len, d_model]`.
///
/// The table is fixed rather than learned, so it has no trainable parameters.
/// Dropout after the addition is omitted because this implementation only
/// performs inference.
pub struct PositionalEncoding {
    table: Tensor<CpuRuntime>,
    /// Maximum number of positions the table can represent.
    pub max_len: usize,
    /// Width of each positional vector.
    pub d_model: usize,
}

impl PositionalEncoding {
    /// Builds a sinusoidal encoding table for positions `0..max_len`.
    ///
    /// # Arguments
    ///
    /// * `device` - CPU device for tensor allocation
    /// * `max_len` - Maximum supported sequence length
    /// * `d_model` - Model dimension; must match the token embeddings
    ///
    /// # Errors
    ///
    /// Returns an error if `max_len` or `d_model` is zero, or if tensor
    /// allocation fails.
    pub fn new(device: &CpuDevice, max_len: usize, d_model: usize) -> Result<Self> {
        if max_len == 0 {
            return Err(Error::InvalidArgument {
                arg: "max_len",
                reason: "max_len must be greater than zero".to_string(),
            });
        }
        if d_model == 0 {
            return Err(Error::InvalidArgument {
                arg: "d_model",
                reason: "d_model must be greater than zero".to_string(),
            });
        }

        let mut values = vec![0.0_f32; max_len * d_model];

        for position in 0..max_len {
            for index in 0..d_model {
                let pair = (index / 2) as f32;
                let angle = position as f32 / 10000.0_f32.powf(2.0 * pair / d_model as f32);
                values[position * d_model + index] = if index % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }

        let table = Tensor::from_slice(&values, &[max_len, d_model], device)?;

        Ok(Self {
            table,
            max_len,
            d_model,
        })
    }

    /// Adds positional information to an input tensor.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build the addition in
    /// * `x` - Input of shape `[seq_len, d_model]`, typically token embeddings
    ///
    /// # Returns
    ///
    /// Node ID of a tensor with the same shape as `x`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is not 2-D, if its feature dimension does not
    /// match `d_model`, if it is empty, or if it is longer than `max_len`.
    pub fn forward(&self, graph: &mut Graph, x: autograd::NodeId) -> Result<autograd::NodeId> {
        let shape = graph.nodes[x].data.shape();

        if shape.len() != 2 || shape[1] != self.d_model {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected shape [seq_len, {}], got {:?}",
                    self.d_model, shape
                ),
            });
        }

        let seq_len = shape[0];
        if seq_len == 0 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: "positional encoding requires at least one position".to_string(),
            });
        }
        if seq_len > self.max_len {
            return Err(Error::InvalidArgument {
                arg: "seq_len",
                reason: format!(
                    "sequence length {seq_len} exceeds positional encoding capacity {}",
                    self.max_len
                ),
            });
        }

        let rows = self.table.narrow(0, 0, seq_len)?.contiguous()?;
        let rows = graph.variable(rows)?;

        graph.sum(x, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::{Encoder, TokenEmbedding};
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_positional_encoding_values() -> Result<()> {
        let device = CpuDevice::new();
        let encoding = PositionalEncoding::new(&device, 2, 4)?;
        let table = encoding.table.to_vec::<f32>();

        // Position 0: sin(0) = 0, cos(0) = 1 for every frequency pair.
        assert!((table[0] - 0.0).abs() < 1e-6);
        assert!((table[1] - 1.0).abs() < 1e-6);
        assert!((table[2] - 0.0).abs() < 1e-6);
        assert!((table[3] - 1.0).abs() < 1e-6);

        // Position 1, first pair: sin(1), cos(1).
        assert!((table[4] - 1.0_f32.sin()).abs() < 1e-6);
        assert!((table[5] - 1.0_f32.cos()).abs() < 1e-6);

        // Position 1, second pair uses frequency 10000^(2/4) = 100.
        let angle = 1.0_f32 / 100.0;
        assert!((table[6] - angle.sin()).abs() < 1e-6);
        assert!((table[7] - angle.cos()).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_positional_encoding_forward_adds_rows() -> Result<()> {
        let device = CpuDevice::new();
        let encoding = PositionalEncoding::new(&device, 3, 4)?;
        let mut graph = Graph::new();
        let x = graph.variable(Tensor::zeros(&[2, 4], DType::F32, &device)?)?;

        let out = encoding.forward(&mut graph, x)?;

        assert_eq!(graph.nodes[out].data.shape(), &[2, 4]);

        let expected = encoding.table.narrow(0, 0, 2)?.contiguous()?;
        let expected = expected.to_vec::<f32>();
        let actual = graph.nodes[out].data.to_vec::<f32>();
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!((expected - actual).abs() < 1e-6);
        }

        // With a unit seed, the identity path receives a gradient of ones.
        graph.backward(out)?;
        assert_eq!(graph.nodes[x].grad.to_vec::<f32>(), vec![1.0; 8]);

        Ok(())
    }

    #[test]
    fn test_positional_encoding_rejects_too_long_sequence() -> Result<()> {
        let device = CpuDevice::new();
        let encoding = PositionalEncoding::new(&device, 2, 4)?;
        let mut graph = Graph::new();
        let x = graph.variable(Tensor::zeros(&[3, 4], DType::F32, &device)?)?;

        assert!(encoding.forward(&mut graph, x).is_err());

        Ok(())
    }

    #[test]
    fn test_positional_encoding_rejects_wrong_dimension() -> Result<()> {
        let device = CpuDevice::new();
        let encoding = PositionalEncoding::new(&device, 2, 4)?;
        let mut graph = Graph::new();
        let x = graph.variable(Tensor::zeros(&[2, 5], DType::F32, &device)?)?;

        assert!(encoding.forward(&mut graph, x).is_err());

        Ok(())
    }

    #[test]
    fn test_embedding_and_positional_encoding_feed_encoder() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let d_model = 8;
        let seq_len = 4;
        let embedding = TokenEmbedding::new(&mut graph, &device, 10, d_model)?;
        let positional = PositionalEncoding::new(&device, seq_len, d_model)?;
        let encoder = Encoder::new(&mut graph, &device, d_model, 16, 2)?;

        let token_ids: Vec<usize> = (0..seq_len).collect();
        let embedded = embedding.forward(&mut graph, &device, &token_ids)?;
        let positioned = positional.forward(&mut graph, embedded)?;
        let encoded = encoder.forward(&mut graph, &device, positioned)?;
        let loss = graph.mul(encoded, encoded)?;

        graph.backward(loss)?;

        assert_eq!(graph.nodes[encoded].data.shape(), &[seq_len, d_model]);
        assert!(
            graph.nodes[embedding.table]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&gradient| gradient.abs() > 1e-8),
            "embedding table did not receive a gradient through positional encoding and encoder"
        );

        Ok(())
    }
}
