//! Vocabulary projection for decoder outputs.

use super::{kaiming_init, linear};
use crate::autograd::{self, Graph};
use numr::prelude::*;

/// Projects decoder hidden states to vocabulary logits.
///
/// The projection is a linear layer with weight `[d_model, vocab_size]` and
/// bias `[1, vocab_size]`. Applying [`Graph::softmax`] to the returned logits
/// yields next-token probabilities.
///
/// The weight is initialized with Kaiming normal initialization. Weight tying
/// with [`super::TokenEmbedding`] is not performed here; the two tables are
/// independent parameters.
pub struct OutputProjection {
    /// Weight node with shape `[d_model, vocab_size]`.
    pub w: autograd::NodeId,
    /// Bias node with shape `[1, vocab_size]`.
    pub b: autograd::NodeId,
}

impl OutputProjection {
    /// Creates a randomly initialized vocabulary projection.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to register parameters in
    /// * `device` - Device for tensor allocation
    /// * `d_model` - Decoder hidden-state dimension
    /// * `vocab_size` - Number of output tokens
    ///
    /// # Errors
    ///
    /// Returns an error if `d_model` or `vocab_size` is zero, or if tensor
    /// allocation fails.
    pub fn new<R>(
        graph: &mut Graph<R>,
        device: &R::Device,
        d_model: usize,
        vocab_size: usize,
    ) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
    {
        let w = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * vocab_size, d_model),
            &[d_model, vocab_size],
            device,
        )?)?;
        let b = graph.variable(Tensor::zeros(&[1, vocab_size], DType::F32, device)?)?;

        Ok(Self { w, b })
    }

    /// Maps hidden states to vocabulary logits.
    ///
    /// # Arguments
    ///
    /// * `graph` - Autograd graph to build the projection in
    /// * `x` - Decoder hidden states with shape `[seq_len, d_model]`
    ///
    /// # Returns
    ///
    /// Node ID of logits with shape `[seq_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is not 2-D, if its feature dimension does not
    /// match `d_model`, or if a tensor operation fails.
    pub fn forward<R>(&self, graph: &mut Graph<R>, x: autograd::NodeId) -> Result<autograd::NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
    {
        let logits = linear(graph, x, self.w, self.b)?;

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TokenEmbedding;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_output_projection_shape() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let output = OutputProjection::new(&mut graph, &device, 8, 10)?;
        let x = graph.variable(Tensor::zeros(&[3, 8], DType::F32, &device)?)?;

        let logits = output.forward(&mut graph, x)?;

        assert_eq!(graph.nodes[logits].data.shape(), &[3, 10]);

        Ok(())
    }

    #[test]
    fn test_output_projection_backward_reaches_weights() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let d_model = 4;
        let vocab_size = 6;
        let output = OutputProjection::new(&mut graph, &device, d_model, vocab_size)?;
        let embedding = TokenEmbedding::new(&mut graph, &device, vocab_size, d_model)?;

        let embedded = embedding.forward(&mut graph, &device, &[0, 1, 2])?;
        let logits = output.forward(&mut graph, embedded)?;
        let loss = graph.mul(logits, logits)?;
        graph.backward_with_grad(loss, Tensor::ones(&[3, vocab_size], DType::F32, &device)?)?;

        assert!(
            graph.nodes[output.w]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&gradient| gradient.abs() > 1e-8)
        );
        assert!(
            graph.nodes[output.b]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&gradient| gradient.abs() > 1e-8)
        );

        Ok(())
    }
}
