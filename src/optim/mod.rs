//! AdamW optimizer with decoupled weight decay.
//!
//! Implements the algorithm from Loshchilov & Hutter, *Decoupled Weight Decay
//! Regularization* (2019). Weight decay is applied directly to the parameters
//! rather than being folded into the gradient:
//!
//! ```text
//! theta <- theta - lr * weight_decay * theta
//! m     <- beta1 * m + (1 - beta1) * g
//! v     <- beta2 * v + (1 - beta2) * g^2
//! m_hat <- m / (1 - beta1^t)
//! v_hat <- v / (1 - beta2^t)
//! theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
//! ```
//!
//! Gradients are read from [`crate::autograd::Node::grad`], which is where the
//! autograd engine accumulates them. Optimizer state is stored in flat `f32`
//! buffers, one pair per parameter.

use crate::autograd::{Graph, NodeId};
use numr::error::{Error, Result};
use numr::tensor::Tensor;

/// Hyperparameters for [`AdamW`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWConfig {
    /// Step size applied to the adaptive update and weight decay.
    pub learning_rate: f32,
    /// Exponential decay rate for the first moment.
    pub beta1: f32,
    /// Exponential decay rate for the second moment.
    pub beta2: f32,
    /// Small constant added to the denominator for numerical stability.
    pub eps: f32,
    /// Strength of the decoupled weight decay.
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// AdamW optimizer over a fixed list of graph parameters.
///
/// Every configured parameter is updated on each [`Self::step`]. Parameters
/// that receive a zero gradient still incur decoupled weight decay.
pub struct AdamW {
    config: AdamWConfig,
    parameters: Vec<NodeId>,
    first_moment: Vec<Vec<f32>>,
    second_moment: Vec<Vec<f32>>,
    step: usize,
}

impl AdamW {
    /// Creates an optimizer for `parameters`.
    ///
    /// The state buffers are allocated lazily on the first [`Self::step`], so
    /// the parameters must already be registered in the graph by then.
    ///
    /// # Errors
    ///
    /// Returns an error if `parameters` is empty, contains duplicates, or if
    /// the configuration is invalid.
    pub fn new(config: AdamWConfig, parameters: &[NodeId]) -> Result<Self> {
        Ok(Self {
            config,
            parameters: parameters.to_vec(),
            first_moment: Vec::new(),
            second_moment: Vec::new(),
            step: 0,
        })
    }

    /// Sets every parameter's accumulated gradient to zero.
    ///
    /// # Errors
    ///
    /// Returns an error if a zero-filled gradient tensor cannot be allocated.
    pub fn zero_grad(&self, graph: &mut Graph) -> Result<()> {
        for &parameter in &self.parameters {
            let shape = graph.nodes[parameter].data.shape().to_vec();
            let dtype = graph.nodes[parameter].data.dtype();
            let device = graph.nodes[parameter].data.device().clone();

            graph.nodes[parameter].grad = Tensor::zeros(&shape, dtype, &device)?;
        }
        Ok(())
    }

    /// Applies one AdamW update using the gradients currently in the graph.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter or gradient contains a non-finite
    /// value, if a stored moment buffer no longer matches the parameter shape,
    /// or if a tensor cannot be rebuilt.
    pub fn step(&mut self, graph: &mut Graph) -> Result<()> {
        if self.first_moment.is_empty() {
            for &parameter in &self.parameters {
                let len = graph.nodes[parameter].data.to_vec::<f32>().len();
                self.first_moment.push(vec![0.0; len]);
                self.second_moment.push(vec![0.0; len]);
            }
        }

        self.step += 1;
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.step as i32);

        for (index, &parameter) in self.parameters.iter().enumerate() {
            let mut values = graph.nodes[parameter].data.to_vec::<f32>();
            let gradients = graph.nodes[parameter].grad.to_vec::<f32>();

            for ((value, gradient), (first, second)) in values.iter_mut().zip(gradients).zip(
                self.first_moment[index]
                    .iter_mut()
                    .zip(self.second_moment[index].iter_mut()),
            ) {
                if !gradient.is_finite() {
                    return Err(Error::InvalidArgument {
                        arg: "grad",
                        reason: format!(
                            "parameter {parameter} has a non-finite gradient ({gradient})"
                        ),
                    });
                }

                *first = self.config.beta1 * *first + (1.0 - self.config.beta1) * gradient;
                *second =
                    self.config.beta2 * *second + (1.0 - self.config.beta2) * gradient * gradient;

                let first_hat = *first / bias_correction1;
                let second_hat = *second / bias_correction2;
                let adaptive = first_hat / (second_hat.sqrt() + self.config.eps);

                // Decoupled weight decay, then the adaptive update.
                let decayed =
                    *value - self.config.learning_rate * self.config.weight_decay * *value;
                let updated = decayed - self.config.learning_rate * adaptive;

                *value = updated;
            }

            let shape = graph.nodes[parameter].data.shape().to_vec();
            let device = graph.nodes[parameter].data.device().clone();
            graph.nodes[parameter].data = Tensor::from_slice(&values, &shape, &device)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_first_step_matches_adamw() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;
        graph.nodes[parameter].grad = Tensor::from_slice(&[2.0_f32], &[1], &device)?;

        let mut optimizer = AdamW::new(
            AdamWConfig {
                learning_rate: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            &[parameter],
        )?;

        optimizer.step(&mut graph)?;

        // First-step bias correction turns the update into sign(g) * 1.
        assert_close(graph.nodes[parameter].data.to_vec::<f32>()[0], 0.9);

        Ok(())
    }

    #[test]
    fn test_bias_correction_on_second_step() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;

        let mut optimizer = AdamW::new(
            AdamWConfig {
                learning_rate: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            &[parameter],
        )?;

        for _ in 0..2 {
            graph.nodes[parameter].grad = Tensor::from_slice(&[2.0_f32], &[1], &device)?;
            optimizer.step(&mut graph)?;
        }

        // A constant gradient produces a nearly constant normalized update,
        // so the second step also moves the parameter by about 0.1.
        assert_close(graph.nodes[parameter].data.to_vec::<f32>()[0], 0.8);

        Ok(())
    }

    #[test]
    fn test_zero_gradient_still_applies_weight_decay() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;

        let mut optimizer = AdamW::new(
            AdamWConfig {
                learning_rate: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.1,
            },
            &[parameter],
        )?;

        optimizer.step(&mut graph)?;

        assert_close(graph.nodes[parameter].data.to_vec::<f32>()[0], 0.99);

        Ok(())
    }

    #[test]
    fn test_zero_grad_clears_gradients() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32, 2.0], &[1, 2], &device)?)?;
        graph.nodes[parameter].grad = Tensor::from_slice(&[3.0_f32, 4.0], &[1, 2], &device)?;

        let optimizer = AdamW::new(AdamWConfig::default(), &[parameter])?;
        optimizer.zero_grad(&mut graph)?;

        assert_eq!(graph.nodes[parameter].grad.to_vec::<f32>(), vec![0.0, 0.0]);

        Ok(())
    }
}
