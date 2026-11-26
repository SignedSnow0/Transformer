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
//! autograd engine accumulates them.
//!
//! The optimizer keeps its entire state on the selected runtime. Parameters are
//! flattened into one contiguous vector, gradients are concatenated once per
//! step, and the update is a short sequence of element-wise kernels over that
//! vector. This launches a handful of kernels per step instead of a handful per
//! parameter, and it avoids copying parameters or gradients to the host.

use crate::autograd::{Graph, NodeId};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::prelude::*;
use std::collections::HashSet;

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

/// Warmup plus cosine decay for the AdamW learning rate.
///
/// Step `1` starts at `peak / warmup_steps` (or at `peak` when there is no
/// warmup), rises linearly to `peak` by the end of warmup, then decays with a
/// cosine curve to `peak * min_lr_ratio` on the final step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LRSchedule {
    /// Number of linear warmup steps.
    pub warmup_steps: usize,
    /// Learning rate at the end of training, as a fraction of the peak.
    pub min_lr_ratio: f32,
}

impl Default for LRSchedule {
    fn default() -> Self {
        Self {
            warmup_steps: 0,
            min_lr_ratio: 0.1,
        }
    }
}

impl LRSchedule {
    /// Checks that the schedule's ratio is a valid probability.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_lr_ratio` is outside `[0, 1]` or is not finite.
    pub fn validate(&self) -> Result<()> {
        if !self.min_lr_ratio.is_finite() || !(0.0..=1.0).contains(&self.min_lr_ratio) {
            return invalid("min_lr_ratio", "must be finite and in [0, 1]");
        }
        Ok(())
    }

    /// Returns the learning rate for a one-based optimizer step.
    ///
    /// `total_steps` and `step` are clamped into a valid range, so calling this
    /// with `step == 0` produces the first warmup value rather than zero.
    pub fn learning_rate(&self, peak: f32, step: usize, total_steps: usize) -> f32 {
        if !peak.is_finite() || peak <= 0.0 {
            return peak.max(0.0);
        }

        let step = step.max(1);
        let total_steps = total_steps.max(step);
        let warmup = self.warmup_steps.min(total_steps.saturating_sub(1));
        let min_ratio = self.min_lr_ratio.clamp(0.0, 1.0);

        if warmup > 0 && step <= warmup {
            return peak * step as f32 / warmup as f32;
        }

        if total_steps <= warmup + 1 {
            return peak;
        }

        let progress = (step - warmup) as f32 / (total_steps - warmup) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress.min(1.0)).cos());
        peak * (min_ratio + (1.0 - min_ratio) * cosine)
    }
}

/// AdamW optimizer over a fixed list of graph parameters.
///
/// Every configured parameter is updated on each [`Self::step`]. Parameters
/// that receive a zero gradient still incur decoupled weight decay.
///
/// The optimizer is generic over the runtime, so its moment buffers live on the
/// same device as the model. State is allocated lazily on the first step, when
/// the graph can supply parameter shapes and the device.
pub struct AdamW<R: Runtime> {
    config: AdamWConfig,
    parameters: Vec<NodeId>,
    state: Option<AdamWState<R>>,
    step: usize,
}

struct AdamWState<R: Runtime> {
    shapes: Vec<Vec<usize>>,
    lengths: Vec<usize>,
    parameters: Tensor<R>,
    first_moment: Tensor<R>,
    second_moment: Tensor<R>,
}

impl<R> AdamW<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    /// Creates an optimizer for `parameters`.
    ///
    /// The device state is allocated lazily on the first [`Self::step`], so the
    /// parameters must already be registered in the graph by then.
    ///
    /// # Errors
    ///
    /// Returns an error if `parameters` is empty or contains duplicates.
    pub fn new(config: AdamWConfig, parameters: &[NodeId]) -> Result<Self> {
        if parameters.is_empty() {
            return invalid("parameters", "must not be empty");
        }

        let unique: HashSet<NodeId> = parameters.iter().copied().collect();
        if unique.len() != parameters.len() {
            return invalid("parameters", "must not contain duplicates");
        }

        Ok(Self {
            config,
            parameters: parameters.to_vec(),
            state: None,
            step: 0,
        })
    }

    /// Changes the learning rate used by subsequent [`Self::step`] calls.
    ///
    /// # Errors
    ///
    /// Returns an error if `learning_rate` is not finite or is negative.
    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<()> {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return invalid("learning_rate", "must be finite and non-negative");
        }
        self.config.learning_rate = learning_rate;
        Ok(())
    }

    /// Sets every parameter's accumulated gradient to zero.
    ///
    /// # Errors
    ///
    /// Returns an error if a zero-filled gradient tensor cannot be allocated.
    pub fn zero_grad(&self, graph: &mut Graph<R>) -> Result<()> {
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
    /// Returns an error if a parameter is not `f32`, if a stored buffer no
    /// longer matches the parameter layout, or if a tensor operation fails.
    pub fn step(&mut self, graph: &mut Graph<R>) -> Result<()> {
        self.initialize(graph)?;

        self.step += 1;
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.step as i32);
        let config = self.config;
        let parameters = &self.parameters;

        // Flatten the per-parameter gradients into one vector with a single
        // device-side concatenation.
        let gradients: Vec<Tensor<R>> = parameters
            .iter()
            .map(|&parameter| {
                let gradient = &graph.nodes[parameter].grad;
                gradient.contiguous()?.reshape(&[gradient.numel()])
            })
            .collect::<Result<_>>()?;
        let gradient_refs: Vec<&Tensor<R>> = gradients.iter().collect();
        let flat_gradient = Tensor::cat(&gradient_refs, 0)?;

        let state = self.state.as_mut().expect("optimizer state initialized");

        let first_moment = state
            .first_moment
            .mul_scalar(config.beta1 as f64)?
            .add(&flat_gradient.mul_scalar((1.0 - config.beta1) as f64)?)?;
        let second_moment = state.second_moment.mul_scalar(config.beta2 as f64)?.add(
            &flat_gradient
                .mul(&flat_gradient)?
                .mul_scalar((1.0 - config.beta2) as f64)?,
        )?;

        let denominator = second_moment
            .mul_scalar(1.0_f64 / bias_correction2 as f64)?
            .sqrt()?
            .add_scalar(config.eps as f64)?;
        let adaptive = first_moment
            .mul_scalar(1.0_f64 / bias_correction1 as f64)?
            .div(&denominator)?;

        let decay = config.learning_rate * config.weight_decay;
        let updated = state
            .parameters
            .mul_scalar((1.0 - decay) as f64)?
            .sub(&adaptive.mul_scalar(config.learning_rate as f64)?)?;

        // Give every graph parameter a contiguous view of its slice. The
        // `contiguous` call materializes the slice, so later operations do not
        // depend on the optimizer's flat buffer.
        let mut offset = 0;
        for (index, &parameter) in parameters.iter().enumerate() {
            let length = state.lengths[index];
            let data = updated
                .narrow(0, offset, length)?
                .reshape(&state.shapes[index])?
                .contiguous()?;
            graph.nodes[parameter].data = data;
            offset += length;
        }

        state.parameters = updated;
        state.first_moment = first_moment;
        state.second_moment = second_moment;

        Ok(())
    }

    fn initialize(&mut self, graph: &Graph<R>) -> Result<()> {
        if self.state.is_some() {
            return Ok(());
        }

        let mut shapes = Vec::with_capacity(self.parameters.len());
        let mut lengths = Vec::with_capacity(self.parameters.len());
        let mut flattened = Vec::with_capacity(self.parameters.len());

        for &parameter in &self.parameters {
            let data = &graph.nodes[parameter].data;
            if data.dtype() != DType::F32 {
                return invalid(
                    "parameters",
                    format!("parameter {parameter} is not f32 ({:?})", data.dtype()),
                );
            }
            shapes.push(data.shape().to_vec());
            lengths.push(data.numel());
            flattened.push(data.contiguous()?.reshape(&[data.numel()])?);
        }

        let references: Vec<&Tensor<R>> = flattened.iter().collect();
        let parameters = Tensor::cat(&references, 0)?;
        let total: usize = lengths.iter().sum();
        let device = parameters.device().clone();
        let first_moment = Tensor::zeros(&[total], DType::F32, &device)?;
        let second_moment = Tensor::zeros(&[total], DType::F32, &device)?;

        self.state = Some(AdamWState {
            shapes,
            lengths,
            parameters,
            first_moment,
            second_moment,
        });

        Ok(())
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
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_first_step_matches_adamw() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<CpuRuntime>::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;
        graph.nodes[parameter].grad = Tensor::from_slice(&[2.0_f32], &[1], &device)?;

        let mut optimizer = AdamW::<CpuRuntime>::new(
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
        let mut graph = Graph::<CpuRuntime>::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;

        let mut optimizer = AdamW::<CpuRuntime>::new(
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
        let mut graph = Graph::<CpuRuntime>::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;

        let mut optimizer = AdamW::<CpuRuntime>::new(
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
    fn test_updates_parameters_with_different_shapes() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<CpuRuntime>::new();
        let vector = graph.variable(Tensor::from_slice(&[1.0_f32, -1.0], &[2], &device)?)?;
        let matrix = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &device,
        )?)?;

        graph.nodes[vector].grad = Tensor::from_slice(&[1.0_f32, -1.0], &[2], &device)?;
        graph.nodes[matrix].grad = Tensor::from_slice(&[1.0_f32, 1.0, 1.0, 1.0], &[2, 2], &device)?;

        let mut optimizer = AdamW::<CpuRuntime>::new(
            AdamWConfig {
                learning_rate: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            &[vector, matrix],
        )?;
        optimizer.step(&mut graph)?;

        assert_eq!(graph.nodes[vector].data.shape(), &[2]);
        assert_eq!(graph.nodes[matrix].data.shape(), &[2, 2]);
        assert_close(graph.nodes[vector].data.to_vec::<f32>()[0], 0.9);
        assert_close(graph.nodes[vector].data.to_vec::<f32>()[1], -0.9);
        for (index, value) in graph.nodes[matrix].data.to_vec::<f32>().iter().enumerate() {
            assert_close(*value, [0.9, 1.9, 2.9, 3.9][index]);
        }

        Ok(())
    }

    #[test]
    fn test_zero_grad_clears_gradients() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<CpuRuntime>::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32, 2.0], &[1, 2], &device)?)?;
        graph.nodes[parameter].grad = Tensor::from_slice(&[3.0_f32, 4.0], &[1, 2], &device)?;

        let optimizer = AdamW::<CpuRuntime>::new(AdamWConfig::default(), &[parameter])?;
        optimizer.zero_grad(&mut graph)?;

        assert_eq!(graph.nodes[parameter].grad.to_vec::<f32>(), vec![0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_rejects_empty_and_duplicate_parameters() {
        let config = AdamWConfig::default();

        assert!(AdamW::<CpuRuntime>::new(config, &[]).is_err());
        assert!(AdamW::<CpuRuntime>::new(config, &[0, 0]).is_err());
    }

    #[test]
    fn test_schedule_warms_up_then_decays() {
        let schedule = LRSchedule {
            warmup_steps: 2,
            min_lr_ratio: 0.1,
        };

        assert_close(schedule.learning_rate(0.01, 1, 4), 0.005);
        assert_close(schedule.learning_rate(0.01, 2, 4), 0.01);
        assert!(schedule.learning_rate(0.01, 3, 4) < 0.01);
        assert_close(schedule.learning_rate(0.01, 4, 4), 0.001);
    }

    #[test]
    fn test_schedule_rejects_invalid_ratio() {
        let schedule = LRSchedule {
            warmup_steps: 1,
            min_lr_ratio: 1.5,
        };
        assert!(schedule.validate().is_err());
    }

    #[test]
    fn test_set_learning_rate_changes_update() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<CpuRuntime>::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;
        graph.nodes[parameter].grad = Tensor::from_slice(&[10.0_f32], &[1], &device)?;

        let mut optimizer = AdamW::<CpuRuntime>::new(
            AdamWConfig {
                learning_rate: 0.1,
                weight_decay: 0.0,
                ..AdamWConfig::default()
            },
            &[parameter],
        )?;
        optimizer.set_learning_rate(0.02)?;
        optimizer.step(&mut graph)?;

        assert_close(graph.nodes[parameter].data.to_vec::<f32>()[0], 0.98);
        assert!(optimizer.set_learning_rate(f32::NAN).is_err());
        Ok(())
    }
}
