//! Toy sequence-reversal task and training loop.
//!
//! The encoder receives a sequence of distinct digit tokens. The decoder must
//! emit the same digits in reverse order and then `<eos>`.
//!
//! Vocabulary layout:
//!
//! | ID | Token |
//! |----|-------|
//! | 0  | `<pad>` |
//! | 1  | `<sos>` |
//! | 2  | `<eos>` |
//! | 3..13 | digits `0..9` |
//!
//! Training uses teacher forcing: the decoder input is `<sos>` followed by the
//! reversed sequence, and the targets are the reversed sequence followed by
//! `<eos>`.

use super::loss::softmax_cross_entropy;
use crate::autograd::Graph;
use crate::optim::{AdamW, AdamWConfig};
use crate::transformer::{DecodingConfig, SamplingStrategy, Transformer};
use numr::error::Result;
use numr::runtime::cpu::CpuDevice;
use numr::tensor::Tensor;
use rand::RngExt;

/// Padding token. Unused by the reversal task, but reserved so the vocabulary
/// matches the usual seq2seq convention.
#[allow(dead_code)]
pub const PAD_TOKEN: usize = 0;
/// Start-of-sequence token used as the first decoder input.
pub const SOS_TOKEN: usize = 1;
/// End-of-sequence token that terminates generation.
pub const EOS_TOKEN: usize = 2;
/// First digit token ID.
pub const FIRST_DIGIT_TOKEN: usize = 3;
/// Number of digit tokens.
pub const NUM_DIGITS: usize = 10;
/// Total vocabulary size: pad, sos, eos, and ten digits.
pub const VOCAB_SIZE: usize = FIRST_DIGIT_TOKEN + NUM_DIGITS;

/// One reversal example, including teacher-forced decoder inputs and targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReversalExample {
    /// Encoder input digits.
    pub input: Vec<usize>,
    /// Decoder input: `<sos>` followed by the reversed digits.
    pub decoder_input: Vec<usize>,
    /// Targets: the reversed digits followed by `<eos>`.
    pub targets: Vec<usize>,
}

impl ReversalExample {
    /// Returns the expected generated sequence: the input in reverse order.
    pub fn target_sequence(&self) -> Vec<usize> {
        self.input.iter().rev().copied().collect()
    }
}

/// Samples a reversal example with distinct digits.
///
/// # Errors
///
/// Returns an error if `len` is zero or greater than [`NUM_DIGITS`].
pub fn sample_reversal(rng: &mut impl RngExt, len: usize) -> Result<ReversalExample> {
    // Partial Fisher-Yates shuffle: the first `len` entries are a uniform
    // sample without replacement from the digits.
    let mut digits: Vec<usize> = (FIRST_DIGIT_TOKEN..VOCAB_SIZE).collect();
    for index in 0..len {
        let swap_with: usize = index + rng.random_range(0..(digits.len() - index));
        digits.swap(index, swap_with);
    }
    digits.truncate(len);

    let mut decoder_input = Vec::with_capacity(len + 1);
    decoder_input.push(SOS_TOKEN);
    decoder_input.extend(digits.iter().rev().copied());

    let mut targets = Vec::with_capacity(len + 1);
    targets.extend(digits.iter().rev().copied());
    targets.push(EOS_TOKEN);

    Ok(ReversalExample {
        input: digits,
        decoder_input,
        targets,
    })
}

/// Training and evaluation settings for the reversal task.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Number of optimizer steps.
    pub steps: usize,
    /// Length of every encoder input sequence.
    pub seq_len: usize,
    /// Evaluate every this many steps. Step zero is always evaluated.
    pub eval_interval: usize,
    /// Number of held-out examples used for each evaluation.
    pub eval_examples: usize,
    /// AdamW hyperparameters.
    pub optimizer: AdamWConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 2_000,
            seq_len: 4,
            eval_interval: 100,
            eval_examples: 8,
            optimizer: AdamWConfig {
                learning_rate: 3e-3,
                weight_decay: 0.01,
                ..AdamWConfig::default()
            },
        }
    }
}

/// Evaluation result at a particular training step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluation {
    /// Optimizer step at which the evaluation ran; zero is before training.
    pub step: usize,
    /// Mean teacher-forced cross-entropy loss over the evaluation examples.
    pub loss: f32,
    /// Fraction of examples generated exactly correctly with greedy decoding.
    pub exact_match: f32,
}

/// Losses and periodic evaluations collected during training.
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Training loss for every step.
    pub losses: Vec<f32>,
    /// Periodic evaluation results, starting with step zero.
    pub evaluations: Vec<Evaluation>,
}

impl TrainingHistory {
    /// Returns the most recent training loss.
    pub fn final_loss(&self) -> Option<f32> {
        self.losses.last().copied()
    }

    /// Returns the most recent evaluation.
    pub fn final_evaluation(&self) -> Option<Evaluation> {
        self.evaluations.last().copied()
    }
}

/// Trains `model` to reverse sequences with AdamW.
///
/// The graph is assumed to contain the model parameters when training starts.
/// Nodes created during forward passes are removed after every step, which
/// keeps the append-only graph bounded. The model must use [`SOS_TOKEN`] and
/// [`EOS_TOKEN`] and have enough positional capacity for the decoder input.
///
/// # Errors
///
/// Returns an error if the configuration or model is incompatible with the
/// task, or if a forward, backward, optimizer, or evaluation operation fails.
pub fn train_reversal(
    graph: &mut Graph,
    device: &CpuDevice,
    model: &Transformer,
    config: &TrainConfig,
    rng: &mut impl RngExt,
) -> Result<TrainingHistory> {
    let parameters = model.parameters();
    let base_len = parameters
        .iter()
        .copied()
        .max()
        .map(|last| last + 1)
        .unwrap_or(0);
    graph.truncate(base_len)?;

    let mut optimizer = AdamW::new(config.optimizer, &parameters)?;
    let mut history = TrainingHistory::default();

    let eval_examples = (0..config.eval_examples)
        .map(|_| sample_reversal(rng, config.seq_len))
        .collect::<Result<Vec<_>>>()?;

    history.evaluations.push(evaluate(
        graph,
        device,
        model,
        &eval_examples,
        config,
        0,
        rng,
        base_len,
    )?);

    for step in 1..=config.steps {
        let example = sample_reversal(rng, config.seq_len)?;

        optimizer.zero_grad(graph)?;

        let encoded = model.encode(graph, device, &example.input)?;
        let logits = model.decode_logits(graph, device, &example.decoder_input, encoded)?;
        let logits_shape = graph.nodes[logits].data.shape().to_vec();
        let (loss, gradient) = softmax_cross_entropy(
            &graph.nodes[logits].data.to_vec::<f32>(),
            &example.targets,
            model.vocab_size,
        )?;
        let gradient = Tensor::from_slice(&gradient, &logits_shape, device)?;

        graph.backward_with_grad(logits, gradient)?;
        optimizer.step(graph)?;
        graph.truncate(base_len)?;
        history.losses.push(loss);

        if step.is_multiple_of(config.eval_interval) || step == config.steps {
            history.evaluations.push(evaluate(
                graph,
                device,
                model,
                &eval_examples,
                config,
                step,
                rng,
                base_len,
            )?);
        }
    }

    Ok(history)
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    graph: &mut Graph,
    device: &CpuDevice,
    model: &Transformer,
    examples: &[ReversalExample],
    config: &TrainConfig,
    step: usize,
    rng: &mut impl RngExt,
    base_len: usize,
) -> Result<Evaluation> {
    let mut total_loss = 0.0_f32;
    for example in examples {
        let encoded = model.encode(graph, device, &example.input)?;
        let logits = model.decode_logits(graph, device, &example.decoder_input, encoded)?;
        let (loss, _) = softmax_cross_entropy(
            &graph.nodes[logits].data.to_vec::<f32>(),
            &example.targets,
            model.vocab_size,
        )?;
        total_loss += loss;
    }

    let decoding = DecodingConfig {
        max_new_tokens: config.seq_len + 1,
        strategy: SamplingStrategy::Greedy,
    };
    let mut correct = 0;
    for example in examples {
        let generated = model.generate(graph, device, &example.input, &decoding, rng)?;
        if generated == example.target_sequence() {
            correct += 1;
        }
    }

    graph.truncate(base_len)?;

    Ok(Evaluation {
        step,
        loss: total_loss / examples.len() as f32,
        exact_match: correct as f32 / examples.len() as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerConfig;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn model_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: VOCAB_SIZE,
            d_model: 8,
            d_ff: 16,
            n_heads: 2,
            max_len: 8,
            sos_token: SOS_TOKEN,
            eos_token: EOS_TOKEN,
        }
    }

    #[test]
    fn test_sample_reversal_structure() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(7);
        let example = sample_reversal(&mut rng, 4)?;

        assert_eq!(example.input.len(), 4);
        assert!(
            example
                .input
                .iter()
                .all(|&token| (FIRST_DIGIT_TOKEN..VOCAB_SIZE).contains(&token))
        );

        let mut unique = example.input.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            4,
            "digits must be sampled without replacement"
        );

        let reversed = example.target_sequence();
        let mut expected_decoder_input = vec![SOS_TOKEN];
        expected_decoder_input.extend_from_slice(&reversed);
        assert_eq!(example.decoder_input, expected_decoder_input);

        let mut expected_targets = reversed;
        expected_targets.push(EOS_TOKEN);
        assert_eq!(example.targets, expected_targets);

        Ok(())
    }

    #[test]
    fn test_train_reversal_runs_and_updates_parameters() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::new();
        let model = Transformer::new(&mut graph, &device, model_config())?;
        let parameters = model.parameters();
        let base_len = parameters
            .iter()
            .copied()
            .max()
            .expect("model has parameters")
            + 1;
        let before: Vec<Vec<f32>> = parameters
            .iter()
            .map(|&parameter| graph.nodes[parameter].data.to_vec::<f32>())
            .collect();

        let config = TrainConfig {
            steps: 3,
            seq_len: 4,
            eval_interval: 2,
            eval_examples: 2,
            optimizer: AdamWConfig {
                learning_rate: 1e-2,
                ..AdamWConfig::default()
            },
        };
        let mut rng = StdRng::seed_from_u64(11);
        let history = train_reversal(&mut graph, &device, &model, &config, &mut rng)?;

        assert_eq!(history.losses.len(), 3);
        // Step zero, step two, and the final step.
        assert_eq!(history.evaluations.len(), 3);
        assert!(history.losses.iter().all(|loss| loss.is_finite()));
        assert!(history.evaluations.iter().all(|evaluation| {
            evaluation.loss.is_finite() && (0.0..=1.0).contains(&evaluation.exact_match)
        }));

        let after: Vec<Vec<f32>> = parameters
            .iter()
            .map(|&parameter| graph.nodes[parameter].data.to_vec::<f32>())
            .collect();
        assert!(
            before
                .iter()
                .flatten()
                .zip(after.iter().flatten())
                .any(|(old, new)| (old - new).abs() > 1e-9),
            "training did not change any parameter"
        );

        assert_eq!(
            graph.nodes.len(),
            base_len,
            "training should leave only the parameter nodes in the graph"
        );

        Ok(())
    }
}
