//! Runnable BERT demos.
//!
//! The command-line binary in `src/main.rs` only selects a backend and calls
//! one of these examples. `mlm.rs` contains the whole English–Italian masked
//! language-model example (data loading, masking, training, evaluation, and
//! prediction); `spam.rs` contains the whole labelled SMS classification
//! example. The generic BERT model lives in the parent `bert` module.

mod mlm;
mod spam;

pub use mlm::run_mlm;
pub use spam::run_spam;

use crate::autograd::{Graph, NodeId};
use crate::bert::BertConfig;
use numr::error::Result;
use numr::prelude::*;
use numr::runtime::Runtime;
use rand::RngExt;
use std::env;
use std::error::Error as StdError;
use std::path::PathBuf;

/// Probability used by the masked-language-model example.
pub(crate) const MASK_PROBABILITY: f32 = 0.15;
/// Maximum sequence length used by both examples by default.
pub(crate) const MAX_LEN: usize = 64;
/// Default corpus for the masked-language-model example.
pub(crate) const DEFAULT_MLM_DATA: &str = "data/eng-it-full.tsv";
/// Default corpus for the spam example.
pub(crate) const DEFAULT_SPAM_DATA: &str = "data/sms-spam-full.tsv";
/// Default optimizer steps for the masked-language-model example.
pub(crate) const DEFAULT_MLM_STEPS: usize = 20_000;
/// Default BPE vocabulary size for the masked-language-model example.
pub(crate) const MLM_VOCAB_SIZE: usize = 8_000;
/// Number of training pairs used to learn BPE merges on large corpora.
pub(crate) const MLM_VOCAB_CORPUS_PAIRS: usize = 20_000;
/// Maximum number of fixed held-out examples used for validation.
pub(crate) const MLM_MAX_EVAL_EXAMPLES: usize = 512;
/// Default linear warmup length for the MLM learning-rate schedule.
pub(crate) const MLM_WARMUP_STEPS: usize = 500;
/// Default optimizer steps for the spam example.
pub(crate) const DEFAULT_SPAM_STEPS: usize = 1_000;

/// Model size settings shared by both demos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ModelCapacity {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub max_len: usize,
    pub dropout: f32,
}

impl Default for ModelCapacity {
    fn default() -> Self {
        Self {
            d_model: 128,
            d_ff: 256,
            n_heads: 4,
            n_layers: 2,
            max_len: MAX_LEN,
            dropout: 0.1,
        }
    }
}

impl ModelCapacity {
    pub(crate) fn validate(&self) -> std::result::Result<(), String> {
        if self.d_model == 0 || self.d_ff == 0 || self.n_heads == 0 || self.n_layers == 0 {
            return Err("model dimensions must be greater than zero".to_string());
        }
        if self.max_len < 4 {
            return Err("--max-len must be at least four".to_string());
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(format!(
                "--n-heads ({}) must divide --d-model ({})",
                self.n_heads, self.d_model
            ));
        }
        if !(0.0..1.0).contains(&self.dropout) || !self.dropout.is_finite() {
            return Err("--dropout must be in [0, 1)".to_string());
        }
        Ok(())
    }

    pub(crate) fn bert_config(&self, vocab_size: usize) -> BertConfig {
        BertConfig {
            vocab_size,
            d_model: self.d_model,
            d_ff: self.d_ff,
            n_heads: self.n_heads,
            n_layers: self.n_layers,
            max_len: self.max_len,
            dropout: self.dropout,
            tie_word_embeddings: true,
        }
    }
}

/// Deep snapshot of the parameters with the best validation score so far.
pub(crate) struct Checkpoint<R: Runtime> {
    step: usize,
    parameters: Vec<Tensor<R>>,
}

impl<R: Runtime> Checkpoint<R> {
    pub(crate) fn capture(graph: &Graph<R>, parameters: &[NodeId], step: usize) -> Result<Self> {
        let snapshot = parameters
            .iter()
            .map(|&id| graph.nodes[id].data.clone_deep())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            step,
            parameters: snapshot,
        })
    }

    pub(crate) fn step(&self) -> usize {
        self.step
    }

    pub(crate) fn restore(&self, graph: &mut Graph<R>, parameters: &[NodeId]) -> Result<()> {
        for (&id, snapshot) in parameters.iter().zip(&self.parameters) {
            graph.nodes[id].data = snapshot.clone_deep()?;
        }
        Ok(())
    }
}

/// Tracks validation improvements and decides when to stop early.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EarlyStopping {
    best_loss: f32,
    best_step: usize,
    stale_validations: usize,
}

impl EarlyStopping {
    pub(crate) fn new() -> Self {
        Self {
            best_loss: f32::INFINITY,
            best_step: 0,
            stale_validations: 0,
        }
    }

    /// Records one validation loss.
    ///
    /// Returns whether this is a new best and whether training should stop.
    pub(crate) fn observe(
        &mut self,
        step: usize,
        loss: f32,
        min_delta: f32,
        patience: usize,
    ) -> (bool, bool) {
        let improved = loss + min_delta < self.best_loss;
        if improved {
            self.best_loss = loss;
            self.best_step = step;
            self.stale_validations = 0;
        } else {
            self.stale_validations += 1;
        }
        let should_stop = patience > 0 && self.stale_validations >= patience;
        (improved, should_stop)
    }
}

/// Parsed command-line arguments used by both demos.
#[derive(Debug, Clone)]
pub(crate) struct ExampleArgs {
    pub data_path: Option<PathBuf>,
    pub steps: Option<usize>,
    pub capacity: ModelCapacity,
}

impl ExampleArgs {
    /// Parses arguments after removing the backend flags handled by `main`.
    pub(crate) fn parse() -> std::result::Result<Self, Box<dyn StdError>> {
        let raw: Vec<String> = env::args()
            .skip(1)
            .filter(|argument| argument != "--cpu" && argument != "--spam")
            .collect();
        let mut capacity = ModelCapacity::default();
        let mut positional: Vec<String> = Vec::new();
        let mut index = 0;

        while index < raw.len() {
            let argument = &raw[index];
            let (name, inline_value) = match argument.split_once('=') {
                Some((name, value)) => (name, Some(value.to_string())),
                None => (argument.as_str(), None),
            };

            match name {
                "--d-model" | "--d-ff" | "--n-heads" | "--layers" | "--max-len" | "--dropout" => {
                    let value = match inline_value {
                        Some(value) => value,
                        None => {
                            index += 1;
                            raw.get(index)
                                .cloned()
                                .ok_or_else(|| format!("{name} requires a value"))?
                        }
                    };
                    match name {
                        "--d-model" => capacity.d_model = parse_value(name, &value)?,
                        "--d-ff" => capacity.d_ff = parse_value(name, &value)?,
                        "--n-heads" => capacity.n_heads = parse_value(name, &value)?,
                        "--layers" => capacity.n_layers = parse_value(name, &value)?,
                        "--max-len" => capacity.max_len = parse_value(name, &value)?,
                        "--dropout" => capacity.dropout = parse_value(name, &value)?,
                        _ => unreachable!(),
                    }
                }
                _ if argument.starts_with('-') => {
                    return Err(format!("unknown option `{argument}`").into());
                }
                _ => positional.push(argument.clone()),
            }
            index += 1;
        }

        capacity
            .validate()
            .map_err(Into::<Box<dyn StdError>>::into)?;
        if positional.len() > 2 {
            return Err("expected at most two positional arguments: [DATA_TSV] [STEPS]".into());
        }

        let data_path = positional.first().cloned().map(PathBuf::from);
        let steps = match positional.get(1) {
            Some(value) => Some(parse_value("STEPS", value)?),
            None => None,
        };

        Ok(Self {
            data_path,
            steps,
            capacity,
        })
    }
}

fn parse_value<T>(name: &str, value: &str) -> std::result::Result<T, Box<dyn StdError>>
where
    T: std::str::FromStr,
    T::Err: StdError + Send + Sync + 'static,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid value for {name}: {error}").into())
}

/// Holds out every `nth` item for evaluation and returns `(train, evaluation)`.
///
/// Both examples use this helper with `nth = 10`, so their splits are
/// deterministic without needing to shuffle the generated corpora.
pub(crate) fn split_every_nth<T: Clone>(items: &[T], nth: usize) -> (Vec<T>, Vec<T>) {
    let mut training = Vec::new();
    let mut evaluation = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if nth > 0 && index % nth == 0 {
            evaluation.push(item.clone());
        } else {
            training.push(item.clone());
        }
    }
    (training, evaluation)
}

/// Selects up to `limit` items uniformly at random without replacement.
///
/// Uses reservoir sampling so the input does not need to be shuffled or copied
/// in full.
pub(crate) fn reservoir_sample<T: Clone>(
    items: &[T],
    limit: usize,
    rng: &mut impl RngExt,
) -> Vec<T> {
    if items.len() <= limit {
        return items.to_vec();
    }

    let mut selected = items[..limit].to_vec();
    for (index, item) in items.iter().enumerate().skip(limit) {
        let slot = rng.random_range(0..=index);
        if slot < limit {
            selected[slot] = item.clone();
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_model_capacity_validation() {
        assert!(ModelCapacity::default().validate().is_ok());
        let invalid = ModelCapacity {
            n_heads: 3,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
        let invalid = ModelCapacity {
            n_heads: 4,
            dropout: 1.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parse_value_reports_context() {
        let error = parse_value::<usize>("--d-model", "abc").unwrap_err();
        assert!(error.to_string().contains("--d-model"));
    }

    #[test]
    fn test_early_stopping_tracks_best_and_patience() {
        let mut stopping = EarlyStopping::new();
        assert_eq!(stopping.observe(0, 2.0, 0.0, 2), (true, false));
        assert_eq!(stopping.observe(10, 1.0, 0.0, 2), (true, false));
        assert_eq!(stopping.observe(20, 1.5, 0.0, 2), (false, false));
        assert_eq!(stopping.observe(30, 1.4, 0.0, 2), (false, true));
        assert_eq!(stopping.observe(40, 0.5, 0.0, 0), (true, false));
    }

    #[test]
    fn test_checkpoint_restores_parameter_data() -> Result<()> {
        let device = CpuDevice::new();
        let mut graph = Graph::<CpuRuntime>::new();
        let parameter = graph.variable(Tensor::from_slice(&[1.0_f32], &[1], &device)?)?;
        let checkpoint = Checkpoint::capture(&graph, &[parameter], 7)?;
        graph.nodes[parameter].data = Tensor::from_slice(&[9.0_f32], &[1], &device)?;

        checkpoint.restore(&mut graph, &[parameter])?;

        assert_eq!(checkpoint.step(), 7);
        assert_eq!(graph.nodes[parameter].data.to_vec::<f32>(), vec![1.0]);
        Ok(())
    }
}
