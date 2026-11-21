//! Complete SMS ham/spam example on top of the encoder-only BERT model.
//!
//! This file contains the labelled TSV loader, two-class `[CLS]` head, training
//! loop, evaluation, predictions, and the [`run_spam`] entry point used by the
//! binary. The head is a small linear layer registered immediately after the
//! BERT parameters, so one AdamW instance updates the whole model.
//!
//! As in the masked-language-model example, the loss and its logit gradient are
//! computed with device tensor operations. A training step reads back no tensor
//! values; only the final aggregate loss/accuracy and a few display predictions
//! cross the device boundary.

use super::{MAX_LEN, split_every_nth};
use crate::autograd::{Graph, NodeId};
use crate::bert::encoding::{BERT_CLS_TOKEN, BERT_SEP_TOKEN, BertVocab, tokenize};
use crate::bert::{Bert, BertConfig};
use crate::training::{AdamW, AdamWConfig};
use crate::transformer::{kaiming_init, linear};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::prelude::*;
use numr::runtime::Runtime;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env;
use std::error::Error as StdError;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Label for a legitimate message.
pub const HAM: usize = 0;
/// Label for a spam message.
pub const SPAM: usize = 1;
/// Number of output classes.
pub const SPAM_CLASSES: usize = 2;

/// One labelled SMS message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpamRecord {
    /// Message text.
    pub text: String,
    /// `HAM` or `SPAM`.
    pub label: usize,
}

impl SpamRecord {
    /// Creates a labelled message.
    pub fn new(text: impl Into<String>, label: usize) -> Self {
        Self {
            text: text.into(),
            label,
        }
    }
}

/// Loads a `label<TAB>text` file with `ham`/`spam` labels.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be read, a line is not a two-column
/// TSV record, a label is unknown, or the corpus is empty.
pub fn load_spam_records(path: impl AsRef<Path>) -> io::Result<Vec<SpamRecord>> {
    let contents = fs::read_to_string(path)?;
    parse_spam_records(&contents)
}

/// Parses the TSV format accepted by [`load_spam_records`].
///
/// # Errors
///
/// Returns an error if a non-empty line is malformed or the corpus is empty.
pub fn parse_spam_records(contents: &str) -> io::Result<Vec<SpamRecord>> {
    let mut records = Vec::new();

    for (index, line) in contents.lines().enumerate() {
        let line = line.trim_start_matches('\u{feff}').trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (label, text) = line.split_once('\t').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "line {}: expected `label<TAB>text`",
                    index.saturating_add(1)
                ),
            )
        })?;

        let label = match label.trim() {
            "ham" => HAM,
            "spam" => SPAM,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "line {}: unknown label `{other}`, expected `ham` or `spam`",
                        index.saturating_add(1)
                    ),
                ));
            }
        };

        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        records.push(SpamRecord::new(text, label));
    }

    if records.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the spam corpus is empty",
        ));
    }

    Ok(records)
}

/// Encodes a message as `[CLS] tokens [SEP]`, truncating to `max_len`.
fn encode_record(
    record: &SpamRecord,
    vocab: &BertVocab,
    max_len: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let max_content = max_len.saturating_sub(2);
    if max_content == 0 {
        return invalid("max_len", "must be at least two");
    }

    let mut token_ids = Vec::with_capacity(max_content + 2);
    let mut token_type_ids = Vec::with_capacity(max_content + 2);
    token_ids.push(BERT_CLS_TOKEN);
    token_type_ids.push(0);
    for token in tokenize(&record.text).into_iter().take(max_content) {
        token_ids.push(vocab.id(&token));
        token_type_ids.push(0);
    }
    token_ids.push(BERT_SEP_TOKEN);
    token_type_ids.push(0);

    Ok((token_ids, token_type_ids))
}

/// Two-class classification head over the `[CLS]` hidden state.
pub struct SpamHead {
    weight: NodeId,
    bias: NodeId,
}

impl SpamHead {
    /// Registers the head parameters directly after the BERT parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor allocation fails.
    pub fn new<R>(graph: &mut Graph<R>, device: &R::Device, d_model: usize) -> Result<Self>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
    {
        let weight = graph.variable(Tensor::from_slice(
            &kaiming_init(d_model * SPAM_CLASSES, d_model),
            &[d_model, SPAM_CLASSES],
            device,
        )?)?;
        let bias = graph.variable(Tensor::from_slice(
            &[0.0_f32; SPAM_CLASSES],
            &[1, SPAM_CLASSES],
            device,
        )?)?;

        Ok(Self { weight, bias })
    }

    /// Selects the `[CLS]` row, then applies the linear classifier.
    ///
    /// The selection matrix is a fresh leaf node for each forward pass, so the
    /// matmul remains differentiable with respect to the encoder hidden states
    /// while the selection itself is a constant.
    ///
    /// # Errors
    ///
    /// Returns an error if the hidden state is not two-dimensional or a tensor
    /// operation fails.
    pub fn forward<R>(
        &self,
        graph: &mut Graph<R>,
        device: &R::Device,
        hidden: NodeId,
    ) -> Result<NodeId>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
    {
        let shape = graph.nodes[hidden].data.shape();
        if shape.len() != 2 {
            return invalid("hidden", format!("expected a 2-D tensor, got {shape:?}"));
        }

        let mut selection = vec![0.0_f32; shape[0]];
        selection[0] = 1.0;
        let selection = graph.variable(Tensor::from_slice(&selection, &[1, shape[0]], device)?)?;
        let cls = graph.matmul(selection, hidden)?;
        linear(graph, cls, self.weight, self.bias)
    }

    /// Returns the head's trainable parameters.
    pub fn parameters(&self) -> Vec<NodeId> {
        vec![self.weight, self.bias]
    }
}

/// Training settings for the spam classifier.
#[derive(Debug, Clone, Copy)]
pub struct SpamTrainConfig {
    /// Number of optimizer steps.
    pub steps: usize,
    /// Print a progress line every this many steps. Zero disables progress.
    pub progress_interval: usize,
    /// AdamW hyperparameters.
    pub optimizer: AdamWConfig,
}

impl Default for SpamTrainConfig {
    fn default() -> Self {
        Self {
            steps: 250,
            progress_interval: 0,
            optimizer: AdamWConfig {
                learning_rate: 1e-3,
                weight_decay: 0.01,
                ..AdamWConfig::default()
            },
        }
    }
}

/// Final evaluation result for the spam classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpamEvaluation {
    /// Optimizer step at which evaluation ran.
    pub step: usize,
    /// Mean cross-entropy over the evaluation records.
    pub loss: f32,
    /// Fraction of records classified correctly.
    pub accuracy: f32,
    /// Number of evaluation records.
    pub examples: usize,
    /// Number of spam records in the evaluation set.
    pub spam_examples: usize,
}

/// Losses and the final evaluation collected during spam training.
#[derive(Debug, Clone, Default)]
pub struct SpamHistory {
    /// Training loss for every step.
    pub losses: Vec<f32>,
    /// Held-out evaluation after the final optimizer step.
    pub final_evaluation: Option<SpamEvaluation>,
}

impl SpamHistory {
    /// Returns the most recent training loss.
    pub fn final_loss(&self) -> Option<f32> {
        self.losses.last().copied()
    }
}

/// One spam prediction produced by [`predict_spam`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpamPrediction {
    /// Correct label (`HAM` or `SPAM`).
    pub target: usize,
    /// Predicted label (`HAM` or `SPAM`).
    pub predicted: usize,
    /// Model probability of the `SPAM` class.
    pub spam_probability: f32,
}

/// Corpus, vocabulary, and sequence length used for spam training.
pub struct SpamTrainingSet<'a> {
    /// Word-level vocabulary shared by training and evaluation.
    pub vocab: &'a BertVocab,
    /// Labelled messages used for training.
    pub train_records: &'a [SpamRecord],
    /// Held-out labelled messages used for final evaluation.
    pub eval_records: &'a [SpamRecord],
    /// Maximum sequence length, including `[CLS]` and `[SEP]`.
    pub max_len: usize,
}

/// Held-out records and their encoding settings, used for spam evaluation.
pub struct SpamEvaluationSet<'a> {
    /// Word-level vocabulary shared with training.
    pub vocab: &'a BertVocab,
    /// Labelled messages held out from training.
    pub records: &'a [SpamRecord],
    /// Maximum sequence length, including `[CLS]` and `[SEP]`.
    pub max_len: usize,
}

/// Trains the BERT encoder and spam head with two-class cross-entropy.
///
/// The graph is assumed to contain `model` and `head` parameters when training
/// starts. Per-step forward intermediates are removed with
/// [`Graph::truncate`](crate::autograd::Graph::truncate); only the final losses
/// are copied back to the host.
///
/// # Errors
///
/// Returns an error if the corpus or configuration is invalid, or if a tensor
/// operation fails.
pub fn train_spam<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    head: &SpamHead,
    data: SpamTrainingSet<'_>,
    config: &SpamTrainConfig,
    rng: &mut impl RngExt,
) -> Result<SpamHistory>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    if data.train_records.is_empty() {
        return invalid("train_records", "must not be empty");
    }
    if data.eval_records.is_empty() {
        return invalid("eval_records", "must not be empty");
    }
    if data.max_len < 2 {
        return invalid("max_len", "must be at least two");
    }

    let mut parameters = model.parameters();
    parameters.extend(head.parameters());
    let base_len = parameter_prefix_len(&parameters);
    graph.truncate(base_len)?;

    let mut optimizer = AdamW::<R>::new(config.optimizer, &parameters)?;
    let mut history = SpamHistory::default();
    let mut step_losses: Vec<Tensor<R>> = Vec::with_capacity(config.steps);
    let started = Instant::now();

    for step in 1..=config.steps {
        let record = &data.train_records[rng.random_range(0..data.train_records.len())];
        let (token_ids, token_type_ids) = encode_record(record, data.vocab, data.max_len)?;

        optimizer.zero_grad(graph)?;
        let hidden = model.encode(graph, device, &token_ids, &token_type_ids)?;
        let logits = head.forward(graph, device, hidden)?;
        let (loss, gradient) = spam_loss_and_gradient(graph, logits, record.label, device)?;
        graph.backward_with_grad(logits, gradient)?;
        optimizer.step(graph)?;
        graph.truncate(base_len)?;
        step_losses.push(loss);

        if config.progress_interval > 0
            && (step.is_multiple_of(config.progress_interval) || step == config.steps)
        {
            let width = config.steps.to_string().len();
            eprintln!(
                "  step {:>width$}/{} ({:.1}s)",
                step,
                config.steps,
                started.elapsed().as_secs_f64(),
                width = width
            );
        }
    }

    history.final_evaluation = Some(evaluate_spam(
        graph,
        device,
        model,
        head,
        SpamEvaluationSet {
            vocab: data.vocab,
            records: data.eval_records,
            max_len: data.max_len,
        },
        config.steps,
    )?);

    if !step_losses.is_empty() {
        let references: Vec<&Tensor<R>> = step_losses.iter().collect();
        history.losses = Tensor::cat(&references, 0)?.to_vec::<f32>();
    }

    Ok(history)
}

/// Evaluates spam loss and accuracy on held-out records.
///
/// # Errors
///
/// Returns an error if `records` is empty or a tensor operation fails.
pub fn evaluate_spam<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    head: &SpamHead,
    data: SpamEvaluationSet<'_>,
    step: usize,
) -> Result<SpamEvaluation>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    if data.records.is_empty() {
        return invalid("records", "must not be empty");
    }

    let mut parameters = model.parameters();
    parameters.extend(head.parameters());
    let base_len = parameter_prefix_len(&parameters);

    let mut total_loss: Option<Tensor<R>> = None;
    let mut total_correct: Option<Tensor<R>> = None;
    let mut spam_examples = 0;

    for record in data.records {
        let (token_ids, token_type_ids) = encode_record(record, data.vocab, data.max_len)?;
        let hidden = model.encode(graph, device, &token_ids, &token_type_ids)?;
        let logits = head.forward(graph, device, hidden)?;
        let (loss, _) = spam_loss_and_gradient(graph, logits, record.label, device)?;

        let targets = Tensor::from_slice(&[record.label as i32], &[1], device)?;
        let one_hot = targets.one_hot(SPAM_CLASSES)?;
        let predictions = graph.nodes[logits].data.argmax(1, false)?;
        let correct = predictions
            .one_hot(SPAM_CLASSES)?
            .mul(&one_hot)?
            .sum(&[1], false)?
            .sum(&[0], true)?;

        total_loss = Some(match total_loss {
            Some(total) => total.add(&loss)?,
            None => loss,
        });
        total_correct = Some(match total_correct {
            Some(total) => total.add(&correct)?,
            None => correct,
        });

        spam_examples += usize::from(record.label == SPAM);
        graph.truncate(base_len)?;
    }

    let total_loss = total_loss.expect("records are non-empty").to_vec::<f32>()[0];
    let total_correct = total_correct
        .expect("records are non-empty")
        .to_vec::<f32>()[0];

    Ok(SpamEvaluation {
        step,
        loss: total_loss / data.records.len() as f32,
        accuracy: total_correct / data.records.len() as f32,
        examples: data.records.len(),
        spam_examples,
    })
}

/// Predicts the class of one message, reading back only the two probabilities.
///
/// # Errors
///
/// Returns an error if a tensor operation fails.
pub fn predict_spam<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    head: &SpamHead,
    vocab: &BertVocab,
    record: &SpamRecord,
    max_len: usize,
) -> Result<SpamPrediction>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let mut parameters = model.parameters();
    parameters.extend(head.parameters());
    let base_len = parameter_prefix_len(&parameters);

    let (token_ids, token_type_ids) = encode_record(record, vocab, max_len)?;
    let hidden = model.encode(graph, device, &token_ids, &token_type_ids)?;
    let logits = head.forward(graph, device, hidden)?;
    let probabilities = graph.nodes[logits].data.softmax(-1)?.to_vec::<f32>();
    if probabilities.len() != SPAM_CLASSES {
        return invalid(
            "logits",
            format!(
                "expected {SPAM_CLASSES} probabilities, got {}",
                probabilities.len()
            ),
        );
    }

    let spam_probability = probabilities[SPAM];
    let predicted = if spam_probability > probabilities[HAM] {
        SPAM
    } else {
        HAM
    };

    graph.truncate(base_len)?;
    Ok(SpamPrediction {
        target: record.label,
        predicted,
        spam_probability,
    })
}

fn spam_loss_and_gradient<R>(
    graph: &Graph<R>,
    logits: NodeId,
    label: usize,
    device: &R::Device,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let logits_data = &graph.nodes[logits].data;
    if logits_data.shape() != [1, SPAM_CLASSES] {
        return invalid(
            "logits",
            format!(
                "expected shape [1, {SPAM_CLASSES}], got {:?}",
                logits_data.shape()
            ),
        );
    }
    if label >= SPAM_CLASSES {
        return invalid("label", format!("{label} is outside the class range"));
    }

    let targets = Tensor::from_slice(&[label as i32], &[1], device)?;
    let one_hot = targets.one_hot(SPAM_CLASSES)?;
    let log_probabilities = logits_data.log_softmax(-1)?;
    let loss = log_probabilities
        .mul(&one_hot)?
        .sum(&[1], false)?
        .sum(&[0], true)?
        .mul_scalar(-1.0)?;
    let gradient = log_probabilities.exp()?.sub(&one_hot)?;

    Ok((loss, gradient))
}

fn parameter_prefix_len(parameters: &[NodeId]) -> usize {
    parameters.iter().copied().max().map_or(0, |last| last + 1)
}

fn invalid<T>(argument: &'static str, reason: impl Into<String>) -> Result<T> {
    Err(Error::InvalidArgument {
        arg: argument,
        reason: reason.into(),
    })
}

/// Runs the spam-detection demo on the selected backend.
///
/// Positional command-line arguments are `[DATA_TSV] [STEPS]`; `--cpu` and
/// `--spam` are ignored here because the binary has already filtered them.
///
/// # Errors
///
/// Returns an error if the corpus or step count is invalid, or if a tensor
/// operation fails.
pub fn run_spam<R>(device: &R::Device) -> std::result::Result<(), Box<dyn StdError>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let mut arguments = env::args()
        .skip(1)
        .filter(|argument| argument != "--cpu" && argument != "--spam");
    let data_path = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/sms-spam-sample.tsv"));
    let steps = match arguments.next() {
        Some(value) => value.parse::<usize>()?,
        None => 250,
    };

    let records = load_spam_records(&data_path)?;
    let (training_records, evaluation_records) = split_every_nth(&records, 10);

    // Build the vocabulary from training messages only so held-out words cannot
    // leak into it. Unknown evaluation words map to `[UNK]`.
    let vocab = BertVocab::build_from_texts(
        training_records.iter().map(|record| record.text.as_str()),
        4_000,
        1,
    );
    let mut graph = Graph::<R>::new();
    let model = Bert::new(
        &mut graph,
        device,
        BertConfig {
            vocab_size: vocab.len(),
            d_model: 64,
            d_ff: 128,
            n_heads: 4,
            n_layers: 2,
            max_len: MAX_LEN,
            tie_word_embeddings: true,
        },
    )?;
    let head = SpamHead::new(&mut graph, device, model.config.d_model)?;

    let corpus_spam = records.iter().filter(|record| record.label == SPAM).count();

    println!("BERT SMS spam detection");
    println!("  backend:           {}", R::name());
    println!("  corpus:            {}", data_path.display());
    println!("  steps:             {}", steps);
    println!(
        "  messages:          {} ({} train / {} evaluation; {} spam / {} ham)",
        records.len(),
        training_records.len(),
        evaluation_records.len(),
        corpus_spam,
        records.len() - corpus_spam
    );
    println!("  vocabulary:        {} word pieces", vocab.len());
    println!(
        "  model:             {} layers, d_model={}, d_ff={}, {} heads, 2 classes",
        model.config.n_layers, model.config.d_model, model.config.d_ff, model.config.n_heads
    );
    println!();
    println!("Note: this fine-tunes the same encoder with a two-class [CLS] head;");
    println!("      labels are ham=0 and spam=1.");
    println!();

    let mut rng = StdRng::seed_from_u64(7);
    let history = train_spam(
        &mut graph,
        device,
        &model,
        &head,
        SpamTrainingSet {
            vocab: &vocab,
            train_records: &training_records,
            eval_records: &evaluation_records,
            max_len: MAX_LEN,
        },
        &SpamTrainConfig {
            steps,
            progress_interval: steps.div_ceil(20).max(1),
            optimizer: AdamWConfig {
                learning_rate: 1e-3,
                weight_decay: 0.01,
                ..Default::default()
            },
        },
        &mut rng,
    )?;

    if let Some(evaluation) = history.final_evaluation {
        let mean_training_loss = if history.losses.is_empty() {
            f32::NAN
        } else {
            history.losses.iter().sum::<f32>() / history.losses.len() as f32
        };

        println!();
        println!("Final result (step {}):", evaluation.step);
        println!(
            "  training loss:   {:.4} (mean), {:.4} (last step)",
            mean_training_loss,
            history.final_loss().unwrap_or(f32::NAN)
        );
        println!("  evaluation loss: {:.4}", evaluation.loss);
        println!(
            "  accuracy:        {:.1}% ({} messages, {} spam)",
            evaluation.accuracy * 100.0,
            evaluation.examples,
            evaluation.spam_examples
        );
    }

    println!();
    println!("Sample spam predictions:");
    for record in evaluation_records.iter().take(8) {
        let prediction = predict_spam(&mut graph, device, &model, &head, &vocab, record, MAX_LEN)?;
        let label_name = if record.label == SPAM { "spam" } else { "ham" };
        let predicted_name = if prediction.predicted == SPAM {
            "spam"
        } else {
            "ham"
        };
        println!(
            "  target {:<4} predicted {:<4} p(spam)={:>5.1}%  {}",
            label_name,
            predicted_name,
            prediction.spam_probability * 100.0,
            short_text(&record.text, 72)
        );
    }

    Ok(())
}

fn short_text(text: &str, max_chars: usize) -> String {
    let short: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        format!("{short}...")
    } else {
        short
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bert::BertConfig;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn records() -> Vec<SpamRecord> {
        vec![
            SpamRecord::new("Free money now, call 0900", SPAM),
            SpamRecord::new("Hey, are we still on for lunch?", HAM),
            SpamRecord::new("Win a prize, text YES to 8008", SPAM),
            SpamRecord::new("I will be home late tonight", HAM),
        ]
    }

    #[test]
    fn test_parse_spam_records() -> io::Result<()> {
        let parsed = parse_spam_records("ham\tHello\nspam\tFree cash\n")?;
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], SpamRecord::new("Hello", HAM));
        assert_eq!(parsed[1], SpamRecord::new("Free cash", SPAM));
        Ok(())
    }

    #[test]
    fn test_bundled_spam_corpus_loads() -> io::Result<()> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sms-spam-sample.tsv");
        let records = load_spam_records(path)?;
        assert_eq!(records.len(), 1_200);
        assert_eq!(
            records.iter().filter(|record| record.label == HAM).count(),
            600
        );
        assert_eq!(
            records.iter().filter(|record| record.label == SPAM).count(),
            600
        );
        assert!(records.iter().all(|record| !record.text.is_empty()));
        Ok(())
    }

    #[test]
    fn test_train_spam_reduces_loss() -> Result<()> {
        let device = CpuDevice::new();
        let corpus = records();
        let vocab =
            BertVocab::build_from_texts(corpus.iter().map(|record| record.text.as_str()), 64, 1);
        let mut graph = Graph::<CpuRuntime>::new();
        let model = Bert::new(
            &mut graph,
            &device,
            BertConfig {
                vocab_size: vocab.len(),
                d_model: 16,
                d_ff: 32,
                n_heads: 2,
                n_layers: 1,
                max_len: 12,
                tie_word_embeddings: true,
            },
        )?;
        let head = SpamHead::new(&mut graph, &device, model.config.d_model)?;

        let mut rng = StdRng::seed_from_u64(7);
        let history = train_spam(
            &mut graph,
            &device,
            &model,
            &head,
            SpamTrainingSet {
                vocab: &vocab,
                train_records: &corpus[..2],
                eval_records: &corpus[2..],
                max_len: 12,
            },
            &SpamTrainConfig {
                steps: 4,
                ..SpamTrainConfig::default()
            },
            &mut rng,
        )?;

        assert_eq!(history.losses.len(), 4);
        assert!(history.losses.iter().all(|loss| loss.is_finite()));
        let evaluation = history.final_evaluation.expect("final evaluation");
        assert_eq!(evaluation.examples, 2);
        assert!(evaluation.loss.is_finite());

        let prediction = predict_spam(&mut graph, &device, &model, &head, &vocab, &corpus[0], 12)?;
        assert!(prediction.spam_probability.is_finite());
        assert!(prediction.predicted < SPAM_CLASSES);
        Ok(())
    }
}
