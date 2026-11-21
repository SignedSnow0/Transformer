//! Complete English–Italian masked-language-model example.
//!
//! This file contains the data loader, masking, training loop, evaluation,
//! predictions, and the [`run_mlm`] entry point used by the binary. The training
//! objective is BERT's masked language model: a random subset of content tokens
//! is replaced by `[MASK]`, a random token, or left unchanged, and the model is
//! trained to recover the original tokens from bidirectional context. The demo
//! uses the learned BPE [`EncodingStrategy`], so
//! rare words are encoded as subword pieces rather than `[UNK]`.
//! Tokenization and vocabulary lookup live in [`crate::bert::encoding`].
//!
//! The bundled `data/eng-it-sample.tsv` file is a deterministic sample from
//! OPUS Tatoeba (CC BY 2.0 FR). See `data/README.md` for provenance.

use super::{MASK_PROBABILITY, MAX_LEN, split_every_nth};
use crate::autograd::{Graph, NodeId};
use crate::bert::encoding::{
    BERT_CLS_TOKEN, BERT_MASK_TOKEN, BERT_NUM_SPECIAL_TOKENS, BERT_SEP_TOKEN, BERT_UNK_TOKEN,
    BertVocab, EncodingStrategy, ParallelPair, tokenize,
};
use crate::bert::{Bert, BertConfig};
use crate::training::{AdamW, AdamWConfig};
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

/// Loads a two-column TSV file with English and Italian sentences.
///
/// The expected format is `english<TAB>italian`, one pair per line. Blank
/// lines and lines beginning with `#` are ignored.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be read, if a line is not a
/// two-column TSV record, or if no pairs are found.
pub fn load_parallel_pairs(path: impl AsRef<Path>) -> io::Result<Vec<ParallelPair>> {
    let contents = fs::read_to_string(path)?;
    parse_parallel_pairs(&contents)
}

/// Parses the TSV format accepted by [`load_parallel_pairs`].
///
/// # Errors
///
/// Returns an error if a non-empty line does not contain a tab separator or if
/// the resulting corpus is empty.
pub fn parse_parallel_pairs(contents: &str) -> io::Result<Vec<ParallelPair>> {
    let mut pairs = Vec::new();

    for (index, line) in contents.lines().enumerate() {
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let (english, italian) = line.split_once('\t').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "line {}: expected `english<TAB>italian`",
                    index.saturating_add(1)
                ),
            )
        })?;
        let english = english.trim();
        let italian = italian.trim();
        if english.is_empty() || italian.is_empty() {
            continue;
        }

        pairs.push(ParallelPair::new(english, italian));
    }

    if pairs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the parallel corpus is empty",
        ));
    }

    Ok(pairs)
}

/// A masked-language-model training example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BertExample {
    /// Input token IDs after masking.
    pub token_ids: Vec<usize>,
    /// Segment IDs: 0 for `[CLS]`/English/`[SEP]`, 1 for Italian/`[SEP]`.
    pub token_type_ids: Vec<usize>,
    /// Masked positions and their original target token IDs.
    pub masked: Vec<(usize, usize)>,
}

/// Builds a BERT sentence-pair example and applies MLM masking.
///
/// The sequence layout is:
///
/// ```text
/// [CLS] English tokens [SEP] Italian tokens [SEP]
/// ```
///
/// Text is encoded with the vocabulary's [`EncodingStrategy`], so the same code
/// works with whole words or BPE pieces. At least one maskable content token is
/// always selected. Returns `None` if the pair contains none after truncation.
pub fn build_mlm_example(
    pair: &ParallelPair,
    vocab: &BertVocab,
    max_len: usize,
    mask_probability: f32,
    rng: &mut impl RngExt,
) -> Option<BertExample> {
    let max_content = max_len.saturating_sub(3);
    if max_content == 0 {
        return None;
    }

    let pre_tokens = tokenize(&pair.english);
    let english_tokens = vocab.encode_tokens(&pre_tokens);
    let pre_tokens = tokenize(&pair.italian);
    let italian_tokens = vocab.encode_tokens(&pre_tokens);

    let mut english_len = english_tokens.len().min(max_content / 2);
    let mut italian_len = italian_tokens.len().min(max_content - english_len);

    let remaining = max_content - english_len - italian_len;
    let extra_english = remaining.min(english_tokens.len() - english_len);
    english_len += extra_english;
    let extra_italian = (remaining - extra_english).min(italian_tokens.len() - italian_len);
    italian_len += extra_italian;

    if english_len == 0 && italian_len == 0 {
        return None;
    }

    let mut token_ids = Vec::with_capacity(english_len + italian_len + 3);
    let mut token_type_ids = Vec::with_capacity(english_len + italian_len + 3);

    token_ids.push(BERT_CLS_TOKEN);
    token_type_ids.push(0);
    for &token in &english_tokens[..english_len] {
        token_ids.push(token);
        token_type_ids.push(0);
    }
    token_ids.push(BERT_SEP_TOKEN);
    token_type_ids.push(0);
    for &token in &italian_tokens[..italian_len] {
        token_ids.push(token);
        token_type_ids.push(1);
    }
    token_ids.push(BERT_SEP_TOKEN);
    token_type_ids.push(1);

    let candidates: Vec<usize> = token_ids
        .iter()
        .enumerate()
        .filter_map(|(position, &id)| vocab.is_maskable(id).then_some(position))
        .collect();
    if candidates.is_empty() {
        return None;
    }

    let probability = if mask_probability.is_finite() {
        mask_probability.clamp(0.0, 1.0)
    } else {
        0.15
    };
    let mut masked = Vec::new();
    for &position in &candidates {
        if rng.random::<f32>() < probability {
            masked.push(apply_mask(&mut token_ids, position, vocab.len(), rng));
        }
    }

    if masked.is_empty() {
        let position = candidates[rng.random_range(0..candidates.len())];
        masked.push(apply_mask(&mut token_ids, position, vocab.len(), rng));
    }

    Some(BertExample {
        token_ids,
        token_type_ids,
        masked,
    })
}

fn apply_mask(
    token_ids: &mut [usize],
    position: usize,
    vocab_size: usize,
    rng: &mut impl RngExt,
) -> (usize, usize) {
    let target = token_ids[position];
    let roll = rng.random::<f32>();

    token_ids[position] = if roll < 0.8 {
        BERT_MASK_TOKEN
    } else if roll < 0.9 {
        let content_size = vocab_size.saturating_sub(BERT_NUM_SPECIAL_TOKENS);
        if content_size == 0 {
            BERT_UNK_TOKEN
        } else {
            BERT_NUM_SPECIAL_TOKENS + rng.random_range(0..content_size)
        }
    } else {
        target
    };

    (position, target)
}

/// Training settings for BERT MLM.
#[derive(Debug, Clone, Copy)]
pub struct BertTrainConfig {
    /// Number of optimizer steps.
    pub steps: usize,
    /// Print a progress line every this many steps. Zero disables progress.
    pub progress_interval: usize,
    /// Probability of selecting a content token for masking.
    pub mask_probability: f32,
    /// AdamW hyperparameters.
    pub optimizer: AdamWConfig,
}

impl Default for BertTrainConfig {
    fn default() -> Self {
        Self {
            steps: 500,
            progress_interval: 0,
            mask_probability: 0.15,
            optimizer: AdamWConfig {
                learning_rate: 1e-3,
                weight_decay: 0.01,
                ..AdamWConfig::default()
            },
        }
    }
}

/// Final evaluation result for a fixed set of masked examples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertEvaluation {
    /// Optimizer step at which final evaluation ran.
    pub step: usize,
    /// Mean masked-token cross-entropy loss.
    pub loss: f32,
    /// Fraction of masked tokens predicted correctly.
    pub masked_accuracy: f32,
    /// Number of masked tokens evaluated.
    pub masked_tokens: usize,
}

/// Losses and the final evaluation collected during BERT training.
#[derive(Debug, Clone, Default)]
pub struct BertTrainingHistory {
    /// Training loss for every step.
    pub losses: Vec<f32>,
    /// Held-out evaluation after the final optimizer step.
    pub final_evaluation: Option<BertEvaluation>,
}

impl BertTrainingHistory {
    /// Returns the most recent training loss.
    pub fn final_loss(&self) -> Option<f32> {
        self.losses.last().copied()
    }
}

/// One masked-token prediction produced by [`predict_masked_tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct BertPrediction {
    /// Position in the input sequence.
    pub position: usize,
    /// Original token ID that was masked.
    pub target: usize,
    /// Token ID with the largest logit.
    pub predicted: usize,
    /// Softmax probability assigned to the predicted token.
    pub confidence: f32,
    /// Highest-probability candidates, best first.
    pub alternatives: Vec<(usize, f32)>,
}

/// Corpus, vocabulary, and fixed evaluation masks used for BERT training.
pub struct BertTrainingSet<'a> {
    /// Vocabulary and encoding strategy shared by training and evaluation.
    pub vocab: &'a BertVocab,
    /// Sentence pairs sampled for training.
    pub train_pairs: &'a [ParallelPair],
    /// Fixed masked examples used for evaluation.
    pub eval_examples: &'a [BertExample],
}

/// Trains `model` with masked-language modeling on English–Italian pairs.
///
/// The graph is assumed to contain the model parameters when training starts.
/// Per-step forward intermediates are removed with
/// [`Graph::truncate`](crate::autograd::Graph::truncate).
///
/// Held-out evaluation runs once, after the final optimizer step. A caller that
/// wants a baseline should call [`evaluate_bert`] explicitly before training.
///
/// # Errors
///
/// Returns an error if the corpus or evaluation set is empty, if the
/// configuration is invalid, or if a forward, backward, or optimizer operation
/// fails.
pub fn train_bert<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    data: BertTrainingSet<'_>,
    config: &BertTrainConfig,
    rng: &mut impl RngExt,
) -> Result<BertTrainingHistory>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    if data.train_pairs.is_empty() {
        return invalid("train_pairs", "must not be empty");
    }
    if data.eval_examples.is_empty() {
        return invalid("eval_examples", "must not be empty");
    }
    if !(0.0..=1.0).contains(&config.mask_probability) {
        return invalid("mask_probability", "must be in [0, 1]");
    }

    let parameters = model.parameters();
    let base_len = parameter_prefix_len(&parameters);
    graph.truncate(base_len)?;

    let mut optimizer = AdamW::<R>::new(config.optimizer, &parameters)?;
    let mut history = BertTrainingHistory::default();

    // Keep each step's scalar loss on the device and read the whole vector
    // once after training. This keeps the optimization loop free of
    // device-to-host copies.
    let mut step_losses: Vec<Tensor<R>> = Vec::with_capacity(config.steps);
    let started = Instant::now();

    for step in 1..=config.steps {
        let example = sample_training_example(
            data.train_pairs,
            data.vocab,
            model.config.max_len,
            config.mask_probability,
            rng,
        )
        .ok_or_else(|| Error::InvalidArgument {
            arg: "train_pairs",
            reason: "could not build an example with a maskable content token".to_string(),
        })?;

        optimizer.zero_grad(graph)?;

        let output = model.forward(graph, device, &example.token_ids, &example.token_type_ids)?;
        let (loss, gradient) =
            masked_loss_and_gradient(graph, output.logits, &example, model.vocab_size(), device)?;
        graph.backward_with_grad(output.logits, gradient)?;
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

    history.final_evaluation = Some(evaluate_bert(
        graph,
        device,
        model,
        data.eval_examples,
        config.steps,
    )?);

    if !step_losses.is_empty() {
        let references: Vec<&Tensor<R>> = step_losses.iter().collect();
        history.losses = Tensor::cat(&references, 0)?.to_vec::<f32>();
    }

    Ok(history)
}

fn sample_training_example(
    train_pairs: &[ParallelPair],
    vocab: &BertVocab,
    max_len: usize,
    mask_probability: f32,
    rng: &mut impl RngExt,
) -> Option<BertExample> {
    // A very short pair can still contain no maskable pieces after truncation.
    // Try a few other pairs before giving up so one unlucky draw does not abort
    // a long run.
    for _ in 0..8 {
        let pair = &train_pairs[rng.random_range(0..train_pairs.len())];
        if let Some(example) = build_mlm_example(pair, vocab, max_len, mask_probability, rng) {
            return Some(example);
        }
    }
    None
}

/// Evaluates masked-token loss and accuracy on fixed examples.
///
/// The forward intermediates created during evaluation are removed before the
/// function returns, leaving the model parameters and their gradients intact.
/// Losses and correct-token counts are accumulated on the device; only the two
/// final scalars are copied back to the host.
///
/// # Errors
///
/// Returns an error if `examples` is empty or a model operation fails.
pub fn evaluate_bert<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    examples: &[BertExample],
    step: usize,
) -> Result<BertEvaluation>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    if examples.is_empty() {
        return invalid("examples", "must not be empty");
    }

    let base_len = parameter_prefix_len(&model.parameters());
    let vocab_size = model.vocab_size();
    let mut total_loss: Option<Tensor<R>> = None;
    let mut total_correct: Option<Tensor<R>> = None;
    let mut masked_tokens = 0_usize;

    for example in examples {
        let output = model.forward(graph, device, &example.token_ids, &example.token_type_ids)?;
        let batch = masked_batch(graph, output.logits, example, vocab_size, device)?;
        let (loss, _) = masked_cross_entropy(&batch.selected, &batch.one_hot, batch.count)?;

        // Each example's loss is a mean over its own masked tokens, so weight
        // it by the count before accumulating.
        let weighted_loss = loss.mul_scalar(batch.count as f64)?;
        total_loss = Some(match total_loss {
            Some(total) => total.add(&weighted_loss)?,
            None => weighted_loss,
        });

        let predictions = batch.selected.argmax(1, false)?;
        let matches = predictions
            .one_hot(vocab_size)?
            .mul(&batch.one_hot)?
            .sum(&[1], false)?;
        let correct = matches.sum(&[0], true)?;
        total_correct = Some(match total_correct {
            Some(total) => total.add(&correct)?,
            None => correct,
        });

        masked_tokens += batch.count;
        graph.truncate(base_len)?;
    }

    let total_loss = total_loss.expect("examples are non-empty").to_vec::<f32>()[0];
    let total_correct = total_correct
        .expect("examples are non-empty")
        .to_vec::<f32>()[0];

    Ok(BertEvaluation {
        step,
        loss: total_loss / masked_tokens as f32,
        masked_accuracy: total_correct / masked_tokens as f32,
        masked_tokens,
    })
}

/// Predicts every masked token in `example`.
///
/// This is intended for displaying model output after training. It reads back
/// only the small probability rows needed for the display; the full logits stay
/// on the device. The forward intermediates are removed before it returns.
///
/// # Errors
///
/// Returns an error if the example has no masked positions or a model operation
/// fails.
pub fn predict_masked_tokens<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    example: &BertExample,
) -> Result<Vec<BertPrediction>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let base_len = parameter_prefix_len(&model.parameters());
    let vocab_size = model.vocab_size();

    let output = model.forward(graph, device, &example.token_ids, &example.token_type_ids)?;
    let batch = masked_batch(graph, output.logits, example, vocab_size, device)?;

    const TOP_K: usize = 5;

    let probability_rows = batch.selected.softmax(-1)?.to_vec::<f32>();
    let predictions = example
        .masked
        .iter()
        .enumerate()
        .map(|(row, &(position, target))| {
            let start = row * vocab_size;
            let mut alternatives: Vec<(usize, f32)> = probability_rows[start..start + vocab_size]
                .iter()
                .copied()
                .enumerate()
                .collect();
            alternatives.sort_by(|left, right| right.1.total_cmp(&left.1));
            alternatives.truncate(TOP_K);
            let (predicted, confidence) = alternatives[0];
            BertPrediction {
                position,
                target,
                predicted,
                confidence,
                alternatives,
            }
        })
        .collect();

    graph.truncate(base_len)?;
    Ok(predictions)
}

/// Masked logits selected from a full `[seq_len, vocab_size]` tensor.
struct MaskedBatch<R: Runtime> {
    /// Selected logits with shape `[num_masked, vocab_size]`.
    selected: Tensor<R>,
    /// Selection matrix with shape `[num_masked, seq_len]`.
    selection: Tensor<R>,
    /// One-hot target rows with shape `[num_masked, vocab_size]`.
    one_hot: Tensor<R>,
    /// Number of masked positions.
    count: usize,
}

fn masked_batch<R>(
    graph: &Graph<R>,
    logits: NodeId,
    example: &BertExample,
    vocab_size: usize,
    device: &R::Device,
) -> Result<MaskedBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let shape = graph.nodes[logits].data.shape();
    if shape.len() != 2 || shape[1] != vocab_size {
        return invalid(
            "logits",
            format!("expected shape [seq_len, {vocab_size}], got {shape:?}"),
        );
    }
    if example.masked.is_empty() {
        return invalid("masked", "must contain at least one position");
    }

    let sequence_len = shape[0];
    let count = example.masked.len();
    let mut selection = vec![0.0_f32; count * sequence_len];
    let mut targets = Vec::with_capacity(count);

    for (row, &(position, target)) in example.masked.iter().enumerate() {
        if position >= sequence_len {
            return invalid(
                "masked",
                format!("position {position} is outside a sequence of length {sequence_len}"),
            );
        }
        if target >= vocab_size {
            return invalid(
                "masked",
                format!("target {target} is outside the vocabulary"),
            );
        }

        selection[row * sequence_len + position] = 1.0;
        targets.push(target as i32);
    }

    let selection = Tensor::from_slice(&selection, &[count, sequence_len], device)?;
    let targets = Tensor::from_slice(&targets, &[count], device)?;
    let one_hot = targets.one_hot(vocab_size)?;
    let selected = selection.matmul(&graph.nodes[logits].data)?;

    Ok(MaskedBatch {
        selected,
        selection,
        one_hot,
        count,
    })
}

/// Computes mean masked cross-entropy and its gradient with respect to the
/// selected logits, without copying either to the host.
fn masked_cross_entropy<R>(
    selected_logits: &Tensor<R>,
    one_hot: &Tensor<R>,
    count: usize,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    if count == 0 {
        return invalid("count", "must be greater than zero");
    }

    let log_probabilities = selected_logits.log_softmax(-1)?;
    let target_log_probabilities = log_probabilities.mul(one_hot)?.sum(&[1], false)?;
    let loss = target_log_probabilities
        .sum(&[0], true)?
        .mul_scalar(-1.0 / count as f64)?;

    let probabilities = log_probabilities.exp()?;
    let gradient = probabilities.sub(one_hot)?.mul_scalar(1.0 / count as f64)?;

    Ok((loss, gradient))
}

fn masked_loss_and_gradient<R>(
    graph: &Graph<R>,
    logits: NodeId,
    example: &BertExample,
    vocab_size: usize,
    device: &R::Device,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let batch = masked_batch(graph, logits, example, vocab_size, device)?;
    let (loss, selected_gradient) =
        masked_cross_entropy(&batch.selected, &batch.one_hot, batch.count)?;

    // Scatter the selected rows back into a full [seq_len, vocab_size]
    // gradient by multiplying with the transpose of the selection matrix.
    let gradient = batch
        .selection
        .transpose(0, 1)?
        .contiguous()?
        .matmul(&selected_gradient)?;

    let logits_shape = graph.nodes[logits].data.shape();
    if gradient.shape() != logits_shape {
        return invalid(
            "gradient",
            format!(
                "expected shape {logits_shape:?}, got {:?}",
                gradient.shape()
            ),
        );
    }

    Ok((loss, gradient))
}

fn parameter_prefix_len(parameters: &[crate::autograd::NodeId]) -> usize {
    parameters.iter().copied().max().map_or(0, |last| last + 1)
}

fn invalid<T>(argument: &'static str, reason: impl Into<String>) -> Result<T> {
    Err(Error::InvalidArgument {
        arg: argument,
        reason: reason.into(),
    })
}

/// Runs the MLM demo on the selected backend.
///
/// Positional command-line arguments are `[DATA_TSV] [STEPS]`; `--cpu` and
/// `--spam` are ignored here because the binary has already filtered them.
///
/// # Errors
///
/// Returns an error if the corpus or step count is invalid, or if a tensor
/// operation fails.
pub fn run_mlm<R>(device: &R::Device) -> std::result::Result<(), Box<dyn StdError>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let mut arguments = env::args().skip(1).filter(|argument| argument != "--cpu");
    let data_path = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/eng-it-sample.tsv"));
    let steps = match arguments.next() {
        Some(value) => value.parse::<usize>()?,
        None => 2_000,
    };

    let pairs = load_parallel_pairs(&data_path)?;
    let (training_pairs, evaluation_pairs) = split_every_nth(&pairs, 10);

    // Train BPE merges only on the training split so evaluation text cannot
    // leak into the vocabulary. Rare evaluation words are composed from known
    // subword pieces instead of collapsing to `[UNK]`.
    let vocab = BertVocab::build_with_strategy(&training_pairs, 3_000, 1, EncodingStrategy::Bpe);
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

    let mut eval_rng = StdRng::seed_from_u64(101);
    let evaluation_examples: Vec<BertExample> = evaluation_pairs
        .iter()
        .filter_map(|pair| {
            build_mlm_example(pair, &vocab, MAX_LEN, MASK_PROBABILITY, &mut eval_rng)
        })
        .collect();

    let encoding_name = match vocab.strategy() {
        EncodingStrategy::Word => "word",
        EncodingStrategy::Bpe => "bpe",
    };

    println!("BERT masked-language-model pretraining");
    println!("  backend:           {}", R::name());
    println!("  encoding:          {encoding_name}");
    println!("  corpus:            {}", data_path.display());
    println!("  steps:             {}", steps);
    println!(
        "  pairs:             {} ({} train / {} evaluation)",
        pairs.len(),
        training_pairs.len(),
        evaluation_pairs.len()
    );
    println!("  vocabulary:        {} tokens", vocab.len());
    println!(
        "  evaluation masks:  {} tokens in {} examples",
        evaluation_examples
            .iter()
            .map(|example| example.masked.len())
            .sum::<usize>(),
        evaluation_examples.len()
    );
    println!(
        "  model:             {} layers, d_model={}, d_ff={}, {} heads",
        model.config.n_layers, model.config.d_model, model.config.d_ff, model.config.n_heads
    );
    println!();
    println!("Note: BERT is encoder-only. This trains masked-token prediction,");
    println!("      not English-to-Italian translation. BPE pieces are learned");
    println!("      from the training split only.");
    println!();

    let mut rng = StdRng::seed_from_u64(7);
    let history = train_bert(
        &mut graph,
        device,
        &model,
        BertTrainingSet {
            vocab: &vocab,
            train_pairs: &training_pairs,
            eval_examples: &evaluation_examples,
        },
        &BertTrainConfig {
            steps,
            progress_interval: steps.div_ceil(20).max(1),
            mask_probability: MASK_PROBABILITY,
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
            "  masked accuracy: {:.1}% ({} tokens)",
            evaluation.masked_accuracy * 100.0,
            evaluation.masked_tokens
        );
    }

    print_examples(&mut graph, device, &model, &vocab, &evaluation_examples)?;
    Ok(())
}

/// Prints a few masked contexts together with the model's best prediction.
fn print_examples<R>(
    graph: &mut Graph<R>,
    device: &R::Device,
    model: &Bert,
    vocab: &BertVocab,
    examples: &[BertExample],
) -> std::result::Result<(), Box<dyn StdError>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    println!();
    println!("Sample masked-token predictions:");
    println!("(other masks in each example are restored so the context is readable)");
    for example in examples.iter().take(5) {
        let Some(prediction) = predict_masked_tokens(graph, device, model, example)?
            .into_iter()
            .next()
        else {
            continue;
        };
        println!(
            "  ... {} ...",
            display_context(example, prediction.position, vocab)
        );
        let target_rank = prediction
            .alternatives
            .iter()
            .position(|&(id, _)| id == prediction.target)
            .map_or_else(
                || "outside top 5".to_string(),
                |rank| format!("rank {}", rank + 1),
            );
        println!(
            "      target {:<14} prediction {:<14} confidence {:>5.1}%  ({})",
            vocab.token(prediction.target).unwrap_or("[UNK]"),
            vocab.token(prediction.predicted).unwrap_or("[UNK]"),
            prediction.confidence * 100.0,
            target_rank
        );
        let candidates = prediction
            .alternatives
            .iter()
            .map(|&(id, probability)| {
                format!(
                    "{}={:.1}%",
                    vocab.token(id).unwrap_or("[UNK]"),
                    probability * 100.0
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("      top 5: {candidates}");
    }
    Ok(())
}

/// Restores every mask except `position`, so the displayed context shows only
/// the token the model is being asked to predict.
fn display_context(example: &BertExample, position: usize, vocab: &BertVocab) -> String {
    let mut token_ids = example.token_ids.clone();
    for &(masked_position, target) in &example.masked {
        token_ids[masked_position] = target;
    }
    token_ids[position] = BERT_MASK_TOKEN;
    vocab.decode(&token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bert::BertConfig;
    use numr::runtime::cpu::CpuDevice;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn pairs() -> Vec<ParallelPair> {
        vec![
            ParallelPair::new("The cat is on the table.", "Il gatto è sul tavolo."),
            ParallelPair::new("I have a red book.", "Ho un libro rosso."),
            ParallelPair::new("Where is the station?", "Dov'è la stazione?"),
        ]
    }

    #[test]
    fn test_parse_parallel_pairs() -> io::Result<()> {
        let parsed = parse_parallel_pairs("Hello\tCiao\n\nWorld\tMondo\n")?;
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], ParallelPair::new("Hello", "Ciao"));
        Ok(())
    }

    #[test]
    fn test_build_mlm_example_layout() {
        let corpus = pairs();
        let vocab = BertVocab::build_with_strategy(&corpus, 32, 1, EncodingStrategy::Word);
        let mut rng = StdRng::seed_from_u64(7);

        let example = build_mlm_example(&corpus[0], &vocab, 16, 0.15, &mut rng)
            .expect("example should contain content");

        assert!(example.token_ids.len() <= 16);
        assert_eq!(example.token_ids.len(), example.token_type_ids.len());
        assert_eq!(example.token_ids[0], BERT_CLS_TOKEN);
        assert_eq!(*example.token_ids.last().unwrap(), BERT_SEP_TOKEN);
        assert!(!example.masked.is_empty());
        assert!(example.masked.iter().all(|(position, target)| {
            *position < example.token_ids.len() && *target >= BERT_NUM_SPECIAL_TOKENS
        }));
    }

    #[test]
    fn test_build_mlm_example_supports_bpe() {
        // Repeat the tiny corpus so BPE pieces pass the mask-frequency rule.
        let corpus: Vec<ParallelPair> = (0..60).flat_map(|_| pairs()).collect();
        let vocab = BertVocab::build_with_strategy(&corpus, 64, 1, EncodingStrategy::Bpe);
        let mut rng = StdRng::seed_from_u64(7);

        let example = build_mlm_example(&corpus[0], &vocab, 16, 0.5, &mut rng)
            .expect("example should contain content");

        assert_eq!(vocab.strategy(), EncodingStrategy::Bpe);
        assert!(
            example
                .masked
                .iter()
                .all(|&(_, target)| vocab.is_maskable(target))
        );
        assert!(vocab.decode(&example.token_ids).starts_with("[CLS]"));
    }

    #[test]
    fn test_display_context_restores_other_masks() {
        let corpus = pairs();
        let vocab = BertVocab::build_with_strategy(&corpus, 32, 1, EncodingStrategy::Word);
        let example = BertExample {
            token_ids: vec![
                BERT_CLS_TOKEN,
                BERT_MASK_TOKEN,
                BERT_MASK_TOKEN,
                BERT_SEP_TOKEN,
            ],
            token_type_ids: vec![0, 0, 0, 0],
            masked: vec![(1, vocab.id("gatto")), (2, vocab.id("tavolo"))],
        };

        assert_eq!(
            display_context(&example, 1, &vocab),
            "[CLS] [MASK] tavolo [SEP]"
        );
    }

    #[test]
    fn test_bundled_corpus_loads() -> io::Result<()> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/eng-it-sample.tsv");
        let pairs = load_parallel_pairs(path)?;
        assert_eq!(pairs.len(), 2_000);
        assert!(pairs.iter().all(|pair| {
            !pair.english.is_empty() && !pair.italian.is_empty() && pair.english != pair.italian
        }));
        Ok(())
    }

    #[test]
    fn test_train_bert_reduces_masked_loss() -> Result<()> {
        let device = CpuDevice::new();
        let corpus = pairs();
        let vocab = BertVocab::build_with_strategy(&corpus, 32, 1, EncodingStrategy::Word);
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
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

        let eval_rng = &mut StdRng::seed_from_u64(11);
        let eval_examples: Vec<BertExample> = corpus
            .iter()
            .filter_map(|pair| build_mlm_example(pair, &vocab, 12, 0.5, eval_rng))
            .collect();
        let mut rng = StdRng::seed_from_u64(7);

        // `train_bert` no longer evaluates before training, so take the
        // baseline explicitly if the test needs a before/after comparison.
        let initial = evaluate_bert(&mut graph, &device, &model, &eval_examples, 0)?;

        let history = train_bert(
            &mut graph,
            &device,
            &model,
            BertTrainingSet {
                vocab: &vocab,
                train_pairs: &corpus,
                eval_examples: &eval_examples,
            },
            &BertTrainConfig {
                steps: 100,
                progress_interval: 0,
                mask_probability: 0.5,
                optimizer: AdamWConfig {
                    learning_rate: 5e-3,
                    ..AdamWConfig::default()
                },
            },
            &mut rng,
        )?;

        let final_evaluation = history.final_evaluation.expect("final evaluation");
        assert!(
            final_evaluation.loss < initial.loss,
            "expected loss to decrease, got {} -> {}",
            initial.loss,
            final_evaluation.loss
        );
        Ok(())
    }

    #[test]
    fn test_train_bert_runs_and_updates_parameters() -> Result<()> {
        let device = CpuDevice::new();
        let corpus = pairs();
        let vocab = BertVocab::build_with_strategy(&corpus, 32, 1, EncodingStrategy::Word);
        let mut graph = Graph::<numr::runtime::cpu::CpuRuntime>::new();
        let model = Bert::new(
            &mut graph,
            &device,
            BertConfig {
                vocab_size: vocab.len(),
                d_model: 8,
                d_ff: 16,
                n_heads: 2,
                n_layers: 1,
                max_len: 12,
                tie_word_embeddings: true,
            },
        )?;

        let parameters = model.parameters();
        let before: Vec<Vec<f32>> = parameters
            .iter()
            .map(|&parameter| graph.nodes[parameter].data.to_vec::<f32>())
            .collect();

        let eval_rng = &mut StdRng::seed_from_u64(11);
        let eval_examples: Vec<BertExample> = corpus
            .iter()
            .filter_map(|pair| build_mlm_example(pair, &vocab, 12, 0.5, eval_rng))
            .collect();
        let mut rng = StdRng::seed_from_u64(7);
        let history = train_bert(
            &mut graph,
            &device,
            &model,
            BertTrainingSet {
                vocab: &vocab,
                train_pairs: &corpus,
                eval_examples: &eval_examples,
            },
            &BertTrainConfig {
                steps: 2,
                progress_interval: 0,
                mask_probability: 0.5,
                optimizer: AdamWConfig {
                    learning_rate: 1e-2,
                    ..AdamWConfig::default()
                },
            },
            &mut rng,
        )?;

        assert_eq!(history.losses.len(), 2);
        assert!(history.losses.iter().all(|loss| loss.is_finite()));
        let final_evaluation = history.final_evaluation.expect("final evaluation");
        assert_eq!(final_evaluation.step, 2);
        assert!(final_evaluation.loss.is_finite());
        assert!((0.0..=1.0).contains(&final_evaluation.masked_accuracy));
        assert!(final_evaluation.masked_tokens > 0);

        let after: Vec<Vec<f32>> = parameters
            .iter()
            .map(|&parameter| graph.nodes[parameter].data.to_vec::<f32>())
            .collect();
        assert!(
            before
                .iter()
                .zip(&after)
                .any(|(before, after)| before != after),
            "training should update at least one parameter"
        );

        Ok(())
    }
}
