//! Complete English–Italian masked-language-model example.
//!
//! This file contains the data loader, masking, training loop, evaluation,
//! predictions, and the [`run_mlm`] entry point used by the binary. The training
//! objective is BERT's masked language model: a random subset of content tokens
//! is replaced by `[MASK]`, a random token, or left unchanged, and the model is
//! trained to recover the original tokens from bidirectional context. The demo
//! uses the learned BPE [`EncodingStrategy`], so rare words are encoded as
//! subword pieces rather than `[UNK]`; all pieces of a selected word are masked
//! together.
//! Validation reports both top-1 and top-5 masked-token accuracy on a fixed,
//! deterministically sampled subset of the held-out pairs so long runs stay
//! practical.
//! Tokenization and vocabulary lookup live in [`crate::bert::encoding`].
//!
//! The default `data/eng-it-full.tsv` is generated from OPUS Tatoeba. See
//! `data/README.md` for provenance and regeneration commands.

use super::{
    Checkpoint, DEFAULT_MLM_DATA, DEFAULT_MLM_STEPS, EarlyStopping, ExampleArgs, MASK_PROBABILITY,
    MLM_MAX_EVAL_EXAMPLES, MLM_VOCAB_CORPUS_PAIRS, MLM_VOCAB_SIZE, MLM_WARMUP_STEPS,
    reservoir_sample, split_every_nth,
};
use crate::autograd::{Graph, NodeId};
use crate::bert::Bert;
use crate::bert::encoding::{
    BERT_CLS_TOKEN, BERT_MASK_TOKEN, BERT_NUM_SPECIAL_TOKENS, BERT_SEP_TOKEN, BERT_UNK_TOKEN,
    BertVocab, EncodingStrategy, ParallelPair, tokenize,
};
use crate::training::{AdamW, AdamWConfig, LRSchedule};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::prelude::*;
use numr::runtime::Runtime;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error as StdError;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Number of candidate tokens shown and counted for top-k accuracy.
const TOP_K: usize = 5;
/// Number of held-out examples displayed after training.
const DISPLAY_EXAMPLES: usize = 5;

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
/// works with whole words or BPE pieces. With BPE, every piece of a selected
/// word is masked together so the objective predicts whole words rather than
/// isolated fragments. At least one maskable word is always selected. Returns
/// `None` if the pair contains none after truncation.
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

    let pre_tokens_for_english = tokenize(&pair.english);
    let english_tokens = vocab.encode_tokens(&pre_tokens_for_english);
    let pre_tokens_for_italian = tokenize(&pair.italian);
    let italian_tokens = vocab.encode_tokens(&pre_tokens_for_italian);

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
    let mut word_groups: Vec<Vec<usize>> = Vec::new();

    token_ids.push(BERT_CLS_TOKEN);
    token_type_ids.push(0);
    append_word_groups(
        &mut word_groups,
        &vocab.encode_token_groups(&pre_tokens_for_english),
        english_len,
        1,
    );
    for &token in &english_tokens[..english_len] {
        token_ids.push(token);
        token_type_ids.push(0);
    }
    token_ids.push(BERT_SEP_TOKEN);
    token_type_ids.push(0);

    let italian_offset = token_ids.len();
    append_word_groups(
        &mut word_groups,
        &vocab.encode_token_groups(&pre_tokens_for_italian),
        italian_len,
        italian_offset,
    );
    for &token in &italian_tokens[..italian_len] {
        token_ids.push(token);
        token_type_ids.push(1);
    }
    token_ids.push(BERT_SEP_TOKEN);
    token_type_ids.push(1);

    let candidates: Vec<Vec<usize>> = word_groups
        .into_iter()
        .filter_map(|group| {
            let maskable: Vec<usize> = group
                .into_iter()
                .filter(|&position| vocab.is_maskable(token_ids[position]))
                .collect();
            (!maskable.is_empty()).then_some(maskable)
        })
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
    for group in &candidates {
        if rng.random::<f32>() < probability {
            apply_word_mask(&mut token_ids, group, vocab.len(), rng, &mut masked);
        }
    }

    if masked.is_empty() {
        let group = &candidates[rng.random_range(0..candidates.len())];
        apply_word_mask(&mut token_ids, group, vocab.len(), rng, &mut masked);
    }

    Some(BertExample {
        token_ids,
        token_type_ids,
        masked,
    })
}

/// Translates per-word piece positions into sequence positions after truncation.
fn append_word_groups(
    groups: &mut Vec<Vec<usize>>,
    encoded_groups: &[Vec<usize>],
    kept_tokens: usize,
    offset: usize,
) {
    for group in encoded_groups {
        let kept: Vec<usize> = group
            .iter()
            .filter_map(|&position| (position < kept_tokens).then_some(offset + position))
            .collect();
        if !kept.is_empty() {
            groups.push(kept);
        }
    }
}

/// Applies the same 80/10/10 masking decision to every piece of one word.
fn apply_word_mask(
    token_ids: &mut [usize],
    positions: &[usize],
    vocab_size: usize,
    rng: &mut impl RngExt,
    masked: &mut Vec<(usize, usize)>,
) {
    let roll = rng.random::<f32>();
    let replacement = if roll < 0.8 {
        Some(BERT_MASK_TOKEN)
    } else if roll < 0.9 {
        let content_size = vocab_size.saturating_sub(BERT_NUM_SPECIAL_TOKENS);
        Some(if content_size == 0 {
            BERT_UNK_TOKEN
        } else {
            BERT_NUM_SPECIAL_TOKENS + rng.random_range(0..content_size)
        })
    } else {
        None
    };

    for &position in positions {
        let target = token_ids[position];
        if let Some(replacement) = replacement {
            token_ids[position] = replacement;
        }
        masked.push((position, target));
    }
}

/// Training settings for BERT MLM.
#[derive(Debug, Clone, Copy)]
pub struct BertTrainConfig {
    /// Number of optimizer steps.
    pub steps: usize,
    /// Print a progress line every this many steps. Zero disables progress.
    pub progress_interval: usize,
    /// Probability of selecting a content word for masking.
    pub mask_probability: f32,
    /// AdamW hyperparameters; the learning rate acts as the schedule's peak.
    pub optimizer: AdamWConfig,
    /// Learning-rate warmup and cosine decay.
    pub schedule: LRSchedule,
    /// Validate every this many steps. Zero validates only at the start/end.
    pub validation_interval: usize,
    /// Stop after this many validations without improvement. Zero disables it.
    pub patience: usize,
    /// Minimum loss decrease counted as an improvement.
    pub min_delta: f32,
    /// Restore the best validation parameters before returning.
    pub restore_best: bool,
}

impl Default for BertTrainConfig {
    fn default() -> Self {
        let validation_interval = DEFAULT_MLM_STEPS.div_ceil(20);
        Self {
            steps: DEFAULT_MLM_STEPS,
            progress_interval: 0,
            mask_probability: MASK_PROBABILITY,
            optimizer: AdamWConfig {
                learning_rate: 1e-3,
                weight_decay: 0.01,
                ..AdamWConfig::default()
            },
            schedule: LRSchedule {
                warmup_steps: MLM_WARMUP_STEPS.min(validation_interval),
                min_lr_ratio: 0.05,
            },
            validation_interval,
            patience: 8,
            min_delta: 1e-4,
            restore_best: true,
        }
    }
}

impl BertTrainConfig {
    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.mask_probability) {
            return invalid("mask_probability", "must be in [0, 1]");
        }
        if !self.min_delta.is_finite() || self.min_delta < 0.0 {
            return invalid("min_delta", "must be finite and non-negative");
        }
        self.schedule.validate()
    }
}

/// Evaluation result for a fixed set of masked examples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertEvaluation {
    /// Optimizer step at which evaluation ran.
    pub step: usize,
    /// Mean masked-token cross-entropy loss.
    pub loss: f32,
    /// Fraction of masked tokens predicted correctly.
    pub masked_accuracy: f32,
    /// Fraction of masked tokens whose target appears in the top five logits.
    pub top5_accuracy: f32,
    /// Number of masked tokens evaluated.
    pub masked_tokens: usize,
}

/// Losses, validation scores, and checkpoints collected during BERT training.
#[derive(Debug, Clone, Default)]
pub struct BertTrainingHistory {
    /// Training loss for every step.
    pub losses: Vec<f32>,
    /// Learning rate used at every step.
    pub learning_rates: Vec<f32>,
    /// Evaluation at step zero and every validation point.
    pub evaluations: Vec<BertEvaluation>,
    /// Best evaluation by masked-token loss.
    pub best_evaluation: Option<BertEvaluation>,
    /// Held-out evaluation of the parameters left in the graph.
    pub final_evaluation: Option<BertEvaluation>,
    /// Number of optimizer steps actually run.
    pub steps_run: usize,
    /// Whether early stopping ended training before `steps`.
    pub stopped_early: bool,
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
/// [`Graph::truncate`](crate::autograd::Graph::truncate). Every step draws a
/// fresh pair and a fresh mask, so a sentence can be reused with different
/// masked tokens instead of being seen only once.
///
/// Evaluation runs at step zero, at every `validation_interval`, and at the
/// final step. Improvements are checkpointed on the device; if early stopping
/// fires or `restore_best` is enabled, the best parameters are put back into the
/// graph before the final evaluation.
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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    if data.train_pairs.is_empty() {
        return invalid("train_pairs", "must not be empty");
    }
    if data.eval_examples.is_empty() {
        return invalid("eval_examples", "must not be empty");
    }
    config.validate()?;

    let parameters = model.parameters();
    let base_len = parameter_prefix_len(&parameters);
    graph.truncate(base_len)?;

    let mut optimizer = AdamW::<R>::new(config.optimizer, &parameters)?;
    let mut history = BertTrainingHistory::default();
    let mut stopping = EarlyStopping::new();
    let mut best_checkpoint: Option<Checkpoint<R>> = None;

    let initial = evaluate_bert(graph, device, model, data.eval_examples, 0)?;
    history.evaluations.push(initial);
    let (improved, _) = stopping.observe(0, initial.loss, config.min_delta, config.patience);
    if improved {
        history.best_evaluation = Some(initial);
        best_checkpoint = Some(Checkpoint::capture(graph, &parameters, 0)?);
    }

    // Keep each step's scalar loss on the device and read the whole vector
    // once after training. This keeps the optimization loop free of
    // device-to-host copies.
    let mut step_losses: Vec<Tensor<R>> = Vec::with_capacity(config.steps);
    let peak_learning_rate = config.optimizer.learning_rate;
    let started = Instant::now();
    let mut last_step = 0;
    let mut stopped_early = false;

    for step in 1..=config.steps {
        let learning_rate = config
            .schedule
            .learning_rate(peak_learning_rate, step, config.steps);
        optimizer.set_learning_rate(learning_rate)?;
        history.learning_rates.push(learning_rate);

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

        let output = model.forward_with_dropout(
            graph,
            device,
            &example.token_ids,
            &example.token_type_ids,
            true,
            rng,
        )?;
        let (loss, gradient) =
            masked_loss_and_gradient(graph, output.logits, &example, model.vocab_size(), device)?;
        graph.backward_with_grad(output.logits, gradient)?;
        optimizer.step(graph)?;
        graph.truncate(base_len)?;
        step_losses.push(loss);
        last_step = step;

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

        let validate = step == config.steps
            || (config.validation_interval > 0 && step.is_multiple_of(config.validation_interval));
        if validate {
            let evaluation = evaluate_bert(graph, device, model, data.eval_examples, step)?;
            history.evaluations.push(evaluation);
            let (improved, should_stop) =
                stopping.observe(step, evaluation.loss, config.min_delta, config.patience);
            if improved {
                history.best_evaluation = Some(evaluation);
                best_checkpoint = Some(Checkpoint::capture(graph, &parameters, step)?);
            }
            if should_stop {
                stopped_early = step < config.steps;
                break;
            }
        }
    }

    history.steps_run = last_step;
    history.stopped_early = stopped_early;

    let final_step = best_checkpoint
        .as_ref()
        .filter(|_| config.restore_best)
        .map_or(last_step, Checkpoint::step);
    if config.restore_best
        && let Some(checkpoint) = best_checkpoint.as_ref()
    {
        checkpoint.restore(graph, &parameters)?;
        graph.truncate(base_len)?;
    }

    history.final_evaluation = Some(evaluate_bert(
        graph,
        device,
        model,
        data.eval_examples,
        final_step,
    )?);

    if !step_losses.is_empty() {
        let references: Vec<&Tensor<R>> = step_losses.iter().collect();
        history.losses = Tensor::cat(&references, 0)?.to_vec::<f32>();
    }

    Ok(history)
}

/// Draws a fresh training pair and mask.
///
/// Sampling with replacement is deliberate: a sentence can be revisited with a
/// different mask, which gives more training signal per sentence than a single
/// pass over the corpus.
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

/// Selects the fixed validation subset used during training.
fn sample_evaluation_examples(examples: &[BertExample], rng: &mut impl RngExt) -> Vec<BertExample> {
    reservoir_sample(examples, MLM_MAX_EVAL_EXAMPLES, rng)
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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    if examples.is_empty() {
        return invalid("examples", "must not be empty");
    }

    let base_len = parameter_prefix_len(&model.parameters());
    let vocab_size = model.vocab_size();
    let mut total_loss: Option<Tensor<R>> = None;
    let mut total_correct: Option<Tensor<R>> = None;
    let mut total_top5: Option<Tensor<R>> = None;
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

        // Rank by repeatedly taking the argmax and hiding that candidate. This
        // stays on the device and avoids reading a `[tokens, vocab]` matrix to
        // the host just to report top-k accuracy.
        let mut ranking_logits = batch.selected;
        let mut top5_hits: Option<Tensor<R>> = None;
        for _ in 0..TOP_K {
            let candidate = ranking_logits.argmax(1, false)?;
            let candidate_one_hot = candidate.one_hot(vocab_size)?;
            let hit = candidate_one_hot.mul(&batch.one_hot)?.sum(&[1], false)?;
            top5_hits = Some(match top5_hits {
                Some(total) => total.add(&hit)?,
                None => hit,
            });
            ranking_logits = ranking_logits.sub(&candidate_one_hot.mul_scalar(1e9_f64)?)?;
        }
        let top5_correct = top5_hits
            .expect("top-k loop always records a hit")
            .sum(&[0], true)?;
        total_top5 = Some(match total_top5 {
            Some(total) => total.add(&top5_correct)?,
            None => top5_correct,
        });

        masked_tokens += batch.count;
        graph.truncate(base_len)?;
    }

    let total_loss = total_loss.expect("examples are non-empty").to_vec::<f32>()[0];
    let total_correct = total_correct
        .expect("examples are non-empty")
        .to_vec::<f32>()[0];
    let total_top5 = total_top5.expect("examples are non-empty").to_vec::<f32>()[0];

    Ok(BertEvaluation {
        step,
        loss: total_loss / masked_tokens as f32,
        masked_accuracy: total_correct / masked_tokens as f32,
        top5_accuracy: total_top5 / masked_tokens as f32,
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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    let base_len = parameter_prefix_len(&model.parameters());
    let vocab_size = model.vocab_size();

    let output = model.forward(graph, device, &example.token_ids, &example.token_type_ids)?;
    let batch = masked_batch(graph, output.logits, example, vocab_size, device)?;

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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
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
/// Positional arguments are `[DATA_TSV] [STEPS]`. Capacity flags are
/// `--d-model`, `--d-ff`, `--n-heads`, `--layers`, `--max-len`, and `--dropout`;
/// `--cpu` and `--spam` are handled by the binary.
///
/// # Errors
///
/// Returns an error if an argument, corpus, or tensor operation is invalid.
pub fn run_mlm<R>(device: &R::Device) -> std::result::Result<(), Box<dyn StdError>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    let arguments = ExampleArgs::parse()?;
    let data_path = arguments
        .data_path
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MLM_DATA));
    let steps = arguments.steps.unwrap_or(DEFAULT_MLM_STEPS);
    let capacity = arguments.capacity;
    let max_len = capacity.max_len;

    let pairs = load_parallel_pairs(&data_path)?;
    let (training_pairs, evaluation_pairs) = split_every_nth(&pairs, 10);

    // Train BPE merges only on the training split so evaluation text cannot
    // leak into the vocabulary. Rare evaluation words are composed from known
    // subword pieces instead of collapsing to `[UNK]`.
    // BPE training is quadratic in the number of merges, so learn merges from a
    // deterministic subset of the training split instead of all 215k pairs.
    let vocab_rng = &mut StdRng::seed_from_u64(21);
    let vocab_pairs = reservoir_sample(&training_pairs, MLM_VOCAB_CORPUS_PAIRS, vocab_rng);
    let vocab =
        BertVocab::build_with_strategy(&vocab_pairs, MLM_VOCAB_SIZE, 1, EncodingStrategy::Bpe);
    let mut graph = Graph::<R>::new();
    let model = Bert::new(&mut graph, device, capacity.bert_config(vocab.len()))?;

    let mut eval_rng = StdRng::seed_from_u64(101);
    let all_evaluation_examples: Vec<BertExample> = evaluation_pairs
        .iter()
        .filter_map(|pair| {
            build_mlm_example(pair, &vocab, max_len, MASK_PROBABILITY, &mut eval_rng)
        })
        .collect();

    // Validation runs many times during training, so a fixed deterministic
    // subset is much cheaper than re-scoring every held-out pair. The full
    // held-out split is still visible to the sample display below.
    let evaluation_examples = sample_evaluation_examples(&all_evaluation_examples, &mut eval_rng);

    let encoding_name = match vocab.strategy() {
        EncodingStrategy::Word => "word",
        EncodingStrategy::Bpe => "bpe",
    };
    let validation_interval = steps.div_ceil(20).max(1);

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
    println!(
        "  vocabulary:        {} tokens (BPE from {} training pairs)",
        vocab.len(),
        vocab_pairs.len()
    );
    println!(
        "  evaluation masks:  {} tokens in {} examples (sampled from {})",
        evaluation_examples
            .iter()
            .map(|example| example.masked.len())
            .sum::<usize>(),
        evaluation_examples.len(),
        all_evaluation_examples.len()
    );
    println!(
        "  model:             {} layers, d_model={}, d_ff={}, {} heads, max_len={}, dropout={:.2}",
        model.config.n_layers,
        model.config.d_model,
        model.config.d_ff,
        model.config.n_heads,
        model.config.max_len,
        model.config.dropout
    );
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
            progress_interval: validation_interval,
            mask_probability: MASK_PROBABILITY,
            optimizer: AdamWConfig {
                learning_rate: 1e-3,
                weight_decay: 0.01,
                ..Default::default()
            },
            schedule: LRSchedule {
                warmup_steps: MLM_WARMUP_STEPS.min(validation_interval),
                min_lr_ratio: 0.05,
            },
            validation_interval,
            ..BertTrainConfig::default()
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
        println!(
            "Final result (step {} of {}):",
            evaluation.step, history.steps_run
        );
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
        println!(
            "  top-5 accuracy:  {:.1}% ({} tokens)",
            evaluation.top5_accuracy * 100.0,
            evaluation.masked_tokens
        );
        if let Some(best) = history.best_evaluation {
            println!("  best validation: {:.4} at step {}", best.loss, best.step);
        }
        if history.stopped_early {
            println!("  early stopping ended training before the requested steps");
        }
    }

    // Pick fresh held-out sentences for the sample display on every run. The
    // validation metric above uses its own fixed subset, so this does not make
    // the reported accuracy nondeterministic.
    let sample_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64)
        ^ u64::from(std::process::id());
    let mut display_rng = StdRng::seed_from_u64(sample_seed);
    let display_examples =
        reservoir_sample(&all_evaluation_examples, DISPLAY_EXAMPLES, &mut display_rng);
    print_examples(&mut graph, device, &model, &vocab, &display_examples)?;
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
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    println!();
    println!("Sample masked-token predictions:");
    for example in examples.iter().take(DISPLAY_EXAMPLES) {
        let Some(prediction) = predict_masked_tokens(graph, device, model, example)?
            .into_iter()
            .next()
        else {
            continue;
        };
        println!("  ... {} ...", vocab.decode(&example.token_ids));
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
    fn test_bpe_masking_covers_whole_word() {
        let corpus: Vec<ParallelPair> = (0..60)
            .map(|_| ParallelPair::new("stazione", "gatto"))
            .collect();
        let vocab = BertVocab::build_with_strategy(&corpus, 24, 1, EncodingStrategy::Bpe);
        let pair = ParallelPair::new("stazione", "gatto");
        let mut rng = StdRng::seed_from_u64(7);
        let example = build_mlm_example(&pair, &vocab, 32, 1.0, &mut rng)
            .expect("example should contain content");

        let words = tokenize("stazione");
        let groups = vocab.encode_token_groups(&words);
        let original = vocab.encode_tokens(&words);
        let maskable_positions: Vec<usize> = groups[0]
            .iter()
            .filter_map(|&position| {
                vocab
                    .is_maskable(original[position])
                    .then_some(1 + position)
            })
            .collect();

        assert!(!maskable_positions.is_empty());
        assert!(
            maskable_positions
                .iter()
                .all(|&position| example.masked.iter().any(|&(masked, _)| masked == position)),
            "every maskable piece of a selected word must be masked: {maskable_positions:?}"
        );
    }

    #[test]
    fn test_sample_evaluation_examples_is_deterministic() {
        let examples: Vec<BertExample> = (0..MLM_MAX_EVAL_EXAMPLES + 20)
            .map(|id| BertExample {
                token_ids: vec![id],
                token_type_ids: vec![0],
                masked: vec![(0, id)],
            })
            .collect();
        let mut first_rng = StdRng::seed_from_u64(101);
        let mut second_rng = StdRng::seed_from_u64(101);

        let first = sample_evaluation_examples(&examples, &mut first_rng);
        let second = sample_evaluation_examples(&examples, &mut second_rng);

        assert_eq!(first.len(), MLM_MAX_EVAL_EXAMPLES);
        assert_eq!(first, second);
    }

    #[test]
    fn test_full_corpus_loads() -> io::Result<()> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/eng-it-full.tsv");
        let pairs = load_parallel_pairs(path)?;
        assert_eq!(pairs.len(), 239_245);
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
                dropout: 0.0,
                tie_word_embeddings: true,
            },
        )?;

        let eval_rng = &mut StdRng::seed_from_u64(11);
        let eval_examples: Vec<BertExample> = corpus
            .iter()
            .filter_map(|pair| build_mlm_example(pair, &vocab, 12, 0.5, eval_rng))
            .collect();
        let mut rng = StdRng::seed_from_u64(7);

        // `train_bert` records a step-zero evaluation; take the same baseline
        // here for an explicit before/after comparison.
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
                validation_interval: 0,
                ..BertTrainConfig::default()
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
                dropout: 0.0,
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
                validation_interval: 0,
                restore_best: false,
                ..BertTrainConfig::default()
            },
            &mut rng,
        )?;

        assert_eq!(history.losses.len(), 2);
        assert_eq!(history.learning_rates.len(), 2);
        assert_eq!(history.steps_run, 2);
        assert!(!history.stopped_early);
        assert!(history.evaluations.len() >= 2);
        assert!(history.best_evaluation.is_some());
        assert!(history.losses.iter().all(|loss| loss.is_finite()));
        let final_evaluation = history.final_evaluation.expect("final evaluation");
        assert_eq!(final_evaluation.step, 2);
        assert!(final_evaluation.loss.is_finite());
        assert!((0.0..=1.0).contains(&final_evaluation.masked_accuracy));
        assert!(final_evaluation.top5_accuracy >= final_evaluation.masked_accuracy);
        assert!(final_evaluation.top5_accuracy <= 1.0);
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
