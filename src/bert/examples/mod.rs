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

/// Probability used by the masked-language-model example.
pub(crate) const MASK_PROBABILITY: f32 = 0.15;
/// Maximum sequence length used by both examples, including special tokens.
pub(crate) const MAX_LEN: usize = 32;

/// Holds out every `nth` item for evaluation and returns `(train, evaluation)`.
///
/// Both examples use this helper with `nth = 10`, so their splits are
/// deterministic without needing to shuffle the bundled corpora.
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
