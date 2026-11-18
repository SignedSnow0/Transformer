//! Training utilities for the toy transformer.
//!
//! This module provides the softmax cross-entropy loss (with its gradient with
//! respect to logits) and a sequence-reversal task with an AdamW training
//! loop. The loss is computed outside the autograd graph and fed back through
//! [`crate::autograd::Graph::backward_with_grad`], which keeps the autograd
//! primitives small and general.

mod loss;
mod reversal;

pub use reversal::{
    EOS_TOKEN, FIRST_DIGIT_TOKEN, SOS_TOKEN, TrainConfig, VOCAB_SIZE, sample_reversal,
    train_reversal,
};
