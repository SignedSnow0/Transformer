//! Softmax cross-entropy loss and its gradient with respect to logits.
//!
//! The loss is computed outside the autograd graph so that training can seed
//! [`crate::autograd::Graph::backward_with_grad`] directly with
//! `dL/dlogits = (softmax(logits) - one_hot(target)) / num_tokens`.

use numr::error::Result;

/// Computes mean softmax cross-entropy over a batch of token predictions.
///
/// # Arguments
///
/// * `logits` - Flat, row-major logits with shape `[num_tokens, vocab_size]`
/// * `targets` - Correct token index for each row
/// * `vocab_size` - Number of columns per row
///
/// # Returns
///
/// The mean loss and a flat gradient with the same layout as `logits`. The
/// gradient is already averaged over rows, matching the returned loss.
///
/// # Errors
///
/// Returns an error if `vocab_size` is zero, if `targets` is empty, if the
/// logits length does not match `targets.len() * vocab_size`, if a target is
/// out of range, or if a row does not form a finite probability distribution.
pub fn softmax_cross_entropy(
    logits: &[f32],
    targets: &[usize],
    vocab_size: usize,
) -> Result<(f32, Vec<f32>)> {
    let mut total_loss = 0.0_f32;
    let mut gradient = vec![0.0_f32; logits.len()];
    let inverse_count = 1.0 / targets.len() as f32;

    for (row, &target) in targets.iter().enumerate() {
        let start = row * vocab_size;
        let row_logits = &logits[start..start + vocab_size];
        let max_logit = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut probabilities = vec![0.0_f32; vocab_size];
        let mut sum = 0.0_f32;
        for (probability, &logit) in probabilities.iter_mut().zip(row_logits) {
            *probability = (logit - max_logit).exp();
            sum += *probability;
        }

        let log_probability = row_logits[target] - max_logit - sum.ln();
        total_loss -= log_probability;

        for (column, probability) in probabilities.iter().enumerate() {
            let indicator = if column == target { 1.0 } else { 0.0 };
            gradient[start + column] = (probability / sum - indicator) * inverse_count;
        }
    }

    total_loss *= inverse_count;
    Ok((total_loss, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_uniform_logits_loss_and_gradient() -> Result<()> {
        let vocab_size = 4;
        let targets = [1, 3];
        let logits = vec![0.0_f32; targets.len() * vocab_size];

        let (loss, gradient) = softmax_cross_entropy(&logits, &targets, vocab_size)?;

        assert_close(loss, (vocab_size as f32).ln());
        for (row, &target) in targets.iter().enumerate() {
            let start = row * vocab_size;
            let row_gradient = &gradient[start..start + vocab_size];
            assert_close(row_gradient.iter().sum(), 0.0);
            for (column, &value) in row_gradient.iter().enumerate() {
                let expected = (if column == target { 0.25 - 1.0 } else { 0.25 }) * 0.5;
                assert_close(value, expected);
            }
        }

        Ok(())
    }

    #[test]
    fn test_confident_prediction_has_small_loss() -> Result<()> {
        let logits = [20.0_f32, -20.0];
        let (loss, _) = softmax_cross_entropy(&logits, &[0], 2)?;
        assert!(loss < 1e-6);
        Ok(())
    }

    #[test]
    fn test_gradient_matches_finite_difference() -> Result<()> {
        let logits = [0.3_f32, -0.2, 0.7, 0.1, -0.4, 0.2];
        let targets = [2, 0];
        let vocab_size = 3;

        let (_, gradient) = softmax_cross_entropy(&logits, &targets, vocab_size)?;
        let epsilon = 1e-3;

        for index in 0..logits.len() {
            let mut plus = logits;
            plus[index] += epsilon;
            let mut minus = logits;
            minus[index] -= epsilon;

            let (loss_plus, _) = softmax_cross_entropy(&plus, &targets, vocab_size)?;
            let (loss_minus, _) = softmax_cross_entropy(&minus, &targets, vocab_size)?;
            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);

            assert!(
                (gradient[index] - numerical).abs() < 1e-4,
                "gradient mismatch at {index}: analytic {}, numerical {}",
                gradient[index],
                numerical
            );
        }

        Ok(())
    }
}
