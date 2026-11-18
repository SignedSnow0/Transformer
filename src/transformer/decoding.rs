//! Token selection strategies for autoregressive decoding.
//!
//! Two strategies are provided:
//!
//! - **Greedy**: always choose the most likely token. This is the right default
//!   for tasks with a single correct answer, such as the toy reversal task, and
//!   makes evaluation reproducible.
//! - **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative
//!   probability reaches `p`, renormalize, and sample from that set. This
//!   adapts the candidate set to the shape of the distribution and is generally
//!   preferable to top-k for open-ended generation. Temperature controls how
//!   sharp the distribution is before the nucleus is formed.
//!
//! Top-p sampling introduces output variance, so it is useful for qualitative
//! generation but not the best choice for measuring exact-match accuracy.

use numr::error::{Error, Result};
use std::cmp::Ordering;

/// Strategy used to choose the next token from a vector of logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Always choose the highest-probability token.
    Greedy,
    /// Nucleus sampling with probability mass `p` and softmax `temperature`.
    TopP {
        /// Cumulative probability mass to keep; must be in `(0, 1]`.
        p: f32,
        /// Softmax temperature; must be finite and greater than zero.
        temperature: f32,
    },
}

impl SamplingStrategy {
    /// Validates the strategy's parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is outside `(0, 1]` or `temperature` is not a
    /// finite positive number.
    pub fn validate(&self) -> Result<()> {
        match *self {
            SamplingStrategy::Greedy => Ok(()),
            SamplingStrategy::TopP { p, temperature } => {
                if !p.is_finite() || p <= 0.0 || p > 1.0 {
                    return Err(Error::InvalidArgument {
                        arg: "p",
                        reason: format!("top-p must be in (0, 1], got {p}"),
                    });
                }
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(Error::InvalidArgument {
                        arg: "temperature",
                        reason: format!(
                            "temperature must be finite and greater than zero, got {temperature}"
                        ),
                    });
                }
                Ok(())
            }
        }
    }
}

/// Configuration for autoregressive generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodingConfig {
    /// Maximum number of content tokens to generate.
    pub max_new_tokens: usize,
    /// Token selection strategy.
    pub strategy: SamplingStrategy,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            strategy: SamplingStrategy::Greedy,
        }
    }
}

/// Selects one token index from a vector of logits.
///
/// `uniform` is a sample from `[0, 1)` used by top-p sampling; it is ignored by
/// greedy decoding. Exposing it as a parameter keeps the sampler deterministic
/// and easy to test.
///
/// # Errors
///
/// Returns an error if `logits` is empty, if the strategy is invalid, if
/// `uniform` is outside `[0, 1)` for top-p sampling, or if the logits cannot be
/// converted to a finite probability distribution.
pub fn select_token(logits: &[f32], strategy: SamplingStrategy, uniform: f32) -> Result<usize> {
    if logits.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: "token selection requires at least one logit".to_string(),
        });
    }

    strategy.validate()?;

    match strategy {
        SamplingStrategy::Greedy => Ok(argmax(logits)),
        SamplingStrategy::TopP { p, temperature } => {
            if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
                return Err(Error::InvalidArgument {
                    arg: "uniform",
                    reason: format!("uniform sample must be in [0, 1), got {uniform}"),
                });
            }
            top_p(logits, p, temperature, uniform)
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = index;
        }
    }
    best
}

fn softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: "logits must contain at least one finite value".to_string(),
        });
    }

    let mut probabilities = Vec::with_capacity(logits.len());
    let mut sum = 0.0_f32;
    for &logit in logits {
        let value = ((logit - max_logit) / temperature).exp();
        probabilities.push(value);
        sum += value;
    }

    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: "logits do not form a finite probability distribution".to_string(),
        });
    }

    for probability in &mut probabilities {
        *probability /= sum;
    }

    Ok(probabilities)
}

fn top_p(logits: &[f32], p: f32, temperature: f32, uniform: f32) -> Result<usize> {
    let probabilities = softmax(logits, temperature)?;

    let mut candidates: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut cumulative = 0.0_f32;
    let mut keep = 0;
    for (_, probability) in &candidates {
        cumulative += *probability;
        keep += 1;
        if cumulative >= p {
            break;
        }
    }

    // `keep` is at least one because `logits` is non-empty and probabilities
    // form a distribution with total mass one.
    let nucleus = &candidates[..keep];
    let total: f32 = nucleus.iter().map(|(_, probability)| *probability).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: "nucleus probabilities do not sum to a positive finite value".to_string(),
        });
    }

    let mut threshold = uniform * total;
    for (index, probability) in nucleus {
        if threshold < *probability {
            return Ok(*index);
        }
        threshold -= *probability;
    }

    Ok(nucleus.last().expect("top-p nucleus is non-empty").0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_selects_argmax() -> Result<()> {
        assert_eq!(
            select_token(&[0.1, 2.0, -1.0], SamplingStrategy::Greedy, 0.0)?,
            1
        );
        Ok(())
    }

    #[test]
    fn test_top_p_tiny_mass_selects_argmax() -> Result<()> {
        let strategy = SamplingStrategy::TopP {
            p: 1e-6,
            temperature: 1.0,
        };
        for uniform in [0.0, 0.5, 0.999] {
            assert_eq!(select_token(&[0.1, 2.0, -1.0], strategy, uniform)?, 1);
        }
        Ok(())
    }

    #[test]
    fn test_top_p_full_mass_samples_entire_distribution() -> Result<()> {
        let strategy = SamplingStrategy::TopP {
            p: 1.0,
            temperature: 1.0,
        };

        // Equal probabilities: the lower uniform value stays on the first
        // candidate, while the higher one crosses into the second.
        assert_eq!(select_token(&[0.0, 0.0], strategy, 0.25)?, 0);
        assert_eq!(select_token(&[0.0, 0.0], strategy, 0.75)?, 1);

        Ok(())
    }

    #[test]
    fn test_top_p_filters_distribution_tail() -> Result<()> {
        let strategy = SamplingStrategy::TopP {
            p: 0.9,
            temperature: 1.0,
        };

        // The first token holds essentially all of the mass, so even a high
        // uniform draw must still select it.
        for uniform in [0.0, 0.999] {
            assert_eq!(select_token(&[10.0, 0.0, 0.0], strategy, uniform)?, 0);
        }

        Ok(())
    }

    #[test]
    fn test_temperature_sharpens_distribution() -> Result<()> {
        let cold = softmax(&[0.0, 1.0], 0.1)?;
        let hot = softmax(&[0.0, 1.0], 10.0)?;

        assert!(cold[1] > hot[1]);

        Ok(())
    }

    #[test]
    fn test_invalid_sampling_parameters_are_rejected() {
        assert!(
            select_token(
                &[0.0, 1.0],
                SamplingStrategy::TopP {
                    p: 0.0,
                    temperature: 1.0,
                },
                0.5,
            )
            .is_err()
        );
        assert!(
            select_token(
                &[0.0, 1.0],
                SamplingStrategy::TopP {
                    p: 1.1,
                    temperature: 1.0,
                },
                0.5,
            )
            .is_err()
        );
        assert!(
            select_token(
                &[0.0, 1.0],
                SamplingStrategy::TopP {
                    p: 0.9,
                    temperature: 0.0,
                },
                0.5,
            )
            .is_err()
        );
        assert!(
            select_token(
                &[0.0, 1.0],
                SamplingStrategy::TopP {
                    p: 0.9,
                    temperature: 1.0,
                },
                1.0,
            )
            .is_err()
        );
        assert!(select_token(&[], SamplingStrategy::Greedy, 0.0).is_err());
    }
}
