mod autograd;
mod optim;
mod training;
mod transformer;

use numr::runtime::cpu::CpuDevice;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let seq_len = 4;
    let d_model = 32;
    let d_ff = 64;
    let n_heads = 4;

    let mut graph = autograd::Graph::new();
    let model = transformer::Transformer::new(
        &mut graph,
        &device,
        transformer::TransformerConfig {
            vocab_size: training::VOCAB_SIZE,
            d_model,
            d_ff,
            n_heads,
            max_len: seq_len + 2,
            sos_token: training::SOS_TOKEN,
            eos_token: training::EOS_TOKEN,
        },
    )?;

    let config = training::TrainConfig {
        steps: 3_000,
        seq_len,
        eval_interval: 250,
        eval_examples: 8,
        optimizer: optim::AdamWConfig {
            learning_rate: 3e-3,
            weight_decay: 0.01,
            ..optim::AdamWConfig::default()
        },
    };

    println!(
        "\nTraining with AdamW: {} steps, lr={}, weight decay={}",
        config.steps, config.optimizer.learning_rate, config.optimizer.weight_decay
    );

    let mut rng = StdRng::seed_from_u64(7);
    let history = training::train_reversal(&mut graph, &device, &model, &config, &mut rng)?;

    println!("\nTraining loss (mean over each reporting window):");
    for step in 1..=history.losses.len() {
        if step.is_multiple_of(config.eval_interval) || step == history.losses.len() {
            let start = step.saturating_sub(config.eval_interval);
            let window = &history.losses[start..step];
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            println!("  step {step:>4}: {mean:.4}");
        }
    }

    println!("\nEvaluation on held-out examples:");
    for evaluation in &history.evaluations {
        println!(
            "  step {:>4}: loss {:.4}, exact match {:>5.1}%",
            evaluation.step,
            evaluation.loss,
            evaluation.exact_match * 100.0
        );
    }

    println!(
        "\nSample greedy predictions (digit token = id - {}):",
        training::FIRST_DIGIT_TOKEN
    );
    let decoding = transformer::DecodingConfig {
        max_new_tokens: seq_len + 1,
        strategy: transformer::SamplingStrategy::Greedy,
    };
    let mut correct = 0;
    let sample_count = 5;
    for _ in 0..sample_count {
        let example = training::sample_reversal(&mut rng, seq_len)?;
        let generated = model.generate(&mut graph, &device, &example.input, &decoding, &mut rng)?;
        let target = example.target_sequence();
        let ok = generated == target;
        correct += usize::from(ok);
        println!(
            "  {:>15} -> {:>15}  target {:>15}  {}",
            format_tokens(&example.input),
            format_tokens(&generated),
            format_tokens(&target),
            if ok { "OK" } else { "WRONG" }
        );
    }

    let top_p_example = training::sample_reversal(&mut rng, seq_len)?;
    let top_p = model.generate(
        &mut graph,
        &device,
        &top_p_example.input,
        &transformer::DecodingConfig {
            max_new_tokens: seq_len + 1,
            strategy: transformer::SamplingStrategy::TopP {
                p: 0.9,
                temperature: 0.8,
            },
        },
        &mut rng,
    )?;
    println!(
        "\nTop-p sample (p=0.9, temperature=0.8): {:>15} -> {:>15}  target {:>15}",
        format_tokens(&top_p_example.input),
        format_tokens(&top_p),
        format_tokens(&top_p_example.target_sequence())
    );

    if let (Some(loss), Some(evaluation)) = (history.final_loss(), history.final_evaluation()) {
        println!(
            "\nFinal: training loss {loss:.4}, evaluation loss {:.4}, exact match {:.1}%",
            evaluation.loss,
            evaluation.exact_match * 100.0
        );
    }
    println!("Sample predictions correct: {correct}/{sample_count}");

    Ok(())
}

fn format_tokens(tokens: &[usize]) -> String {
    tokens
        .iter()
        .map(|&token| {
            if token >= training::FIRST_DIGIT_TOKEN {
                (token - training::FIRST_DIGIT_TOKEN).to_string()
            } else {
                format!("<{token}>")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
