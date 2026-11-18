mod autograd;
mod transformer;

use numr::runtime::cpu::CpuDevice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let seq_len = 10;
    let d_model = 16;
    let d_ff = 64;
    let n_heads = 4;

    // Toy vocabulary: 0 = <pad>, 1 = <sos>, 2 = <eos>, 3.. are content tokens.
    const VOCAB_SIZE: usize = 13;
    const SOS_TOKEN: usize = 1;
    const EOS_TOKEN: usize = 2;

    let encoder_tokens: Vec<usize> = (3..3 + seq_len).collect();
    // Decoder input is shifted right so each position predicts the next token.
    let decoder_tokens: Vec<usize> = std::iter::once(SOS_TOKEN)
        .chain(encoder_tokens.iter().copied().take(seq_len - 1))
        .collect();

    let mut graph = autograd::Graph::new();
    let model = transformer::Transformer::new(
        &mut graph,
        &device,
        transformer::TransformerConfig {
            vocab_size: VOCAB_SIZE,
            d_model,
            d_ff,
            n_heads,
            max_len: seq_len + 2,
            sos_token: SOS_TOKEN,
            eos_token: EOS_TOKEN,
        },
    )?;

    println!("Encoder tokens: {:?}", encoder_tokens);
    println!("Teacher-forced decoder input: {:?}", decoder_tokens);

    // Teacher-forced forward/backward sanity check.
    let encoder_output = model.encode(&mut graph, &device, &encoder_tokens)?;
    let logits = model.decode_logits(&mut graph, &device, &decoder_tokens, encoder_output)?;
    let loss = graph.mul(logits, logits)?;
    graph.reset_gradients()?;
    graph.backward(loss)?;

    println!("Logit shape: {:?}", graph.nodes[logits].data.shape());
    println!(
        "Embedding gradient shape: {:?}",
        graph.nodes[model.embedding.table].grad.shape()
    );
    println!(
        "Output weight gradient shape: {:?}",
        graph.nodes[model.output.w].grad.shape()
    );

    // Autoregressive decoding with both strategies.
    let mut rng = rand::rng();
    let greedy = model.generate(
        &mut graph,
        &device,
        &encoder_tokens,
        &transformer::DecodingConfig {
            max_new_tokens: seq_len,
            strategy: transformer::SamplingStrategy::Greedy,
        },
        &mut rng,
    )?;
    let top_p = model.generate(
        &mut graph,
        &device,
        &encoder_tokens,
        &transformer::DecodingConfig {
            max_new_tokens: seq_len,
            strategy: transformer::SamplingStrategy::TopP {
                p: 0.9,
                temperature: 0.8,
            },
        },
        &mut rng,
    )?;

    println!("Greedy tokens: {:?}", greedy);
    println!("Top-p tokens: {:?}", top_p);

    Ok(())
}
