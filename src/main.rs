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

    let encoder_tokens: Vec<usize> = (3..3 + seq_len).collect();
    // Decoder input is shifted right so each position predicts the next token.
    let decoder_tokens: Vec<usize> = std::iter::once(SOS_TOKEN)
        .chain(encoder_tokens.iter().copied().take(seq_len - 1))
        .collect();

    let mut graph = autograd::Graph::new();
    // The encoder and decoder share one embedding table, as in the paper.
    let embedding = transformer::TokenEmbedding::new(&mut graph, &device, VOCAB_SIZE, d_model)?;
    let input_embeddings = embedding.forward(&mut graph, &device, &encoder_tokens)?;
    let decoder_input = embedding.forward(&mut graph, &device, &decoder_tokens)?;

    let encoder = transformer::Encoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
    let encoder_output = encoder.forward(&mut graph, &device, input_embeddings)?;

    let decoder = transformer::Decoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
    let output = decoder.forward(&mut graph, &device, decoder_input, encoder_output)?;

    graph.reset_gradients()?;
    graph.backward(output)?;

    println!(
        "Encoder input embedding shape: {:?}",
        graph.nodes[input_embeddings].data.shape()
    );
    println!(
        "Decoder output: {:?}",
        graph.nodes[output].data.to_vec::<f32>()
    );
    println!(
        "Gradient of encoder input embeddings: {:?}",
        graph.nodes[input_embeddings].grad.to_vec::<f32>()
    );

    println!("All gradients computed successfully.");

    Ok(())
}
