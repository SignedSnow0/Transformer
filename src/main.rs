mod autograd;
mod transformer;

use numr::runtime::cpu::CpuDevice;
use numr::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let seq_len = 10;
    let d_model = 16;
    let d_ff = 64;
    let n_heads = 4;

    let input_data: Vec<f32> = (0..(seq_len * d_model))
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let input_embeddings = Tensor::from_slice(&input_data, &[seq_len, d_model], &device)?;

    let mut graph = autograd::Graph::new();
    let input_embeddings = graph.variable(input_embeddings)?;
    let encoder = transformer::Encoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
    let encoder_output = encoder.forward(&mut graph, &device, input_embeddings)?;

    let decoder = transformer::Decoder::new(&mut graph, &device, d_model, d_ff, n_heads)?;
    let decoder_input = graph.variable(Tensor::from_slice(
        &input_data,
        &[seq_len, d_model],
        &device,
    )?)?;
    let output = decoder.forward(&mut graph, &device, decoder_input, encoder_output)?;

    graph.reset_gradients()?;
    graph.backward(output)?;

    println!(
        "Decoder output: {:?}",
        graph.nodes[output].data.to_vec::<f32>()
    );
    println!(
        "Gradient of input embeddings: {:?}",
        graph.nodes[input_embeddings].grad.to_vec::<f32>()
    );

    println!("All gradients computed successfully.");

    Ok(())
}
