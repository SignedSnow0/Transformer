mod autograd;
mod transformer;

use numr::runtime::cpu::CpuDevice;
use numr::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let seq_len = 10;
    let d_model = 16;
    let d_ff = 64;

    let input_embeddings = Tensor::from_slice(
        &vec![0.5_f32; seq_len * d_model],
        &[seq_len, d_model],
        &device,
    )?;

    let mut graph = autograd::Graph::new();
    let input_embeddings = graph.variable(input_embeddings)?;
    let encoder = transformer::Encoder::new(&mut graph, &device, d_model, d_ff)?;
    let output = encoder.forward(&mut graph, &device, input_embeddings, d_model)?;

    graph.backward(output)?;

    println!("Output: {:?}", graph.nodes[output].data.to_vec::<f32>());
    println!(
        "Gradient of input embeddings: {:?}",
        graph.nodes[input_embeddings].grad.to_vec::<f32>()
    );

    println!("All gradients computed successfully.");

    Ok(())
}
