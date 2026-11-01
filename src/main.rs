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
    let transformer_block = transformer::TransformerBlock::new(&mut graph, &device, d_model, d_ff)?;
    let output =
        transformer_block.apply_forward_block(&mut graph, &device, input_embeddings, d_model)?;

    graph.backward(output)?;

    println!("Output: {:?}", graph.nodes[output].data.to_vec::<f32>());
    println!(
        "Gradient of input embeddings: {:?}",
        graph.nodes[input_embeddings].grad.to_vec::<f32>()
    );
    println!(
        "Gradient of w_q: {:?}",
        graph.nodes[transformer_block.w_q].grad.to_vec::<f32>()
    );
    println!(
        "Gradient of w_k: {:?}",
        graph.nodes[transformer_block.w_k].grad.to_vec::<f32>()
    );
    println!(
        "Gradient of w_v: {:?}",
        graph.nodes[transformer_block.w_v].grad.to_vec::<f32>()
    );

    println!(
        "Gradient of w_1: {:?}",
        graph.nodes[transformer_block.w_1].grad.to_vec::<f32>()
    );

    println!(
        "Gradient of w_2: {:?}",
        graph.nodes[transformer_block.w_2].grad.to_vec::<f32>()
    );

    println!("All gradients computed successfully.");

    Ok(())
}
