mod autograd;

use numr::runtime::cpu::CpuDevice;
use numr::tensor::Tensor;

fn main() {
    let device = CpuDevice::new();

    let mut graph = autograd::Graph::new();
    let var_1 = graph
        .variable(
            Tensor::from_slice(
                &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                &[3, 3],
                &device,
            )
            .unwrap(),
        )
        .expect("Failed to create variable");
    let var_2 = graph
        .variable(
            Tensor::from_slice(
                &[9.0_f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                &[3, 3],
                &device,
            )
            .unwrap(),
        )
        .expect("Failed to create variable");
    let node_3 = graph
        .matmul(var_1, var_2)
        .expect("Failed to perform matrix multiplication");
    let _node_4 = graph
        .causal_mask(node_3)
        .expect("Failed to perform causal mask operation");
    let _node_5 = graph
        .softmax(var_1)
        .expect("Failed to perform softmax operation");

    graph.backward(var_1).expect("Failed to compute gradients");

    println!("Values: {:#?}", graph.nodes[var_1].data);
    println!("Gradients: {:#?}", graph.nodes[var_1].grad);
}
