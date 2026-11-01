//! A small reverse-mode automatic differentiation engine for computational graphs.
//!
//! [`Graph`] stores intermediate tensors and the [`Operation`] that produced
//! them. [`Graph::backward`] walks those nodes in reverse order and accumulates
//! gradients in [`Node::grad`].
use numr::prelude::*;
use numr::runtime::cpu::CpuDevice;

/// Reference to a node in a [`Graph`].
///
/// Node IDs are returned by builder methods and should not be constructed manually.
pub type NodeId = usize;

/// Operation produced by a graph node.
///
/// Each variant stores the input node IDs needed to apply its derivative
/// during reverse-mode differentiation.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Element-wise addition: `x_i = a_i + b_i`.
    Sum(NodeId, NodeId),
    /// Element-wise multiplication: `x_i = a_i * b_i`.
    Mul(NodeId, NodeId),
    /// Matrix multiplication: `x_ij = a_ik * b_kj`.
    MatMul(NodeId, NodeId),
    /// Causal masking: `x_ij = a_ij` if `j <= i`, else `-inf`.
    CausalMask(NodeId),
    /// Softmax over the final tensor dimension.
    SoftMax(NodeId),
    /// ReLU activation: `x_i = max(0, x_i)`.
    Relu(NodeId),
    /// Two-dimensional transpose: `x_ij = a_ji`.
    Transpose(NodeId),
}

/// A single computational step in a graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Tensor produced by the forward pass.
    pub data: Tensor<CpuRuntime>,
    /// Gradient accumulated during the backward pass.
    pub grad: Tensor<CpuRuntime>,
    /// Operation that produced `data`, or `None` for an input variable.
    pub op: Option<Operation>,
}

/// Computational graph used for forward evaluation and reverse-mode autodiff.
pub struct Graph {
    /// Nodes in insertion order, addressable by [`NodeId`].
    pub nodes: Vec<Node>,
}

impl Graph {
    /// Creates an empty computational graph.
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    /// Adds a leaf tensor to the graph.
    ///
    /// The new node starts with a zero gradient and no producing operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the gradient tensor cannot be allocated.
    pub fn variable(&mut self, data: Tensor<CpuRuntime>) -> Result<NodeId> {
        let shape = data.shape().to_vec();
        let dtype = data.dtype();
        let device = data.device();
        let grad = Tensor::zeros(&shape, dtype, device)?;

        let node = Node {
            data,
            grad,
            op: None,
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Adds two graph nodes element-wise and records the addition operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor operation or gradient allocation fails.
    pub fn sum(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        let data = node_a.data.add(&node_b.data)?;
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::Sum(a, b)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Multiplies two graph nodes element-wise and records the operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor operation or gradient allocation fails.
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        let data = node_a.data.mul(&node_b.data)?;
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::Mul(a, b)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Multiplies two graph nodes as matrices and records the operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix dimensions are incompatible or a tensor
    /// allocation fails.
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let node_b = &self.nodes[b];

        let data = node_a.data.matmul(&node_b.data)?;
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::MatMul(a, b)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Applies a causal attention mask to a matrix.
    ///
    /// Values in the strict upper triangle are replaced with `-inf`; the
    /// diagonal and lower triangle are preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is not 2-D or a tensor operation fails.
    pub fn causal_mask(&mut self, a: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let upper_mask = upper_triangular_mask(node_a.data.shape(), node_a.data.device())?;
        let data = node_a.data.masked_fill(&upper_mask, f64::NEG_INFINITY)?;

        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::CausalMask(a)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Applies softmax along the final dimension of a node.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operation fails.
    pub fn softmax(&mut self, a: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let data = node_a.data.softmax(-1)?;

        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::SoftMax(a)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Applies the ReLU activation function to a node.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operation fails.
    pub fn relu(&mut self, a: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let data = node_a.data.relu()?;

        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::Relu(a)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Transposes the first two dimensions of a node.
    ///
    /// The result is made contiguous so it can be consumed reliably by later
    /// matrix operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor cannot be transposed or made contiguous.
    pub fn transpose(&mut self, a: NodeId) -> Result<NodeId> {
        let node_a = &self.nodes[a];
        let data = node_a.data.transpose(0, 1)?.contiguous()?;

        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::Transpose(a)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Computes gradients from an output node back through the graph.
    ///
    /// The output gradient is initialized to ones, and each node's gradient is
    /// accumulated in [`Node::grad`]. Call [`Self::reset_gradients`] before a
    /// new backward pass when gradients from an earlier pass should not be
    /// retained.
    ///
    /// # Errors
    ///
    /// Returns an error if any derivative operation or gradient accumulation
    /// fails.
    pub fn backward(&mut self, output: NodeId) -> Result<()> {
        let seed_grad = Tensor::ones(
            self.nodes[output].data.shape(),
            self.nodes[output].data.dtype(),
            self.nodes[output].data.device(),
        )?;
        self.nodes[output].grad = seed_grad;

        for i in (0..=output).rev() {
            let node_grad = self.nodes[i].grad.clone();

            if let Some(op) = self.nodes[i].op.clone() {
                match op {
                    Operation::Sum(a, b) => {
                        self.nodes[a].grad = self.nodes[a].grad.add(&node_grad)?;
                        self.nodes[b].grad = self.nodes[b].grad.add(&node_grad)?;
                    }
                    Operation::Mul(a, b) => {
                        let node_a = &self.nodes[a];
                        let node_b = &self.nodes[b];

                        let grad_a = node_grad.mul(&node_b.data)?;
                        let grad_b = node_a.data.mul(&node_grad)?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_a)?;
                        self.nodes[b].grad = self.nodes[b].grad.add(&grad_b)?;
                    }
                    Operation::MatMul(a, b) => {
                        let node_a = &self.nodes[a];
                        let node_b = &self.nodes[b];

                        let a_transpose = node_a.data.transpose(0, 1)?.contiguous()?;
                        let b_transpose = node_b.data.transpose(0, 1)?.contiguous()?;

                        let grad_a = node_grad.matmul(&b_transpose)?;
                        let grad_b = a_transpose.matmul(&node_grad)?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_a)?;
                        self.nodes[b].grad = self.nodes[b].grad.add(&grad_b)?;
                    }
                    Operation::CausalMask(a) => {
                        let node_a = &self.nodes[a];
                        let upper_mask =
                            upper_triangular_mask(node_a.data.shape(), node_a.data.device())?;
                        let masked_grad = node_grad.masked_fill(&upper_mask, 0.0)?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&masked_grad)?;
                    }
                    Operation::SoftMax(a) => {
                        let softmax_output = &self.nodes[i].data;
                        let last_dim = softmax_output.shape().len() - 1;

                        let sum_dy_y = softmax_output.mul(&node_grad)?.sum(&[last_dim], true)?;
                        let grad_input = softmax_output.mul(&node_grad.sub(&sum_dy_y)?)?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_input)?;
                    }
                    Operation::Relu(a) => {
                        let node_a = &self.nodes[a];
                        let zeros = Tensor::zeros(
                            node_a.data.shape(),
                            node_a.data.dtype(),
                            node_a.data.device(),
                        )?;

                        let relu_mask = node_a.data.gt(&zeros)?;
                        let grad_input = node_grad.mul(&relu_mask)?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_input)?;
                    }
                    Operation::Transpose(a) => {
                        let grad_a = node_grad.transpose(0, 1)?.contiguous()?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_a)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Clears every node's accumulated gradient.
    ///
    /// Call this before starting an independent backward pass to avoid
    /// accumulating gradients from earlier passes.
    ///
    /// # Errors
    ///
    /// Returns an error if a zero-filled gradient tensor cannot be allocated.
    pub fn reset_gradients(&mut self) -> Result<()> {
        for node in &mut self.nodes {
            node.grad = Tensor::zeros(node.data.shape(), node.data.dtype(), node.data.device())?;
        }
        Ok(())
    }
}

fn upper_triangular_mask(shape: &[usize], device: &CpuDevice) -> Result<Tensor<CpuRuntime>> {
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: "causal masking requires a 2-D tensor".to_string(),
        });
    }

    let mask: Vec<u8> = (0..shape[0])
        .flat_map(|row| (0..shape[1]).map(move |column| u8::from(column > row)))
        .collect();

    Tensor::from_slice(&mask, shape, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, Graph) {
        (CpuDevice::new(), Graph::new())
    }

    // =========================================================================
    // 1. SINGULAR OPERATION UNIT TESTS
    // =========================================================================

    #[test]
    fn test_op_sum() -> Result<()> {
        let (device, mut graph) = setup();

        let a = graph.variable(Tensor::from_slice(&[1.0_f32, 2.0], &[2], &device)?)?;
        let b = graph.variable(Tensor::from_slice(&[3.0_f32, 4.0], &[2], &device)?)?;
        let out = graph.sum(a, b)?;

        graph.backward(out)?;

        // Forward: [1, 2] + [3, 4] = [4, 6]
        assert_eq!(graph.nodes[out].data.to_vec::<f32>(), vec![4.0, 6.0]);
        // Backward: dL/dA = [1, 1], dL/dB = [1, 1]
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![1.0, 1.0]);
        assert_eq!(graph.nodes[b].grad.to_vec::<f32>(), vec![1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_op_mul() -> Result<()> {
        let (device, mut graph) = setup();

        let a = graph.variable(Tensor::from_slice(&[2.0_f32, 3.0], &[2], &device)?)?;
        let b = graph.variable(Tensor::from_slice(&[4.0_f32, 5.0], &[2], &device)?)?;
        let out = graph.mul(a, b)?;

        graph.backward(out)?;

        // Forward: [2 * 4, 3 * 5] = [8, 15]
        assert_eq!(graph.nodes[out].data.to_vec::<f32>(), vec![8.0, 15.0]);
        // Backward: dL/dA = B = [4, 5], dL/dB = A = [2, 3]
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![4.0, 5.0]);
        assert_eq!(graph.nodes[b].grad.to_vec::<f32>(), vec![2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_op_matmul() -> Result<()> {
        let (device, mut graph) = setup();

        // A (2x2): [[1, 2], [3, 4]]
        let a = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &device,
        )?)?;
        // B (2x2): [[0.5, 0.0], [0.0, 0.5]] (Scaled Identity)
        let b = graph.variable(Tensor::from_slice(
            &[0.5_f32, 0.0, 0.0, 0.5],
            &[2, 2],
            &device,
        )?)?;

        let out = graph.matmul(a, b)?;
        graph.backward(out)?;

        // Forward: A . B = [[0.5, 1.0], [1.5, 2.0]]
        assert_eq!(
            graph.nodes[out].data.to_vec::<f32>(),
            vec![0.5, 1.0, 1.5, 2.0]
        );
        // dL/dA = dL/dY . B^T = [[1, 1], [1, 1]] . [[0.5, 0.0], [0.0, 0.5]] = [[0.5, 0.5], [0.5, 0.5]]
        assert_eq!(
            graph.nodes[a].grad.to_vec::<f32>(),
            vec![0.5, 0.5, 0.5, 0.5]
        );
        // dL/dB = A^T . dL/dY = [[1, 3], [2, 4]] . [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
        assert_eq!(
            graph.nodes[b].grad.to_vec::<f32>(),
            vec![4.0, 4.0, 6.0, 6.0]
        );

        Ok(())
    }

    #[test]
    fn test_op_causal_mask() -> Result<()> {
        let (device, mut graph) = setup();

        // Input 2x2 matrix: [[1.0, 2.0], [3.0, 4.0]]
        let a = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &device,
        )?)?;
        let out = graph.causal_mask(a)?;

        graph.backward(out)?;

        let out_vec = graph.nodes[out].data.to_vec::<f32>();

        // Upper triangle element (0, 1) should be -inf
        assert_eq!(out_vec[0], 1.0);
        assert!(out_vec[1].is_infinite() && out_vec[1].is_sign_negative());
        assert_eq!(out_vec[2], 3.0);
        assert_eq!(out_vec[3], 4.0);

        // Backward: Gradient for masked position (0, 1) must be zeroed out
        assert_eq!(
            graph.nodes[a].grad.to_vec::<f32>(),
            vec![1.0, 0.0, 1.0, 1.0]
        );

        Ok(())
    }

    #[test]
    fn test_op_softmax() -> Result<()> {
        let (device, mut graph) = setup();

        // Input 1x2 row: [[0.0, 0.0]]
        let a = graph.variable(Tensor::from_slice(&[0.0_f32, 0.0], &[1, 2], &device)?)?;
        let out = graph.softmax(a)?;

        graph.backward(out)?;

        // Forward: Softmax([0, 0]) = [0.5, 0.5]
        assert_eq!(graph.nodes[out].data.to_vec::<f32>(), vec![0.5, 0.5]);

        // Analytical Gradient for Softmax with seed dL/dY = [1, 1]:
        // sum(dL/dY * Y) = 1*0.5 + 1*0.5 = 1.0
        // dL/dX_i = Y_i * (dL/dY_i - 1.0) = 0.5 * (1.0 - 1.0) = 0.0
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_op_relu() -> Result<()> {
        let (device, mut graph) = setup();

        // Input 1x4 row: [[-1.0, 0.0, 1.0, 2.0]]
        let a = graph.variable(Tensor::from_slice(
            &[-1.0_f32, 0.0, 1.0, 2.0],
            &[1, 4],
            &device,
        )?)?;
        let out = graph.relu(a)?;

        graph.backward(out)?;

        // Forward: ReLU([-1, 0, 1, 2]) = [0, 0, 1, 2]
        assert_eq!(
            graph.nodes[out].data.to_vec::<f32>(),
            vec![0.0, 0.0, 1.0, 2.0]
        );

        // Backward: Gradient should be zero for negative and zero inputs
        assert_eq!(
            graph.nodes[a].grad.to_vec::<f32>(),
            vec![0.0, 0.0, 1.0, 1.0]
        );

        Ok(())
    }

    #[test]
    fn test_op_transpose() -> Result<()> {
        let (device, mut graph) = setup();

        // Input 2x3 matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let a = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &device,
        )?)?;
        let out = graph.transpose(a)?;

        graph.backward(out)?;

        // Forward: Transpose([[1, 2, 3], [4, 5, 6]]) = [[1, 4], [2, 5], [3, 6]]
        assert_eq!(
            graph.nodes[out].data.to_vec::<f32>(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );

        // Backward: Gradient should be transposed back to original shape
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![1.0; 6]);

        Ok(())
    }

    // =========================================================================
    // 2. END-TO-END TRANSFORMER CAUSAL SELF-ATTENTION TEST
    // =========================================================================

    #[test]
    fn test_transformer_causal_self_attention() -> Result<()> {
        let (device, mut graph) = setup();

        // Sequence length L = 2, Key/Value dimension d_k = 2

        // Queries (Q) [2, 2]
        let q = graph.variable(Tensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 1.0],
            &[2, 2],
            &device,
        )?)?;

        // Keys Transposed (K_T) [2, 2]
        let k_t = graph.variable(Tensor::from_slice(
            &[1.0_f32, 0.0, 0.0, 1.0],
            &[2, 2],
            &device,
        )?)?;

        // Values (V) [2, 2]
        let v = graph.variable(Tensor::from_slice(
            &[10.0_f32, 20.0, 30.0, 40.0],
            &[2, 2],
            &device,
        )?)?;

        // --- TRANSFORMER PIPELINE ---
        // 1. Raw Scores: S = Q . K^T
        let raw_scores = graph.matmul(q, k_t)?;

        // 2. Causal Masking: Upper-triangle set to -inf
        let masked_scores = graph.causal_mask(raw_scores)?;

        // 3. Attention Weights: Softmax across sequence dimension
        let attn_weights = graph.softmax(masked_scores)?;

        // 4. Output Projection: O = Attention . V
        let output = graph.matmul(attn_weights, v)?;

        // --- BACKWARD PASS ---
        graph.backward(output)?;

        // --- VERIFICATIONS ---
        let attn_probs = graph.nodes[attn_weights].data.to_vec::<f32>();

        // Causal Property Check:
        // Row 0 can only attend to position 0 -> Attn[0] = [1.0, 0.0]
        assert!((attn_probs[0] - 1.0).abs() < 1e-5);
        assert!((attn_probs[1] - 0.0).abs() < 1e-5);

        // Attention probabilities per row must sum to 1.0
        let row1_sum = attn_probs[2] + attn_probs[3];
        assert!((row1_sum - 1.0).abs() < 1e-5);

        // Output Verification:
        // Row 0 output must equal V[0] = [10.0, 20.0] because it only attends to position 0
        let out_vec = graph.nodes[output].data.to_vec::<f32>();
        assert!((out_vec[0] - 10.0).abs() < 1e-5);
        assert!((out_vec[1] - 20.0).abs() < 1e-5);

        // Backward Pass Gradient Flow Check:
        // Ensure non-zero gradients flowed all the way back to Q, K_T, and V
        assert!(
            graph.nodes[q]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&g| g != 0.0)
        );
        assert!(
            graph.nodes[k_t]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&g| g != 0.0)
        );
        assert!(
            graph.nodes[v]
                .grad
                .to_vec::<f32>()
                .iter()
                .any(|&g| g != 0.0)
        );

        Ok(())
    }
}
