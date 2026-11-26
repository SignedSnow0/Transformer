//! A small reverse-mode automatic differentiation engine for computational graphs.
//!
//! [`Graph`] stores intermediate tensors and the [`Operation`] that produced
//! them. [`Graph::backward_with_grad`] walks those nodes in reverse order and
//! accumulates gradients in [`Node::grad`].
use numr::ops::ScatterReduceOp;
use numr::prelude::*;
use numr::runtime::cpu::CpuRuntime;
use rand::RngExt;

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
    #[allow(dead_code)]
    Relu(NodeId),
    /// GELU activation using the tanh approximation.
    Gelu(NodeId),
    /// Inverted dropout with a stored, scaled mask node.
    Dropout(NodeId, NodeId),
    /// Embedding table lookup with a two-dimensional index matrix used by the
    /// backward scatter.
    EmbeddingLookup(NodeId, NodeId),
    /// Two-dimensional transpose: `x_ij = a_ji`.
    Transpose(NodeId),
    /// Layer normalization: `x_i = gamma * (a_i - mean(a)) / sqrt(var(a) + eps) + beta`.
    LayerNorm(NodeId, NodeId, NodeId, f32),
    /// Concatenation of multiple inputs along a dimension.
    Concat(Vec<NodeId>, usize),
}

/// A single computational step in a graph.
#[derive(Debug, Clone)]
pub struct Node<R: Runtime = CpuRuntime> {
    /// Tensor produced by the forward pass.
    pub data: Tensor<R>,
    /// Gradient accumulated during the backward pass.
    pub grad: Tensor<R>,
    /// Operation that produced `data`, or `None` for an input variable.
    pub op: Option<Operation>,
}

/// Computational graph used for forward evaluation and reverse-mode autodiff.
///
/// The runtime defaults to the CPU so unit tests and small examples can run
/// without a GPU; applications choose another backend by spelling the type,
/// for example `Graph<WgpuRuntime>`.
pub struct Graph<R: Runtime = CpuRuntime> {
    /// Nodes in insertion order, addressable by [`NodeId`].
    pub nodes: Vec<Node<R>>,
}

impl<R: Runtime> Default for Graph<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Runtime> Graph<R> {
    /// Creates an empty computational graph.
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }
}

impl<R> Graph<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    /// Adds a leaf tensor to the graph.
    ///
    /// The new node starts with a zero gradient and no producing operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the gradient tensor cannot be allocated.
    pub fn variable(&mut self, data: Tensor<R>) -> Result<NodeId> {
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
        let offset = triangular_mask(
            node_a.data.shape(),
            0.0,
            f32::NEG_INFINITY,
            node_a.data.device(),
        )?;
        let data = node_a.data.add(&offset)?;

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
    #[allow(dead_code)]
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

    /// Applies the tanh approximation of GELU to a node.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operation fails.
    pub fn gelu(&mut self, a: NodeId) -> Result<NodeId> {
        let x = &self.nodes[a].data;
        let inner = x
            .add(&x.mul(x)?.mul(x)?.mul_scalar(0.044715)?)?
            .mul_scalar(0.797_884_6)?;
        let data = x.mul(&inner.tanh()?.add_scalar(1.0)?)?.mul_scalar(0.5)?;
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        self.nodes.push(Node {
            data,
            grad,
            op: Some(Operation::Gelu(a)),
        });
        Ok(self.nodes.len() - 1)
    }

    /// Applies inverted dropout to a node.
    ///
    /// The mask is sampled from `rng` and stored in the graph so the backward
    /// pass uses exactly the same mask. A probability of zero returns `a`
    /// unchanged, which is also how evaluation passes bypass dropout.
    ///
    /// # Errors
    ///
    /// Returns an error if `probability` is outside `[0, 1)` or a tensor
    /// operation fails.
    pub fn dropout(
        &mut self,
        a: NodeId,
        probability: f32,
        rng: &mut impl RngExt,
    ) -> Result<NodeId> {
        if !(0.0..1.0).contains(&probability) {
            return Err(Error::InvalidArgument {
                arg: "probability",
                reason: "dropout probability must be in [0, 1)".to_string(),
            });
        }
        if probability == 0.0 {
            return Ok(a);
        }

        let (dtype, device, mask, data) = {
            let node = &self.nodes[a];
            let scale = 1.0 / (1.0 - probability);
            let values: Vec<f32> = (0..node.data.numel())
                .map(|_| {
                    if rng.random::<f32>() < probability {
                        0.0
                    } else {
                        scale
                    }
                })
                .collect();
            let mask = Tensor::from_slice(&values, node.data.shape(), node.data.device())?;
            let data = node.data.mul(&mask)?;
            (node.data.dtype(), node.data.device().clone(), mask, data)
        };

        let mask_node = self.variable(mask)?;
        let grad = Tensor::zeros(data.shape(), dtype, &device)?;

        self.nodes.push(Node {
            data,
            grad,
            op: Some(Operation::Dropout(a, mask_node)),
        });
        Ok(self.nodes.len() - 1)
    }

    /// Looks up rows of a trainable embedding table.
    ///
    /// `indices_1d` contains one token ID per row. `indices_2d` repeats those
    /// IDs across the embedding dimension so the gradient can be scattered
    /// back into the table without a host round trip.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors have incompatible shapes or the backend
    /// lookup fails.
    pub fn embedding_lookup(
        &mut self,
        table: NodeId,
        indices_1d: NodeId,
        indices_2d: NodeId,
    ) -> Result<NodeId> {
        let device = self.nodes[table].data.device().clone();
        let data = R::default_client(&device)
            .embedding_lookup(&self.nodes[table].data, &self.nodes[indices_1d].data)?;
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        self.nodes.push(Node {
            data,
            grad,
            op: Some(Operation::EmbeddingLookup(table, indices_2d)),
        });
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

    /// Applies layer normalization to a node.
    ///     
    /// # Arguments
    ///
    /// * `a` - The node to normalize.
    /// * `gamma` - The scale parameter.
    /// * `beta` - The shift parameter.
    /// * `eps` - A small value added to the variance for numerical stability.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer normalization operation fails.
    pub fn layer_norm(
        &mut self,
        a: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
    ) -> Result<NodeId> {
        let x_data = &self.nodes[a].data;
        let g_data = &self.nodes[gamma].data;
        let b_data = &self.nodes[beta].data;

        let last_dim = x_data.shape().len() - 1;

        let mean = x_data.mean(&[last_dim], true)?;

        let x_centered = x_data.sub(&mean)?;
        let x_sq = x_centered.mul(&x_centered)?;
        let var = x_sq.mean(&[last_dim], true)?;

        let std_inv = var.add_scalar(eps as f64)?.sqrt()?.recip()?;
        let x_hat = x_centered.mul(&std_inv)?;

        let data = x_hat.mul(g_data)?.add(b_data)?;
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::LayerNorm(a, gamma, beta, eps)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Concatenates nodes along a dimension.
    ///
    /// All inputs must have the same shape except along `dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if `inputs` is empty or the tensors cannot be
    /// concatenated.
    pub fn concat(&mut self, inputs: &[NodeId], dim: usize) -> Result<NodeId> {
        let data = {
            let tensors: Vec<&Tensor<R>> = inputs
                .iter()
                .map(|&input| &self.nodes[input].data)
                .collect();
            Tensor::cat(&tensors, dim as isize)?
        };
        let grad = Tensor::zeros(data.shape(), data.dtype(), data.device())?;

        let node = Node {
            data,
            grad,
            op: Some(Operation::Concat(inputs.to_vec(), dim)),
        };

        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    /// Computes gradients from an output node using a caller-supplied upstream
    /// gradient.
    ///
    /// Use this when the loss is computed outside the graph, for example a
    /// softmax cross-entropy loss whose gradient with respect to the logits is
    /// `(softmax(logits) - one_hot(target)) / num_tokens`.
    ///
    /// The supplied seed becomes the output node's gradient, and gradients
    /// flowing backward from it are accumulated in upstream nodes. Use a fresh
    /// graph for each independent backward pass.
    ///
    /// # Errors
    ///
    /// Returns an error if `seed_grad` does not have the same shape and dtype
    /// as the output node, or if a derivative operation fails.
    pub fn backward_with_grad(&mut self, output: NodeId, seed_grad: Tensor<R>) -> Result<()> {
        self.nodes[output].grad = seed_grad;

        for i in (0..=output).rev() {
            let node_grad = self.nodes[i].grad.clone();

            if let Some(op) = self.nodes[i].op.clone() {
                match op {
                    Operation::Sum(a, b) => {
                        let grad_a = reduce_broadcast(&node_grad, self.nodes[a].data.shape())?;
                        let grad_b = reduce_broadcast(&node_grad, self.nodes[b].data.shape())?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_a)?;
                        self.nodes[b].grad = self.nodes[b].grad.add(&grad_b)?;
                    }
                    Operation::Mul(a, b) => {
                        let a_shape = self.nodes[a].data.shape().to_vec();
                        let b_shape = self.nodes[b].data.shape().to_vec();

                        let grad_a =
                            reduce_broadcast(&node_grad.mul(&self.nodes[b].data)?, &a_shape)?;
                        let grad_b =
                            reduce_broadcast(&self.nodes[a].data.mul(&node_grad)?, &b_shape)?;

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
                        let keep =
                            triangular_mask(node_a.data.shape(), 1.0, 0.0, node_a.data.device())?;
                        let masked_grad = node_grad.mul(&keep)?;

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
                    Operation::Gelu(a) => {
                        let x = &self.nodes[a].data;
                        let inner = x
                            .add(&x.mul(x)?.mul(x)?.mul_scalar(0.044715)?)?
                            .mul_scalar(0.797_884_6)?;
                        let tanh = inner.tanh()?;
                        let one_plus_tanh = tanh.add_scalar(1.0)?;
                        let one_minus_tanh_sq =
                            tanh.mul(&tanh)?.mul_scalar(-1.0)?.add_scalar(1.0)?;
                        let d_inner_dx = x
                            .mul(x)?
                            .mul_scalar(3.0 * 0.044715)?
                            .add_scalar(1.0)?
                            .mul_scalar(0.797_884_6)?;
                        let derivative = one_plus_tanh.mul_scalar(0.5)?.add(
                            &x.mul(&one_minus_tanh_sq)?
                                .mul(&d_inner_dx)?
                                .mul_scalar(0.5)?,
                        )?;
                        let grad_input = node_grad.mul(&derivative)?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_input)?;
                    }
                    Operation::Dropout(a, mask) => {
                        let grad_input = node_grad.mul(&self.nodes[mask].data)?;
                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_input)?;
                    }
                    Operation::EmbeddingLookup(table, indices_2d) => {
                        let table_data = &self.nodes[table].data;
                        let zeros = Tensor::zeros(
                            table_data.shape(),
                            table_data.dtype(),
                            table_data.device(),
                        )?;
                        let device = table_data.device().clone();
                        let grad_table = R::default_client(&device).scatter_reduce(
                            &zeros,
                            0,
                            &self.nodes[indices_2d].data,
                            &node_grad,
                            ScatterReduceOp::Sum,
                            false,
                        )?;

                        self.nodes[table].grad = self.nodes[table].grad.add(&grad_table)?;
                    }
                    Operation::Transpose(a) => {
                        let grad_a = node_grad.transpose(0, 1)?.contiguous()?;

                        self.nodes[a].grad = self.nodes[a].grad.add(&grad_a)?;
                    }
                    Operation::LayerNorm(x, gamma, beta, eps) => {
                        let x_data = &self.nodes[x].data;
                        let g_data = &self.nodes[gamma].data;
                        let last_dim = x_data.shape().len() - 1;
                        let d = x_data.shape()[last_dim] as f32;

                        let mean = x_data.mean(&[last_dim], true)?;
                        let x_centered = x_data.sub(&mean)?;
                        let var = x_centered.mul(&x_centered)?.mean(&[last_dim], true)?;
                        let std_inv = var.add_scalar(eps as f64)?.sqrt()?.recip()?;
                        let x_hat = x_centered.mul(&std_inv)?;

                        let d_gamma = reduce_broadcast(&node_grad.mul(&x_hat)?, g_data.shape())?;
                        let d_beta = reduce_broadcast(&node_grad, self.nodes[beta].data.shape())?;

                        let d_xhat = node_grad.mul(g_data)?;

                        let sum_d_xhat = d_xhat.sum(&[last_dim], true)?;
                        let sum_d_xhat_xhat = d_xhat.mul(&x_hat)?.sum(&[last_dim], true)?;

                        let term1 = d_xhat.mul_scalar(d as f64)?;
                        let term2 = term1.sub(&sum_d_xhat)?;
                        let term3 = x_hat.mul(&sum_d_xhat_xhat)?;
                        let d_x = term2
                            .sub(&term3)?
                            .mul(&std_inv.mul_scalar(1.0 / d as f64)?)?;

                        self.nodes[x].grad = self.nodes[x].grad.add(&d_x)?;
                        self.nodes[gamma].grad = self.nodes[gamma].grad.add(&d_gamma)?;
                        self.nodes[beta].grad = self.nodes[beta].grad.add(&d_beta)?;
                    }
                    Operation::Concat(inputs, dim) => {
                        let mut offset = 0;
                        for input in inputs {
                            let len = self.nodes[input].data.shape()[dim];
                            let grad_slice =
                                node_grad.narrow(dim as isize, offset, len)?.contiguous()?;
                            self.nodes[input].grad = self.nodes[input].grad.add(&grad_slice)?;
                            offset += len;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Drops every node at or beyond `len`, keeping `nodes[..len]`.
    ///
    /// This is intended for training loops that rebuild the forward graph on
    /// every step. Parameters are typically registered first, so truncating to
    /// the parameter count frees the intermediates from the previous step while
    /// preserving parameter values and their accumulated gradients.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is greater than the current node count, or if
    /// a retained node refers to a node that would be removed.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        self.nodes.truncate(len);
        Ok(())
    }
}

/// Reduces a broadcasted gradient back to the shape of the operand it belongs
/// to by summing over the dimensions that were broadcast during the forward
/// pass.
fn reduce_broadcast<R>(grad: &Tensor<R>, target_shape: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R> + IndexingOps<R>,
{
    if grad.shape() == target_shape {
        return Ok(grad.clone());
    }

    let mut reduced = grad.clone();

    // Drop extra leading dimensions, e.g. [T, D] -> [D].
    while reduced.shape().len() > target_shape.len() {
        reduced = reduced.sum(&[0], false)?;
    }

    // Sum dimensions that were broadcast from size 1, e.g. [T, D] -> [1, D].
    let ndim = reduced.shape().len();
    for (index, &target_dim) in target_shape.iter().enumerate() {
        let axis = ndim - target_shape.len() + index;
        if target_dim == 1 && reduced.shape()[axis] != 1 {
            reduced = reduced.sum(&[axis], true)?;
        }
    }

    Ok(reduced)
}

/// Builds a 2-D mask with `lower` on the diagonal/lower triangle and `upper`
/// in the strict upper triangle.
///
/// The mask is materialized as an `f32` tensor so the same code works on every
/// backend: CPU `masked_fill` expects `u8` masks while the WebGPU implementation
/// only accepts `u32`, but ordinary arithmetic on `f32` is portable.
fn triangular_mask<R>(
    shape: &[usize],
    lower: f32,
    upper: f32,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    let mask: Vec<f32> = (0..shape[0])
        .flat_map(|row| (0..shape[1]).map(move |column| if column > row { upper } else { lower }))
        .collect();

    Tensor::from_slice(&mask, shape, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn setup() -> (CpuDevice, Graph) {
        (
            CpuDevice::new(),
            Graph::<numr::runtime::cpu::CpuRuntime>::new(),
        )
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

        graph.backward_with_grad(out, Tensor::ones(&[2], DType::F32, &device)?)?;

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

        graph.backward_with_grad(out, Tensor::ones(&[2], DType::F32, &device)?)?;

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
        graph.backward_with_grad(out, Tensor::ones(&[2, 2], DType::F32, &device)?)?;

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

        graph.backward_with_grad(out, Tensor::ones(&[2, 2], DType::F32, &device)?)?;

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

        graph.backward_with_grad(out, Tensor::ones(&[1, 2], DType::F32, &device)?)?;

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

        graph.backward_with_grad(out, Tensor::ones(&[1, 4], DType::F32, &device)?)?;

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

        graph.backward_with_grad(out, Tensor::ones(&[3, 2], DType::F32, &device)?)?;

        // Forward: Transpose([[1, 2, 3], [4, 5, 6]]) = [[1, 4], [2, 5], [3, 6]]
        assert_eq!(
            graph.nodes[out].data.to_vec::<f32>(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );

        // Backward: Gradient should be transposed back to original shape
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![1.0; 6]);

        Ok(())
    }

    #[test]
    fn test_op_concat() -> Result<()> {
        let (device, mut graph) = setup();

        let a = graph.variable(Tensor::from_slice(&[1.0_f32, 2.0], &[2, 1], &device)?)?;
        let b = graph.variable(Tensor::from_slice(&[3.0_f32, 4.0], &[2, 1], &device)?)?;
        let out = graph.concat(&[a, b], 1)?;

        graph.backward_with_grad(out, Tensor::ones(&[2, 2], DType::F32, &device)?)?;

        // Forward: [[1], [2]] and [[3], [4]] concatenated -> [[1, 3], [2, 4]]
        assert_eq!(
            graph.nodes[out].data.to_vec::<f32>(),
            vec![1.0, 3.0, 2.0, 4.0]
        );
        // Backward: each input receives the gradient for its own columns.
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![1.0, 1.0]);
        assert_eq!(graph.nodes[b].grad.to_vec::<f32>(), vec![1.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_add_bias_grad_shape() -> Result<()> {
        let (device, mut graph) = setup();

        // x is [3, 2], bias is [1, 2]. The forward add broadcasts the bias,
        // so its gradient must be reduced back to [1, 2].
        let x = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &device,
        )?)?;
        let bias = graph.variable(Tensor::from_slice(&[0.0_f32, 0.0], &[1, 2], &device)?)?;
        let out = graph.sum(x, bias)?;

        graph.backward_with_grad(out, Tensor::ones(&[3, 2], DType::F32, &device)?)?;

        assert_eq!(graph.nodes[bias].grad.shape(), &[1, 2]);
        assert_eq!(graph.nodes[bias].grad.to_vec::<f32>(), vec![3.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_backward_with_grad_scales_seed() -> Result<()> {
        let (device, mut graph) = setup();

        let a = graph.variable(Tensor::from_slice(&[1.0_f32, 2.0], &[2], &device)?)?;
        let scale = graph.variable(Tensor::from_slice(&[3.0_f32, 4.0], &[2], &device)?)?;
        let out = graph.mul(a, scale)?;

        let seed = Tensor::from_slice(&[2.0_f32, -1.0], &[2], &device)?;
        graph.backward_with_grad(out, seed)?;

        // dL/da = upstream * scale, dL/dscale = upstream * a.
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![6.0, -4.0]);
        assert_eq!(graph.nodes[scale].grad.to_vec::<f32>(), vec![2.0, -2.0]);

        Ok(())
    }

    #[test]
    fn test_truncate_drops_intermediates_and_keeps_parameters() -> Result<()> {
        let (device, mut graph) = setup();

        let a = graph.variable(Tensor::from_slice(&[1.0_f32, 2.0], &[2], &device)?)?;
        let parameter_count = graph.nodes.len();

        let b = graph.variable(Tensor::from_slice(&[3.0_f32, 4.0], &[2], &device)?)?;
        let out = graph.mul(a, b)?;
        graph.backward_with_grad(out, Tensor::ones(&[2], DType::F32, &device)?)?;

        assert!(graph.nodes.len() > parameter_count);
        graph.truncate(parameter_count)?;

        assert_eq!(graph.nodes.len(), parameter_count);
        assert_eq!(graph.nodes[a].data.to_vec::<f32>(), vec![1.0, 2.0]);
        assert_eq!(graph.nodes[a].grad.to_vec::<f32>(), vec![3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_broadcast_scalar_mul_grad_shape() -> Result<()> {
        let (device, mut graph) = setup();

        let a = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &device,
        )?)?;
        let scale = graph.variable(Tensor::from_slice(&[2.0_f32], &[1], &device)?)?;
        let out = graph.mul(a, scale)?;

        graph.backward_with_grad(out, Tensor::ones(&[2, 2], DType::F32, &device)?)?;

        assert_eq!(graph.nodes[scale].grad.shape(), &[1]);
        assert_eq!(graph.nodes[scale].grad.to_vec::<f32>(), vec![10.0]);

        Ok(())
    }

    #[test]
    fn test_op_gelu_forward_and_gradient() -> Result<()> {
        let (device, mut graph) = setup();
        let x = graph.variable(Tensor::from_slice(&[-1.0_f32, 0.0, 1.0], &[3], &device)?)?;
        let out = graph.gelu(x)?;

        graph.backward_with_grad(out, Tensor::ones(&[3], DType::F32, &device)?)?;

        let actual = graph.nodes[out].data.to_vec::<f32>();
        let gradients = graph.nodes[x].grad.to_vec::<f32>();
        for (index, &value) in [-1.0_f32, 0.0, 1.0].iter().enumerate() {
            let inner = 0.797_884_6 * (value + 0.044715 * value.powi(3));
            let expected = 0.5 * value * (1.0 + inner.tanh());
            assert!((actual[index] - expected).abs() < 1e-5);

            let epsilon = 1e-3;
            let upper = 0.797_884_6 * (value + epsilon + 0.044715 * (value + epsilon).powi(3));
            let lower = 0.797_884_6 * (value - epsilon + 0.044715 * (value - epsilon).powi(3));
            let upper = 0.5 * (value + epsilon) * (1.0 + upper.tanh());
            let lower = 0.5 * (value - epsilon) * (1.0 + lower.tanh());
            let expected_gradient = (upper - lower) / (2.0 * epsilon);
            assert!(
                (gradients[index] - expected_gradient).abs() < 1e-4,
                "expected {expected_gradient}, got {}",
                gradients[index]
            );
        }

        Ok(())
    }

    #[test]
    fn test_op_dropout_uses_stored_mask() -> Result<()> {
        let (device, mut graph) = setup();
        let mut rng = StdRng::seed_from_u64(42);
        let x = graph.variable(Tensor::from_slice(&[1.0_f32; 4], &[4], &device)?)?;
        let out = graph.dropout(x, 0.5, &mut rng)?;

        graph.backward_with_grad(out, Tensor::ones(&[4], DType::F32, &device)?)?;

        let output = graph.nodes[out].data.to_vec::<f32>();
        assert!(output.iter().all(|&value| value == 0.0 || value == 2.0));
        assert_eq!(graph.nodes[x].grad.to_vec::<f32>(), output);
        assert!(graph.dropout(x, 1.0, &mut rng).is_err());
        Ok(())
    }

    #[test]
    fn test_embedding_lookup_gathers_and_scatters() -> Result<()> {
        let (device, mut graph) = setup();
        let table = graph.variable(Tensor::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &device,
        )?)?;
        let indices_1d = graph.variable(Tensor::from_slice(&[0_i64, 2, 0], &[3], &device)?)?;
        let indices_2d = graph.variable(Tensor::from_slice(
            &[0_i64, 0, 2, 2, 0, 0],
            &[3, 2],
            &device,
        )?)?;
        let out = graph.embedding_lookup(table, indices_1d, indices_2d)?;

        graph.backward_with_grad(out, Tensor::ones(&[3, 2], DType::F32, &device)?)?;

        assert_eq!(
            graph.nodes[out].data.to_vec::<f32>(),
            vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0]
        );
        assert_eq!(
            graph.nodes[table].grad.to_vec::<f32>(),
            vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]
        );
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
        graph.backward_with_grad(output, Tensor::ones(&[2, 2], DType::F32, &device)?)?;

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
