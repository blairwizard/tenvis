# Autograd in C

A small automatic-differentiation engine and tensor library written from scratch in C, with no dependencies beyond the standard library. It is used to train a neural network that recognizes handwritten digits on MNIST.

**Result:** a 784 → 128 → 10 network trained with this library reaches **95.8% test accuracy** on MNIST after 3 epochs (about 100 seconds per epoch on a single core).

| epoch | mean training loss | test accuracy |
|---|---|---|
| 1 | 0.363 | 93.1% |
| 2 | 0.192 | 94.9% |
| 3 | 0.146 | 95.8% |

## What's in it

**Scalar engine** (`autograd.c`, `autograd.h`). A computation graph of scalar nodes with reverse-mode automatic differentiation. Operations: add, subtract, multiply, divide, ReLU, tanh.

**Tensor engine** (`tensor.c`, `tensor.h`). The same idea on 2D tensors (row-major `float` arrays):

- elementwise add, subtract, multiply, ReLU, tanh
- matrix multiply, with forward and backward passes
- bias broadcasting (`add_bias`), mean
- softmax cross-entropy, computed with the max-subtraction trick so large logits don't overflow
- whole-graph `tensor_backward` and `tensor_free_graph`
- a seeded xorshift32 random number generator (`rng_seed`, `rng_uniform`, `tensor_rand`)
- batching (`tensor_rows`) and an accuracy metric

**MNIST loader** (`mnist.c`, `mnist.h`). Reads the idx file format directly, including the big-endian headers.

## How it works

Every operation creates a new node that remembers its inputs and which operation produced it. Calling `backward` on the loss:

1. walks the graph depth-first to build a **topological order**, so every node comes after the nodes it depends on
2. zeroes all gradients, then seeds the loss gradient with 1
3. visits the nodes in reverse order, and each one pushes `local derivative × incoming gradient` into its inputs

Gradients are **accumulated with `+=`**, which is what makes a value used in more than one place (like `a * a`, or a weight shared across a graph) get the sum of its contributions.

Memory ownership is simple: parameters and datasets are leaves you create and free yourself. `tensor_free_graph(loss)` frees every intermediate node (data, gradient, input list, and the node itself) and keeps the leaves, so a training loop can build and free a graph every step without leaking.

## Build and run

Requires a C compiler (tested with gcc). On Linux and macOS, add `-lm` as shown.

```
# train the MNIST network (see "Getting the data" below)
gcc -O2 tensor.c mnist.c mnist_train.c -o mnist_train -lm
./mnist_train

# small examples
gcc -O2 tensor.c xor.c -o xor -lm && ./xor      # a 2-4-1 network learns XOR
gcc -O2 tensor.c train.c -o linear_fit -lm && ./linear_fit   # fits y = 2x + 1
```

## Getting the data

The MNIST files are not included (about 55 MB unzipped). Download the four idx files from any MNIST mirror, unzip them, and put them in the project folder:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

If your files use hyphens instead of dots (for example `train-images-idx3-ubyte`), rename them or change the four paths at the top of `main` in `mnist_train.c`. The expected sizes are 47,040,016 / 60,008 / 7,840,016 / 10,008 bytes.

## Training setup

| | |
|---|---|
| Architecture | 784 → 128 (ReLU) → 10 |
| Loss | softmax cross-entropy |
| Optimizer | plain SGD, learning rate 0.1 |
| Batch size | 64 |
| Initialization | uniform in ±sqrt(6 / fan_in), biases zero |
| Seed | 42 |

## Tests

```
gcc -Wall -Wextra autograd.c test_autograd.c -o test_autograd -lm && ./test_autograd
gcc -Wall -Wextra tensor.c test_tensor_ops.c -o test_tensor_ops -lm && ./test_tensor_ops
```

Each operation's forward value and gradient is checked against hand-computed numbers, and the gradients were also checked against finite-difference estimates during development. The tests also cover shared inputs, shape mismatches, a batch of one, overflow in softmax, and a loop that builds and frees 100,000 graphs to confirm the leaves survive and nothing leaks. Both programs exit with code 0 when every check passes.

## Limitations

- tensors are 2D only
- `matmul` is a plain triple loop (correct, but not cache-optimized or multithreaded)
- graphs are limited to 1,024 nodes (`TMAX_NODES`)
- training data is not shuffled between epochs
- SGD only (no momentum or Adam)

## Files

| file | purpose |
|---|---|
| `autograd.c/.h` | scalar autograd engine |
| `tensor.c/.h` | tensor engine, RNG, batching, accuracy |
| `mnist.c/.h` | MNIST idx loader |
| `mnist_train.c` | trains the MNIST network |
| `xor.c`, `train.c` | small training examples |
| `test_autograd.c`, `test_tensor_ops.c` | automated tests |
| `mnist_test.c` | prints an MNIST image and label as ASCII art |
