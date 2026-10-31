#ifndef TENSOR_H
#define TENSOR_H
#include <stdint.h>
typedef enum {
    T_LEAF,
    T_ADD,
    T_MUL,
    T_RELU,
    T_TANH,
    T_SUB,
    T_DIV,
    T_MATMUL,
    T_ADD_BIAS,
    T_MEAN,
    T_SOFTMAX_CE
} TOp;
typedef struct shape { int rows; int cols;} Shape;
struct tensor {
float *data;
float *grad;
Shape s;
struct tensor **inputs;
int count;
TOp oper;
int visited;
int size;
};
typedef struct tensor Tensor;
Tensor *tensor_create(const float *data, int rows, int cols);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_relu(Tensor *a);
Tensor *tensor_tanh(Tensor *a);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_add_bias(Tensor *x, Tensor *b);
Tensor *tensor_mean(Tensor *x);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_rand(int rows, int cols, float scale);
Tensor *tensor_softmax_ce(Tensor *logits, Tensor *labels);
Tensor *tensor_rows(const Tensor *src, int start, int count);
void tensor_build(Tensor *n, Tensor **list, int *len);
void tensor_backward(Tensor *loss);
void tensor_backward_step(Tensor *n);
void tensor_free(Tensor *t);
void tensor_free_graph(Tensor *loss);
void rng_seed(uint32_t s);
float rng_uniform(void);
float tensor_accuracy(const Tensor *logits, const Tensor *labels);   
#endif