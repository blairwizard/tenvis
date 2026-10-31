#ifndef MNIST_H
#define MNIST_H
#include "tensor.h"

Tensor *load_images(const char *path, int max_n);   
Tensor *load_labels(const char *path, int max_n);
Tensor *tensor_rows(const Tensor *src, int start, int count);


#endif