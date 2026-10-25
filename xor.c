#include<stdio.h>
#include"tensor.h"

static void sgd_step(Tensor *p, float lr) {
    for (int i=0; i< p->size;i++) p->data[i]-=lr*p->grad[i];
}

int main(void) {
    rng_seed(42);
    float xv[8] = {0,0,0,1,1,0,1,1}, tv[4]={0,1,1,0};
    Tensor *x = tensor_create(xv,4,2), *target = tensor_create(tv,4,1);
    Tensor *W1 = tensor_rand(2,4,0.7f);
    Tensor *b1 = tensor_create(NULL,1,4);
    Tensor *W2 = tensor_rand(4,1,0.5f);
    Tensor *b2 = tensor_create(NULL,1,1);
    float lr=0.1f;
    
    for(int step = 0; step < 5000; step++) {
        Tensor *h = tensor_tanh(tensor_add_bias(tensor_matmul(x,W1),b1));
        Tensor *pred = tensor_add_bias(tensor_matmul(h,W2),b2);
        Tensor *diff = tensor_sub(pred,target);
        Tensor *loss = tensor_mean(tensor_mul(diff,diff));
        sgd_step(W1,lr);
        sgd_step(b1,lr);
        sgd_step(W2,lr);
        sgd_step(b2,lr);
        float loss_val = loss->data[0];
        float p0 = pred->data[0];
        tensor_backward(loss);
        /* four sgd_step calls */
        tensor_free_graph(loss);
        if (step % 500 == 0)
            printf("step %4d  loss %.5f  pred[0] %.4f\n", step, loss_val, p0);
    }
        Tensor *h = tensor_tanh(tensor_add_bias(tensor_matmul(x, W1), b1));
    Tensor *pred = tensor_add_bias(tensor_matmul(h, W2), b2);
    printf("predictions: %.3f %.3f %.3f %.3f  (want 0 1 1 0)\n",
           pred->data[0], pred->data[1], pred->data[2], pred->data[3]);
    tensor_free_graph(pred);
    tensor_free(x); tensor_free(target);
    tensor_free(W1); tensor_free(b1); tensor_free(W2); tensor_free(b2);
    return 0;
}