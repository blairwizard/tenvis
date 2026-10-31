#include "tensor.h"
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#define TMAX_NODES 1024

Tensor *tensor_create(const float *data, int rows, int cols) {
    if(rows<=0 || cols<=0) {
        return NULL;
    }
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if(!t){return NULL;}
    t->size = rows*cols;
    t->data = (float *)calloc(sizeof(float),t->size);
    if(!t->data){free(t);return NULL;}
    t->grad = (float *)calloc(sizeof(float),t->size);
    if(!t->grad){free(t->data);free(t);return NULL;}
    if(data) {
        memcpy(t->data, data, sizeof(float)*t->size);
    }
    t->s.rows = rows;
    t->s.cols=cols;
    t->inputs = NULL;
    t->count = 0;
    t->oper = T_LEAF;
    t->visited = 0;
    return t;
}
Tensor *tensor_add(Tensor *a, Tensor *b) {
    if(!a || !b) return NULL;
    if(a->s.rows != b->s.rows || a->s.cols != b->s.cols) {
        fprintf(stderr, "Incompatible tensor dimensions\n");
        return NULL;
    }
    Tensor *t = tensor_create(NULL, a->s.rows, a->s.cols);
    if(!t) return NULL;
    for(int i=0;i<t->size;i++) {
        t->data[i] = a->data[i] + b->data[i];
    }
    t->inputs = (Tensor **)malloc(sizeof(Tensor *)*2);
    if(!t->inputs) {tensor_free(t);return NULL;}
    t->inputs[0] = a;
    t->inputs[1] = b;
    t->count = 2;
    t->oper = T_ADD;
    return t;
}
Tensor *tensor_mul(Tensor *a, Tensor *b) {
    if(!a || !b) return NULL;
    if(a->s.rows != b->s.rows || a->s.cols != b->s.cols) {
        fprintf(stderr, "Incompatible tensor dimensions\n");
        return NULL;
    }
    Tensor *t = tensor_create(NULL, a->s.rows, a->s.cols);
    if(!t) return NULL;
    for(int i = 0; i< t->size; i++) {
        t->data[i] = a->data[i] * b->data[i];
    }
    t->inputs = (Tensor **)malloc(sizeof(Tensor *)*2);
    if(!t->inputs){tensor_free(t); return NULL;}
    t->inputs[0]=a;
    t->inputs[1]=b;
    t->count = 2;
    t->oper = T_MUL;
    return t;
}
Tensor *tensor_relu(Tensor *a) {
    if(!a){return NULL;}
    Tensor *t = tensor_create(NULL, a->s.rows, a->s.cols);
    if(!t) return NULL;
    for(int i=0;i<t->size;i++) {
        t->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }
    t->inputs = (Tensor **)malloc(sizeof(Tensor *));
    if(!t->inputs){tensor_free(t); return NULL;}
    t->inputs[0] = a;
    t->count = 1;
    t->oper = T_RELU;
    return t;
}
Tensor *tensor_tanh(Tensor *a){
    if(!a){return NULL;}
    Tensor *t=tensor_create(NULL,a->s.rows,a->s.cols);
    if(!t){return NULL;}
    for(int i=0;i<t->size;i++){
        t->data[i]=tanhf(a->data[i]);
    }
    t->inputs=(Tensor **)malloc(sizeof(Tensor *));
    if(!t->inputs){tensor_free(t);return NULL;}
    t->inputs[0]=a;
    t->count=1;
    t->oper=T_TANH;
    return t;
}
Tensor *tensor_matmul(Tensor *a, Tensor *b){
    if(!a || !b) return NULL;
    if(a->s.cols!=b->s.rows){fprintf(stderr,"Shape mismatch\n");return NULL;}
    Tensor *c=tensor_create(NULL,a->s.rows,b->s.cols);
    if(!c) return NULL;
    for (int i=0;i<c->s.rows;i++) {
        for (int j=0;j<c->s.cols;j++) {
            float sum=0;
            for (int p=0;p<a->s.cols;p++){
                sum+=a->data[i*a->s.cols+p]*b->data[p*b->s.cols+j];
            }
            c->data[i*c->s.cols+j] = sum;
        }
    }
    c->inputs=(Tensor **)malloc(sizeof(Tensor *)*2);
    if(!c->inputs){tensor_free(c);return NULL;}
    c->inputs[0]=a;
    c->inputs[1]=b;
    c->count=2;
    c->oper=T_MATMUL;
    return c;
}
Tensor *tensor_add_bias(Tensor *x, Tensor *b) {
    if (!x || !b) return NULL;
    if(b->s.rows!=1 || b->s.cols!=x->s.cols){fprintf(stderr,"tensor_add_bias: shape mismatch (%dx%d vs %dx%d)\n",x->s.rows,x->s.cols,b->s.rows,b->s.cols);return NULL;}
    Tensor *t = tensor_create(NULL,x->s.rows,x->s.cols);
    if(!t) return NULL;
    for(int i=0; i<x->s.rows;i++)
        for(int j=0;j<x->s.cols;j++)
            t->data[i*x->s.cols+j]=x->data[i*x->s.cols+j]+b->data[j];
    t->inputs=(Tensor **)malloc(sizeof(Tensor *)*2);
    if(!t->inputs){tensor_free(t);return NULL;}
    t->inputs[0]=x;
    t->inputs[1]=b;
    t->count=2;
    t->oper=T_ADD_BIAS;
    return t;
}
Tensor *tensor_mean(Tensor *x){
    if(!x) return NULL;
    Tensor *t = tensor_create(NULL, 1, 1);
    if(!t) return NULL;
    float s=0;
    for(int k=0;k< x->size; k++) s+=x->data[k];
    t->data[0]=s/x->size;
    t->inputs=(Tensor **)malloc(sizeof(Tensor *));
    if(!t->inputs){tensor_free(t);return NULL;}
    t->inputs[0]=x;
    t->count=1;
    t->oper=T_MEAN;
    return t;
}
Tensor *tensor_sub(Tensor *a, Tensor *b){
    if(!a|| !b) return NULL;
    if(a->s.rows != b->s.rows || a->s.cols != b->s.cols) {
        fprintf(stderr, "tensor_sub: shape mismatch\n");
        return NULL;
    }
    Tensor *t = tensor_create(NULL,a->s.rows, a->s.cols);
    if (!t) return NULL;
    for (int k=0;k<t->size;k++) t->data[k]=a->data[k]-b->data[k];
    t->inputs = (Tensor **)malloc(sizeof(Tensor *)*2);
    if(!t->inputs){tensor_free(t); return NULL;}
    t->inputs[0]=a;
    t->inputs[1]=b;
    t->count=2;
    t->oper=T_SUB;
    return t;
}
static uint32_t rng_state = 12345;

void rng_seed(uint32_t s){
    rng_state = s?s:1;
}

float rng_uniform(void) {
    rng_state ^= rng_state <<13;
    rng_state ^= rng_state>>17;
    rng_state ^= rng_state<<5;
    float k = (rng_state>>8)/16777216.0f;
    return k;
}
Tensor *tensor_rand(int rows, int cols, float scale){
    Tensor *t =tensor_create(NULL, rows, cols);
    if(!t) return NULL;
    for (int k=0; k<t->size;k++){
        t->data[k]=rng_uniform()*2*scale-scale;
    }
    return t;
}

static void row_max_sum(const float *row, int cols, float *m, float *sum) {
    float mx=row[0];
    for(int j=1;j<cols;j++) if (row[j]>mx) mx = row[j];
    float s= 0;
    for(int j=0;j<cols;j++) s+=expf(row[j]-mx);
    *m=mx;
    *sum=s;
}

Tensor *tensor_softmax_ce(Tensor *logits, Tensor *labels) {
    if(!logits || !labels) return NULL;
    if (labels->s.cols!=1 || labels->s.rows!=logits->s.rows){
        fprintf(stderr,"tensor_softmax_ce: labels must be %dx1\n",logits->s.rows);
        return NULL;
    }
    int rows = logits->s.rows, cols = logits->s.cols;
    float total = 0;
    for(int i=0;i<rows; i++) {
        int c=(int)labels->data[i];
        if(c<0 || c>=cols){fprintf(stderr, "tensor_softmax_ce: label %d out of range [0, %d)\n", c, cols);return NULL;}
        float m, sum;
        row_max_sum(&logits->data[i*cols], cols, &m, &sum);
        total+=logf(sum)-(logits->data[i*cols+c]-m);
    }
    Tensor *t=tensor_create(NULL,1,1);
    if(!t) return NULL;
    t->data[0] = total/rows;
    t->inputs=(Tensor **)malloc(sizeof(Tensor *)*2);
    if(!t->inputs){tensor_free(t);return NULL;}
    t->inputs[0]=logits;
    t->inputs[1]=labels;
    t->count=2;
    t->oper=T_SOFTMAX_CE;
    return t;

}

Tensor *tensor_rows(const Tensor *src, int start, int count){
    if(!src){return NULL;}
    if(start < 0 || count<=0 || start+count > src->s.rows) {
        fprintf(stderr, "tensor_rows: range [%d, %d) invalid for %d rows\n", start, start + count, src->s.rows);
        return NULL;
    }
    Tensor *t = tensor_create(NULL,count,src->s.cols);
    if(!t) return NULL;
    memcpy(t->data, src->data + (size_t)start * src->s.cols, sizeof(float) * t->size);
    return t;
}
float tensor_accuracy(const Tensor *logits, const Tensor *labels){
    if(!logits || !labels) return 0;
    int rows = logits->s.rows, cols = logits->s.cols;
    if (labels->s.rows != rows || labels->s.cols != 1) {
        fprintf(stderr, "tensor_accuracy: labels must be %dx1\n", rows);
        return 0;
    }
    int correct = 0;
    for(int i=0;i<logits->s.rows;i++){
        int best=0;
        for(int j=1;j<cols;j++) {
            if(logits->data[i*cols+j]>logits->data[i*cols+best]) best=j;
        }
        if(best==(int)labels->data[i]) correct++;
    }
    return (float)correct/rows;
}

void tensor_backward_step(Tensor *n) {
    switch(n->oper) {
        case T_LEAF:
            break;
        case T_ADD:
            for(int i=0;i<n->count;i++) {
                Tensor *input = n->inputs[i];
                for(int j=0;j<input->size;j++) {
                    input->grad[j] += n->grad[j];
                }
            }
            break;
        case T_MUL:
            for(int i=0;i<n->count;i++) {
                Tensor *input = n->inputs[i];
                Tensor *other = n->inputs[1-i];
                for(int j=0;j<input->size;j++) {
                    input->grad[j] += n->grad[j] * other->data[j];
                }
            }
            break;
        case T_RELU:
            {
                Tensor *input = n->inputs[0];
                for(int j=0;j<input->size;j++) {
                    input->grad[j] += n->grad[j] * (input->data[j] > 0 ? 1 : 0);
                }
            }
            break;
        case T_TANH:
            {
                Tensor *input = n->inputs[0];
                for(int j=0;j<input->size;j++) {
                    float tanh_val = n->data[j];
                    input->grad[j] += n->grad[j] * (1 - tanh_val * tanh_val);
                }
            }
            break;
        case T_MATMUL:
            {
                Tensor *A = n->inputs[0], *B = n->inputs[1];
                int m  = A->s.rows;
                int k  = A->s.cols;
                int n_ = B->s.cols;
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n_; j++)
                        for (int p = 0; p < k; p++) {
                            float g = n->grad[i*n_ + j];            /* dC[i][j] */
                            A->grad[i*k + p]  += g * B->data[p*n_ + j];
                            B->grad[p*n_ + j] += g * A->data[i*k + p];
                        }
            }
            break;
        case T_ADD_BIAS:
            {
                Tensor *x = n->inputs[0], *b = n->inputs[1];
                for(int i=0;i<n->s.rows;i++)
                    for(int j=0;j<n->s.cols;j++){
                        x->grad[i*n->s.cols+j]+=n->grad[i*n->s.cols+j];
                        b->grad[j]+=n->grad[i*n->s.cols+j];
                    }
            }
            break;
        case T_MEAN:
            {
                Tensor *x=n->inputs[0];
                for(int k=0;k<x->size;k++){
                    x->grad[k]+=n->grad[0]/x->size;
                }
            }
            break;
        case T_SUB:
            { 
                Tensor *a=n->inputs[0], *b = n->inputs[1];
                for(int k=0;k<n->size;k++){
                    a->grad[k]+=n->grad[k];
                    b->grad[k]-=n->grad[k];
                }
            }
            break;
        case T_SOFTMAX_CE:
            {
                Tensor *z = n->inputs[0], *lab = n->inputs[1];
                int rows = z->s.rows, cols = z->s.cols;
                float scale = n->grad[0] / rows;
                for (int i = 0; i < rows; i++) {
                    int c = (int)lab->data[i];
                    float m, sum;
                    row_max_sum(&z->data[i * cols], cols, &m, &sum);
                    for (int j = 0; j < cols; j++) {
                        float p = expf(z->data[i * cols + j] - m) / sum;
                        float onehot = (j == c) ? 1.0f : 0.0f;
                        z->grad[i * cols + j] += (p - onehot) * scale;
                    }
                }
            }
            break;
        default:
            break;
    }
}
void tensor_build(Tensor *n, Tensor **list, int *len) {
    if(n->visited){return;}  
    n->visited = 1;
    for(int i=0;i<n->count;i++) {
        tensor_build(n->inputs[i], list, len);
    }
    if(*len>=TMAX_NODES){fprintf(stderr,"Error: Graph too large\n");exit(1);}
    list[*len] = n;
    (*len)++;
}
void tensor_backward(Tensor *loss) {
    int max_len = TMAX_NODES;
    Tensor **list = (Tensor **)malloc(sizeof(Tensor *)*max_len);
    if(!list){fprintf(stderr,"tensor_backward: out of memory\n");return;}
    int len = 0;
    tensor_build(loss, list, &len);
    for(int i=0;i<len;i++){
        list[i]->visited = 0;
        memset(list[i]->grad, 0, sizeof(float)*list[i]->size);
    }
    for(int i=0;i<loss->size;i++) {
        loss->grad[i] = 1.0f;
    }
    for(int i=len-1;i>=0;i--) {
        tensor_backward_step(list[i]);
    }
    free(list);
}
void tensor_free_graph(Tensor *loss) {
    if (!loss) return;
    Tensor *list[TMAX_NODES];
    int len=0;
    tensor_build(loss, list, &len);
    for (int i = 0; i < len; i++) {
        Tensor *n = list[i];
        if (n->oper==T_LEAF) {
            n->visited=0;
            continue;
        }
        else{
            tensor_free(n);
        }
    }
    
}
void tensor_free(Tensor *t) {
    if(!t) return;
    free(t->data);
    free(t->grad);
    free(t->inputs);
    free(t);
}