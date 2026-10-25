#ifndef AUTO_H
#define AUTO_H

enum op{
    LEAF,
    ADD,
    SUB,
    MUL,
    DIV,
    RELU,
    TANH,
};
typedef enum op Op;

struct Node{
    float val;
    float grad;
    struct Node **inputs;
    int count;
    Op oper;
    int visited;
};

typedef struct Node Node;
Node *leafcreate(float val);
Node *add(Node *a,Node *b);
Node *mult(Node *a,Node *b);
Node *sub(Node *a,Node *b);
Node *div_op(Node *a,Node *b);
Node *relu(Node *a);
Node *tanh_op(Node *a);
void  backward_step(Node *n);
void backward(Node *loss);
void build(Node *n, Node **list, int *len);
void free_graph(Node *loss);
#endif