#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "autograd.h"
#define MAX_NODES 1024

Node *leafcreate(float val){
    Node *n = (Node *)malloc(sizeof(Node));
    if(n==NULL){return NULL;}
    n->val=val;
    n->grad=0.0;
    n->inputs=NULL;
    n->count=0;
    n->oper=LEAF;
    n->visited=0;
    return n;
}

Node *add(Node *a,Node *b){
    float v=a->val+b->val;
    Node *n = leafcreate(v);
    n->inputs=(Node **)malloc(2*sizeof(Node *));
    n->inputs[0]=a;
    n->inputs[1]=b;
    n->count=2;
    n->oper=ADD;
    return n;
}

Node *mult(Node *a,Node *b){
    float v=a->val*b->val;
    Node *n = leafcreate(v);
    n->inputs=(Node **)malloc(2*sizeof(Node *));
    if(n->inputs==NULL){return NULL;}
    n->inputs[0]=a;
    n->inputs[1]=b;
    n->count=2;
    n->oper=MUL;
    return n;
}

void backward_step(Node *n){
    switch(n->oper){
        case ADD:
            n->inputs[0]->grad+=n->grad;
            n->inputs[1]->grad+=n->grad;
            break;
        case MUL:
            n->inputs[0]->grad+=n->inputs[1]->val*n->grad;
            n->inputs[1]->grad+=n->inputs[0]->val*n->grad;
            break;
        case RELU:
            if(n->inputs[0]->val>0){
                n->inputs[0]->grad+=n->grad;
            }
            break;
        case TANH:
            n->inputs[0]->grad+=(1-n->val*n->val)*n->grad;
            break;
        case SUB:
            n->inputs[0]->grad+=n->grad;
            n->inputs[1]->grad-=n->grad;
            break;  
        case DIV:
            n->inputs[0]->grad+=n->grad/n->inputs[1]->val;
            n->inputs[1]->grad-=n->grad*n->inputs[0]->val/(n->inputs[1]->val*n->inputs[1]->val);
            break;
        default:
            return;
    }
}

void build(Node *n, Node **list, int *len){
    if(n->visited==1){return;}
    n->visited=1;
    for(int i=0;i<n->count;i++){
        build(n->inputs[i],list,len);
    }
    if(*len>=MAX_NODES){fprintf(stderr,"Error: Exceeded maximum number of nodes\n");exit(1);}
    list[*len]=n;
    (*len)++;
}

void backward(Node *loss){
    Node *list[MAX_NODES];
    int len = 0;
    build(loss, list, &len);
    for (int i = 0; i < len; i++) {      
        list[i]->visited = 0;
        list[i]->grad = 0;
    }
    loss->grad = 1;                      
    for (int i = len - 1; i >= 0; i--) { 
        backward_step(list[i]);
    }
}

Node *relu(Node *a){
    float val = a->val > 0 ? a->val : 0;
    Node *n = leafcreate(val);
    n->inputs = (Node **)malloc(sizeof(Node *));
    n->count = 1;
    n->inputs[0] = a;
    n->oper = RELU;
    return n;
}

Node *tanh_op(Node *a){
    float val = tanhf(a->val);
    Node *n = leafcreate(val);
    n->inputs = (Node **)malloc(sizeof(Node *));
    n->count = 1;
    n->inputs[0] = a;
    n->oper = TANH;
    return n;
}

Node *sub(Node *a, Node *b) {
    float val= a->val-b->val;
    Node *n=leafcreate(val);
    n->inputs=(Node **)malloc(2*sizeof(Node *));
    if(n->inputs==NULL){return NULL;}
    n->inputs[0]=a;
    n->inputs[1]=b;
    n->count=2;
    n->oper=SUB;
    return n;
}

Node *div_op(Node *a, Node *b) {
    if(b->val == 0) {
        fprintf(stderr, "Error: Division by zero\n");
        exit(1);
    }
    float val= a->val/b->val;
    Node *n=leafcreate(val);
    n->inputs=(Node **)malloc(2*sizeof(Node *));
    if(n->inputs==NULL){return NULL;}
    n->inputs[0]=a;
    n->inputs[1]=b;
    n->count=2;
    n->oper=DIV;
    return n;
}

void free_graph(Node *loss) {
    Node *list[MAX_NODES];
    int len = 0;
    build(loss, list, &len);
    for (int i = 0; i < len; i++) {
        Node *n = list[i];
        if (n->oper==LEAF) {
            n->visited=0;
            continue;
        }
        else{
            free(list[i]->inputs);
            free(list[i]);
        }
    }
}