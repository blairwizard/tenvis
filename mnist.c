#include "mnist.h"
#include "tensor.h"
#include<stdio.h>
#include<stdlib.h>
static int read_be32(FILE *f, uint32_t *out) {
    unsigned char b[4];
    if(fread(b,1,4,f)!=4) return 0;
    *out=((uint32_t)b[0]<<24) | ((uint32_t)b[1]<<16) | ((uint32_t)b[2]<<8) | ((uint32_t)b[3]);
    return 1;
}
Tensor *load_images(const char *path, int max_n){
    FILE *f = fopen(path, "rb");
    if(!f) {
        fprintf(stderr, "load_images: cannot open %s\n",path); return NULL;
    }
    uint32_t magic, count, rows, cols;
    if (!read_be32(f, &magic) || !read_be32(f, &count) || !read_be32(f, &rows) || !read_be32(f, &cols)) {
        fclose(f); fprintf(stderr, "load_images: truncated header\n"); return NULL;
    }
    if(magic != 2051 || rows!=28 || cols!= 28) {
        fclose(f);
        fprintf(stderr,"load_images:invalid");
        return NULL;
    }
    int n= (max_n>0 && (uint32_t)max_n<count)?max_n:(int)count;
    Tensor *t = tensor_create(NULL,n,784);
    if(!t){fclose(f); return NULL;}
    unsigned char buf[784];
    for(int i=0;i<n;i++){
        if(fread(buf,1,784,f)!=784){
            tensor_free(t);
            fclose(f);
            return NULL;
        }
        for(int p=0;p<784;p++) t->data[i*784+p] = buf[p]/255.0f;

    }
    fclose(f);
    return t;
}

Tensor *load_labels(const char *path, int max_n){
    FILE *f = fopen(path,"rb");
    if(!f){fprintf(stderr, "load_labels: cannot open %s\n",path);return NULL;}
    uint32_t magic, count;
    if (!read_be32(f, &magic) || !read_be32(f, &count)) {
        fclose(f); fprintf(stderr, "load_images: truncated header\n"); return NULL;
    }
    if (magic != 2049) {
        fclose(f);
        fprintf(stderr, "load_labels: bad magic %u (expected 2049)\n", magic);
        return NULL;
    }
    int n = (max_n > 0 && (uint32_t)max_n < count)? max_n: (int)count;
    Tensor *t = tensor_create(NULL,n,1);
    if(!t){fclose(f); return NULL;}
    unsigned char *buf = malloc(n);
    if(!buf){tensor_free(t);fclose(f);return NULL;}
    if (fread(buf, 1, n, f) != (size_t)n) {
        free(buf); tensor_free(t); fclose(f);
        fprintf(stderr, "load_labels: truncated data\n");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (buf[i] > 9) {
            fprintf(stderr, "load_labels: label %u at index %d is out of range\n", buf[i], i);
            free(buf); tensor_free(t); fclose(f);
            return NULL;
        }
        t->data[i] = (float)buf[i];
    }
    free(buf);
    fclose(f);
    return t;
}


