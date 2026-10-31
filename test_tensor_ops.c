/* Tensor engine tests: every op's forward value and gradient, checked against
   hand-computed numbers, plus batching, accuracy, RNG and a memory loop.
   Build: gcc -Wall -Wextra tensor.c test_tensor_ops.c -o test_tensor_ops -lm
   Run:   ./test_tensor_ops      (exit code 0 means every check passed) */
#include <stdio.h>
#include <math.h>
#include "tensor.h"

static int checks = 0, fails = 0;

static void check_near(const char *name, float got, float want, float tol) {
    checks++;
    if (fabsf(got - want) > tol) {
        fails++;
        printf("FAIL %-34s got %.5f  want %.5f\n", name, got, want);
    }
}

static void check_arr(const char *name, const float *got, const float *want, int n, float tol) {
    checks++;
    for (int i = 0; i < n; i++) {
        if (fabsf(got[i] - want[i]) > tol) {
            fails++;
            printf("FAIL %-34s index %d: got %.5f  want %.5f\n", name, i, got[i], want[i]);
            return;
        }
    }
}

static void check_true(const char *name, int cond) {
    checks++;
    if (!cond) {
        fails++;
        printf("FAIL %s\n", name);
    }
}

int main(void) {
    float av[4] = {1, 2, 3, 4}, bv[4] = {10, 20, 30, 40};

    /* add: values, gradients, shared input */
    {
        Tensor *a = tensor_create(av, 2, 2), *b = tensor_create(bv, 2, 2);
        Tensor *out = tensor_add(a, b);
        float want[4] = {11, 22, 33, 44}, ones[4] = {1, 1, 1, 1};
        check_arr("add values", out->data, want, 4, 1e-6f);
        tensor_backward(out);
        check_arr("add da", a->grad, ones, 4, 1e-6f);
        check_arr("add db", b->grad, ones, 4, 1e-6f);

        Tensor *c = tensor_create(av, 2, 2);
        float twos[4] = {2, 2, 2, 2};
        tensor_backward(tensor_add(c, c));
        check_arr("add shared input", c->grad, twos, 4, 1e-6f);

        Tensor *wide = tensor_create(NULL, 3, 2);
        check_true("add shape mismatch -> NULL", tensor_add(a, wide) == NULL);
    }

    /* mul: each input receives the other's data; a*a gives 2a */
    {
        Tensor *a = tensor_create(av, 2, 2), *b = tensor_create(bv, 2, 2);
        Tensor *out = tensor_mul(a, b);
        float want[4] = {10, 40, 90, 160};
        check_arr("mul values", out->data, want, 4, 1e-6f);
        tensor_backward(out);
        check_arr("mul da (= b)", a->grad, bv, 4, 1e-6f);
        check_arr("mul db (= a)", b->grad, av, 4, 1e-6f);

        Tensor *s = tensor_create(av, 2, 2);
        float two_a[4] = {2, 4, 6, 8};
        tensor_backward(tensor_mul(s, s));
        check_arr("mul a*a gradient", s->grad, two_a, 4, 1e-6f);
    }

    /* relu: blocked and pass-through, and a shared input */
    {
        float rv[4] = {-2, -1, 3, 4};
        Tensor *r = tensor_create(rv, 2, 2);
        Tensor *ro = tensor_relu(r);
        float wv[4] = {0, 0, 3, 4}, wg[4] = {0, 0, 1, 1};
        check_arr("relu values", ro->data, wv, 4, 1e-6f);
        tensor_backward(ro);
        check_arr("relu gradient", r->grad, wg, 4, 1e-6f);

        float sv[2] = {3, -1}, sg[2] = {2, 0};
        Tensor *s = tensor_create(sv, 1, 2);
        tensor_backward(tensor_add(tensor_relu(s), tensor_relu(s)));
        check_arr("relu(a)+relu(a) gradient", s->grad, sg, 2, 1e-6f);
    }

    /* tanh: derivative 1 - tanh^2 (values match the scalar engine) */
    {
        float tv[3] = {0.5f, -1.0f, 2.0f}, tg[3] = {0.78645f, 0.41997f, 0.07065f};
        Tensor *t = tensor_create(tv, 1, 3);
        tensor_backward(tensor_tanh(t));
        check_arr("tanh gradient", t->grad, tg, 3, 1e-4f);
    }

    /* matmul: square, non-square, mismatch */
    {
        float mv[4] = {5, 6, 7, 8};
        Tensor *A = tensor_create(av, 2, 2), *B = tensor_create(mv, 2, 2);
        Tensor *C = tensor_matmul(A, B);
        float wc[4] = {19, 22, 43, 50}, wa[4] = {11, 15, 11, 15}, wb[4] = {4, 4, 6, 6};
        check_arr("matmul values", C->data, wc, 4, 1e-5f);
        tensor_backward(C);
        check_arr("matmul dA", A->grad, wa, 4, 1e-5f);
        check_arr("matmul dB", B->grad, wb, 4, 1e-5f);

        float xv[6] = {1, 2, 3, 4, 5, 6}, yv[3] = {1, 1, 1};
        Tensor *X = tensor_create(xv, 2, 3), *Y = tensor_create(yv, 3, 1);
        Tensor *Z = tensor_matmul(X, Y);
        float wz[2] = {6, 15}, wx[6] = {1, 1, 1, 1, 1, 1}, wy[3] = {5, 7, 9};
        check_arr("matmul 2x3 * 3x1 values", Z->data, wz, 2, 1e-5f);
        tensor_backward(Z);
        check_arr("matmul 2x3 * 3x1 dX", X->grad, wx, 6, 1e-5f);
        check_arr("matmul 2x3 * 3x1 dY", Y->grad, wy, 3, 1e-5f);
        check_true("matmul mismatch -> NULL", tensor_matmul(X, X) == NULL);
    }

    /* add_bias: broadcast forward, gradient summed over rows, batch of one, mismatch */
    {
        float xv[4] = {1, 2, 3, 4}, biasv[2] = {10, 20};
        Tensor *x = tensor_create(xv, 2, 2), *b = tensor_create(biasv, 1, 2);
        Tensor *out = tensor_add_bias(x, b);
        float wo[4] = {11, 22, 13, 24}, wdx[4] = {1, 1, 1, 1}, wdb[2] = {2, 2};
        check_arr("add_bias values", out->data, wo, 4, 1e-6f);
        tensor_backward(out);
        check_arr("add_bias dx", x->grad, wdx, 4, 1e-6f);
        check_arr("add_bias db (sum over rows)", b->grad, wdb, 2, 1e-6f);

        float rv[2] = {5, 6}, w1[2] = {15, 26}, wdb1[2] = {1, 1};
        Tensor *r = tensor_create(rv, 1, 2);
        Tensor *ro = tensor_add_bias(r, b);
        check_arr("add_bias batch of one", ro->data, w1, 2, 1e-6f);
        tensor_backward(ro);
        check_arr("add_bias batch-of-one db", b->grad, wdb1, 2, 1e-6f);

        float badv[3] = {1, 2, 3};
        Tensor *bad = tensor_create(badv, 1, 3);
        check_true("add_bias width mismatch -> NULL", tensor_add_bias(x, bad) == NULL);
    }

    /* mean and sub, and the hand-checked MSE chain */
    {
        Tensor *x = tensor_create(av, 2, 2);
        Tensor *m = tensor_mean(x);
        float quarters[4] = {0.25f, 0.25f, 0.25f, 0.25f};
        check_near("mean value", m->data[0], 2.5f, 1e-6f);
        check_true("mean output is 1x1", m->s.rows == 1 && m->s.cols == 1);
        tensor_backward(m);
        check_arr("mean gradient (1/size)", x->grad, quarters, 4, 1e-6f);

        float pv[2] = {5, 7}, qv[2] = {2, 3}, ws[2] = {3, 4}, da[2] = {1, 1}, db[2] = {-1, -1};
        Tensor *p = tensor_create(pv, 1, 2), *q = tensor_create(qv, 1, 2);
        Tensor *d = tensor_sub(p, q);
        check_arr("sub values", d->data, ws, 2, 1e-6f);
        tensor_backward(d);
        check_arr("sub da", p->grad, da, 2, 1e-6f);
        check_arr("sub db", q->grad, db, 2, 1e-6f);

        float zeros[2] = {0, 0};
        Tensor *s = tensor_create(pv, 1, 2);
        tensor_backward(tensor_sub(s, s));
        check_arr("a-a gradient", s->grad, zeros, 2, 1e-6f);

        float predv[3] = {1, 2, 3}, targv[3] = {2, 2, 5}, wp[3] = {-0.6667f, 0, -1.3333f};
        Tensor *pred = tensor_create(predv, 1, 3), *targ = tensor_create(targv, 1, 3);
        Tensor *diff = tensor_sub(pred, targ);
        Tensor *L = tensor_mean(tensor_mul(diff, diff));
        check_near("MSE value", L->data[0], 5.0f / 3, 1e-5f);
        tensor_backward(L);
        check_arr("MSE dpred", pred->grad, wp, 3, 1e-4f);
    }

    /* softmax cross-entropy: values from the hand calculation, overflow safety, bad label */
    {
        float zv[6] = {2, 1, 0.1f, 0, 0, 0}, lv[2] = {0, 1};
        float wg[6] = {-0.1705f, 0.1212f, 0.0493f, 0.1667f, -0.3333f, 0.1667f};
        Tensor *z = tensor_create(zv, 2, 3), *lab = tensor_create(lv, 2, 1);
        Tensor *L = tensor_softmax_ce(z, lab);
        check_near("softmax_ce loss", L->data[0], 0.7578f, 1e-3f);
        tensor_backward(L);
        check_arr("softmax_ce gradient", z->grad, wg, 6, 1e-3f);

        float big[3] = {1000, 0, 0}, bl[1] = {0};
        Tensor *bz = tensor_create(big, 1, 3), *bt = tensor_create(bl, 1, 1);
        Tensor *BL = tensor_softmax_ce(bz, bt);
        check_true("softmax_ce handles huge logits", BL != NULL && isfinite(BL->data[0]));
        if (BL) check_near("softmax_ce huge logits loss", BL->data[0], 0, 1e-3f);

        float badl[1] = {5};
        Tensor *bad = tensor_create(badl, 1, 1);
        check_true("softmax_ce bad label -> NULL", tensor_softmax_ce(bz, bad) == NULL);
    }

    /* batching and accuracy */
    {
        float v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        Tensor *src = tensor_create(v, 4, 2);
        Tensor *b = tensor_rows(src, 1, 2);
        float want[4] = {3, 4, 5, 6};
        check_arr("rows(1,2) values", b->data, want, 4, 1e-6f);
        check_true("rows result is a clean leaf", b->count == 0 && b->inputs == NULL);
        check_true("rows out of range -> NULL", tensor_rows(src, 3, 2) == NULL);

        float w[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        Tensor *src3 = tensor_create(w, 3, 4);
        Tensor *r = tensor_rows(src3, 1, 1);
        float wr[4] = {5, 6, 7, 8};
        check_arr("rows 3x4 row 1", r->data, wr, 4, 1e-6f);

        float zv[6] = {0.1f, 0.8f, 0.1f, 0.5f, 0.2f, 0.3f};
        float half[2] = {1, 2}, all[2] = {1, 0}, none[2] = {0, 1};
        Tensor *z = tensor_create(zv, 2, 3);
        Tensor *l1 = tensor_create(half, 2, 1), *l2 = tensor_create(all, 2, 1), *l3 = tensor_create(none, 2, 1);
        check_near("accuracy half", tensor_accuracy(z, l1), 0.5f, 1e-6f);
        check_near("accuracy all", tensor_accuracy(z, l2), 1.0f, 1e-6f);
        check_near("accuracy none", tensor_accuracy(z, l3), 0.0f, 1e-6f);
    }

    /* RNG: seeded, uniform in [0,1), centred on 0.5; tensor_rand stays inside [-scale, scale] */
    {
        rng_seed(42);
        double sum = 0;
        int n = 100000, inside = 1;
        for (int i = 0; i < n; i++) {
            float u = rng_uniform();
            sum += u;
            if (u < 0 || u >= 1) inside = 0;
        }
        check_near("rng mean", (float)(sum / n), 0.5f, 0.01f);
        check_true("rng stays in [0,1)", inside);

        Tensor *w = tensor_rand(100, 100, 0.5f);
        int ok = 1;
        double wsum = 0;
        for (int i = 0; i < w->size; i++) {
            if (w->data[i] < -0.5f || w->data[i] > 0.5f) ok = 0;
            wsum += w->data[i];
        }
        check_true("tensor_rand stays in [-scale, scale]", ok);
        check_near("tensor_rand mean near 0", (float)(wsum / w->size), 0.0f, 0.02f);

        rng_seed(7);
        float first = rng_uniform();
        rng_seed(7);
        check_near("same seed repeats the sequence", rng_uniform(), first, 0.0f);
    }

    /* memory: build, backward and free 100000 graphs; the leaf must stay intact and correct */
    {
        float sv[2] = {3, -1}, sg[2] = {2, 0};
        Tensor *a = tensor_create(sv, 1, 2);
        for (int i = 0; i < 100000; i++) {
            Tensor *L = tensor_add(tensor_relu(a), tensor_relu(a));
            tensor_backward(L);
            tensor_free_graph(L);
        }
        check_arr("after 100000 free_graph loops", a->grad, sg, 2, 1e-6f);
    }

    if (fails) {
        printf("%d of %d checks FAILED\n", fails, checks);
        return 1;
    }
    printf("all %d checks passed\n", checks);
    return 0;
}
