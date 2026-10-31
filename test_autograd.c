/* Scalar autograd engine tests.
   Build: gcc -Wall -Wextra autograd.c test_autograd.c -o test_autograd -lm
   Run:   ./test_autograd        (exit code 0 means every check passed) */
#include <stdio.h>
#include <math.h>
#include "autograd.h"

static int checks = 0, fails = 0;

static void check_near(const char *name, float got, float want, float tol) {
    checks++;
    if (fabsf(got - want) > tol) {
        fails++;
        printf("FAIL %-34s got %.5f  want %.5f\n", name, got, want);
    }
}

int main(void) {
    /* L = (a*b) + c  with a=2 b=3 c=4: L=10, grads 3 2 1; second call must not double them */
    {
        Node *a = leafcreate(2), *b = leafcreate(3), *c = leafcreate(4);
        Node *L = add(mult(a, b), c);
        backward(L);
        check_near("T1 value", L->val, 10, 1e-6f);
        check_near("T1 da", a->grad, 3, 1e-6f);
        check_near("T1 db", b->grad, 2, 1e-6f);
        check_near("T1 dc", c->grad, 1, 1e-6f);
        backward(L);
        check_near("T1 da (second call)", a->grad, 3, 1e-6f);
        check_near("T1 dc (second call)", c->grad, 1, 1e-6f);
    }

    /* shared inputs: a+a -> 2, a*a at 3 -> 6 */
    {
        Node *a = leafcreate(3);
        backward(add(a, a));
        check_near("a+a gradient", a->grad, 2, 1e-6f);
        Node *a2 = leafcreate(3);
        backward(mult(a2, a2));
        check_near("a*a gradient", a2->grad, 6, 1e-6f);
    }

    /* shared intermediate node: x=a*b, y=x+x, L=y*x (a=2 b=3): L=72, da=72, db=48 */
    {
        Node *a = leafcreate(2), *b = leafcreate(3);
        Node *x = mult(a, b), *y = add(x, x), *L = mult(y, x);
        backward(L);
        check_near("shared node value", L->val, 72, 1e-4f);
        check_near("shared node da", a->grad, 72, 1e-4f);
        check_near("shared node db", b->grad, 48, 1e-4f);
    }

    /* relu: blocked branch (inner value -5) gives zero everywhere */
    {
        Node *a = leafcreate(2), *b = leafcreate(-3), *c = leafcreate(1);
        Node *L = relu(add(mult(a, b), c));
        backward(L);
        check_near("relu blocked value", L->val, 0, 1e-6f);
        check_near("relu blocked da", a->grad, 0, 1e-6f);
        check_near("relu blocked db", b->grad, 0, 1e-6f);
        check_near("relu blocked dc", c->grad, 0, 1e-6f);
    }

    /* tanh at 0.5: derivative 1 - tanh^2 = 0.78645 */
    {
        Node *a = leafcreate(0.5f);
        backward(tanh_op(a));
        check_near("tanh derivative at 0.5", a->grad, 0.78645f, 1e-4f);
    }

    /* sub and div */
    {
        Node *a = leafcreate(5), *b = leafcreate(3);
        Node *L = sub(a, b);
        backward(L);
        check_near("sub value", L->val, 2, 1e-6f);
        check_near("sub da", a->grad, 1, 1e-6f);
        check_near("sub db", b->grad, -1, 1e-6f);

        Node *s = leafcreate(5);
        backward(sub(s, s));
        check_near("a-a gradient", s->grad, 0, 1e-6f);

        Node *p = leafcreate(6), *q = leafcreate(3);
        Node *D = div_op(p, q);
        backward(D);
        check_near("div value", D->val, 2, 1e-6f);
        check_near("div da (1/b)", p->grad, 1.0f / 3, 1e-5f);
        check_near("div db (-a/b^2)", q->grad, -2.0f / 3, 1e-5f);
    }

    /* composite tanh((a-b)/c), a=1.2 b=0.4 c=1.5, values checked against numerical gradients */
    {
        Node *a = leafcreate(1.2f), *b = leafcreate(0.4f), *c = leafcreate(1.5f);
        backward(tanh_op(div_op(sub(a, b), c)));
        check_near("composite da", a->grad, 0.50795f, 1e-4f);
        check_near("composite db", b->grad, -0.50795f, 1e-4f);
        check_near("composite dc", c->grad, -0.27091f, 1e-4f);
    }

    /* memory: build, backward and free 100000 graphs; the leaf must stay intact and correct */
    {
        Node *a = leafcreate(3);
        for (int i = 0; i < 100000; i++) {
            Node *L = add(relu(a), relu(a));
            backward(L);
            free_graph(L);
        }
        check_near("after 100000 free_graph loops", a->grad, 2, 1e-6f);
    }

    if (fails) {
        printf("%d of %d checks FAILED\n", fails, checks);
        return 1;
    }
    printf("all %d checks passed\n", checks);
    return 0;
}
