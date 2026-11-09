/* ttgemm: sgemm float32 row-major, multi-thread (pthreads), vetorizavel (NEON no ARM). C[m,n] = A[m,k] @ B[k,n]. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <string.h>

#define NB 128      /* colunas por bloco */
#define KB 256      /* profundidade por bloco */
#define MR 4        /* linhas por micro-bloco */

typedef struct { const float *A, *B; float *C; int m, k, n, i0, i1; } Job;

static void bloco(const Job *J) {
    const int k = J->k, n = J->n;
    for (int j0 = 0; j0 < n; j0 += NB) {
        int nb = (n - j0 < NB) ? n - j0 : NB;
        for (int i = J->i0; i < J->i1; i += MR) {
            int mr = (J->i1 - i < MR) ? J->i1 - i : MR;
            float acc[MR][NB] __attribute__((aligned(16)));
            memset(acc, 0, sizeof(acc));
            for (int p0 = 0; p0 < k; p0 += KB) {
                int pe = (p0 + KB < k) ? p0 + KB : k;
                if (mr == MR) {
                    const float *a0 = J->A + (size_t)i * k, *a1 = a0 + k, *a2 = a1 + k, *a3 = a2 + k;
                    for (int p = p0; p < pe; p++) {
                        const float *b = J->B + (size_t)p * n + j0;
                        float x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
                        for (int j = 0; j < nb; j++) {
                            float bj = b[j];
                            acc[0][j] += x0 * bj; acc[1][j] += x1 * bj;
                            acc[2][j] += x2 * bj; acc[3][j] += x3 * bj;
                        }
                    }
                } else {
                    for (int r = 0; r < mr; r++) {
                        const float *ar = J->A + (size_t)(i + r) * k;
                        for (int p = p0; p < pe; p++) {
                            const float *b = J->B + (size_t)p * n + j0;
                            float x = ar[p];
                            for (int j = 0; j < nb; j++) acc[r][j] += x * b[j];
                        }
                    }
                }
            }
            for (int r = 0; r < mr; r++)
                memcpy(J->C + (size_t)(i + r) * n + j0, acc[r], sizeof(float) * nb);
        }
    }
}

static void *trabalho(void *p) { bloco((Job *)p); return NULL; }

static void sgemm(const float *A, const float *B, float *C, int m, int k, int n, int nt) {
    if (nt < 1) nt = 1;
    if ((double)m * k * n < 2e6 || m < MR * 2) nt = 1;
    if (nt > m / MR) nt = m / MR > 0 ? m / MR : 1;
    Job js[16]; pthread_t th[16];
    if (nt > 16) nt = 16;
    int base = m / nt, resto = m % nt, i = 0;
    for (int t = 0; t < nt; t++) {
        int c = base + (t < resto ? 1 : 0);
        /* fatias multiplas de MR sempre que possivel */
        js[t] = (Job){A, B, C, m, k, n, i, i + c};
        i += c;
    }
    int criados = 0;
    for (int t = 1; t < nt; t++) {
        if (pthread_create(&th[t], NULL, trabalho, &js[t]) == 0) criados |= (1 << t);
        else bloco(&js[t]);
    }
    bloco(&js[0]);
    for (int t = 1; t < nt; t++) if (criados & (1 << t)) pthread_join(th[t], NULL);
}

static PyObject *py_sgemm(PyObject *self, PyObject *args) {
    Py_buffer a, b, c; int m, k, n, nt = 4;
    if (!PyArg_ParseTuple(args, "y*y*w*iii|i", &a, &b, &c, &m, &k, &n, &nt)) return NULL;
    PyObject *r = NULL;
    if ((Py_ssize_t)m * k * 4 != a.len || (Py_ssize_t)k * n * 4 != b.len || (Py_ssize_t)m * n * 4 != c.len)
        PyErr_SetString(PyExc_ValueError, "ttgemm: tamanhos nao batem");
    else {
        Py_BEGIN_ALLOW_THREADS
        sgemm((const float *)a.buf, (const float *)b.buf, (float *)c.buf, m, k, n, nt);
        Py_END_ALLOW_THREADS
        Py_INCREF(Py_None); r = Py_None;
    }
    PyBuffer_Release(&a); PyBuffer_Release(&b); PyBuffer_Release(&c);
    return r;
}

static PyMethodDef M[] = {{"sgemm", py_sgemm, METH_VARARGS, "sgemm(a,b,c,m,k,n,nthreads)"}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef D = {PyModuleDef_HEAD_INIT, "ttgemm", NULL, -1, M};
PyMODINIT_FUNC PyInit_ttgemm(void) { return PyModule_Create(&D); }
