/* ttgemm: sgemm float32 row-major, multi-thread (pthreads). C[m,n] = A[m,k] @ B[k,n].
 * Micro-kernel 8x8 com os 16 acumuladores em registradores (NEON fmla por elemento); B empacotado por faixa de 8 colunas.
 * Sem NEON (testes no PC) usa vetores genericos do compilador com a mesma estrutura. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
typedef float32x4_t f4;
#define LD4(p) vld1q_f32(p)
#define ST4(p, v) vst1q_f32(p, v)
#define Z4() vdupq_n_f32(0.f)
#define FMAL(acc, b, a, l) vfmaq_laneq_f32(acc, b, a, l)
#else
typedef float f4 __attribute__((vector_size(16)));
static inline f4 LD4(const float *p) { f4 v; memcpy(&v, p, 16); return v; }
static inline void ST4(float *p, f4 v) { memcpy(p, &v, 16); }
#define Z4() ((f4){0.f, 0.f, 0.f, 0.f})
#define FMAL(acc, b, a, l) ((acc) + (b) * (a)[l])
#endif

#define KB 256

typedef struct { const float *A, *B; float *C; int m, k, n, i0, i1, j0, j1; } Job;

/* C[8x8] (+)= A[8 x kb] @ Bp[kb x 8]; A tem passo lda; Bp e continuo (passo 8). k4 = kb & ~3. */
static void mk8x8(const float *A, int lda, const float *Bp, float *C, int ldc, int kb, int k4, int acc_in) {
    f4 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71;
    if (acc_in) {
        c00 = LD4(C);            c01 = LD4(C + 4);
        c10 = LD4(C + ldc);      c11 = LD4(C + ldc + 4);
        c20 = LD4(C + 2 * ldc);  c21 = LD4(C + 2 * ldc + 4);
        c30 = LD4(C + 3 * ldc);  c31 = LD4(C + 3 * ldc + 4);
        c40 = LD4(C + 4 * ldc);  c41 = LD4(C + 4 * ldc + 4);
        c50 = LD4(C + 5 * ldc);  c51 = LD4(C + 5 * ldc + 4);
        c60 = LD4(C + 6 * ldc);  c61 = LD4(C + 6 * ldc + 4);
        c70 = LD4(C + 7 * ldc);  c71 = LD4(C + 7 * ldc + 4);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = Z4();
        c40 = c41 = c50 = c51 = c60 = c61 = c70 = c71 = Z4();
    }
    const float *a0 = A, *a1 = A + lda, *a2 = A + 2 * lda, *a3 = A + 3 * lda;
    const float *a4 = A + 4 * lda, *a5 = A + 5 * lda, *a6 = A + 6 * lda, *a7 = A + 7 * lda;
    for (int p = 0; p < k4; p += 4) {
        f4 x0 = LD4(a0 + p), x1 = LD4(a1 + p), x2 = LD4(a2 + p), x3 = LD4(a3 + p);
        f4 x4 = LD4(a4 + p), x5 = LD4(a5 + p), x6 = LD4(a6 + p), x7 = LD4(a7 + p);
        const float *b = Bp + p * 8;
#define STEP(L) { f4 b0 = LD4(b + 8 * (L)), b1 = LD4(b + 8 * (L) + 4); \
        c00 = FMAL(c00, b0, x0, L); c01 = FMAL(c01, b1, x0, L); \
        c10 = FMAL(c10, b0, x1, L); c11 = FMAL(c11, b1, x1, L); \
        c20 = FMAL(c20, b0, x2, L); c21 = FMAL(c21, b1, x2, L); \
        c30 = FMAL(c30, b0, x3, L); c31 = FMAL(c31, b1, x3, L); \
        c40 = FMAL(c40, b0, x4, L); c41 = FMAL(c41, b1, x4, L); \
        c50 = FMAL(c50, b0, x5, L); c51 = FMAL(c51, b1, x5, L); \
        c60 = FMAL(c60, b0, x6, L); c61 = FMAL(c61, b1, x6, L); \
        c70 = FMAL(c70, b0, x7, L); c71 = FMAL(c71, b1, x7, L); }
        STEP(0) STEP(1) STEP(2) STEP(3)
    }
    ST4(C, c00);            ST4(C + 4, c01);
    ST4(C + ldc, c10);      ST4(C + ldc + 4, c11);
    ST4(C + 2 * ldc, c20);  ST4(C + 2 * ldc + 4, c21);
    ST4(C + 3 * ldc, c30);  ST4(C + 3 * ldc + 4, c31);
    ST4(C + 4 * ldc, c40);  ST4(C + 4 * ldc + 4, c41);
    ST4(C + 5 * ldc, c50);  ST4(C + 5 * ldc + 4, c51);
    ST4(C + 6 * ldc, c60);  ST4(C + 6 * ldc + 4, c61);
    ST4(C + 7 * ldc, c70);  ST4(C + 7 * ldc + 4, c71);
    for (int p = k4; p < kb; p++)                       /* sobra de k (kb nao multiplo de 4) */
        for (int r = 0; r < 8; r++) {
            float x = A[(size_t)r * lda + p];
            for (int c = 0; c < 8; c++) C[(size_t)r * ldc + c] += x * Bp[p * 8 + c];
        }
}

static void bloco(const Job *J) {
    const int k = J->k, n = J->n;
    float Bp[KB * 8] __attribute__((aligned(16)));
    for (int p0 = 0; p0 < k; p0 += KB) {
        int kb = (k - p0 < KB) ? k - p0 : KB, k4 = kb & ~3, primeiro = (p0 == 0);
        for (int j = J->j0; j < J->j1; j += 8) {
            int nb = (J->j1 - j < 8) ? J->j1 - j : 8;
            for (int p = 0; p < kb; p++) {
                const float *src = J->B + (size_t)(p0 + p) * n + j;
                float *dst = Bp + p * 8;
                int c = 0;
                for (; c < nb; c++) dst[c] = src[c];
                for (; c < 8; c++) dst[c] = 0.f;
            }
            int i = J->i0;
            for (; i + 8 <= J->i1; i += 8) {
                const float *A = J->A + (size_t)i * k + p0;
                float *C = J->C + (size_t)i * n + j;
                if (nb == 8) {
                    mk8x8(A, k, Bp, C, n, kb, k4, !primeiro);
                } else {
                    float T[64] __attribute__((aligned(16)));
                    if (!primeiro) {
                        memset(T, 0, sizeof(T));
                        for (int r = 0; r < 8; r++) memcpy(T + r * 8, C + (size_t)r * n, sizeof(float) * nb);
                    }
                    mk8x8(A, k, Bp, T, 8, kb, k4, !primeiro);
                    for (int r = 0; r < 8; r++) memcpy(C + (size_t)r * n, T + r * 8, sizeof(float) * nb);
                }
            }
            for (; i < J->i1; i++) {                    /* linhas que sobram (m nao multiplo de 8) */
                const float *ar = J->A + (size_t)i * k + p0;
                float *cr = J->C + (size_t)i * n + j;
                for (int c = 0; c < nb; c++) {
                    float s = primeiro ? 0.f : cr[c];
                    for (int p = 0; p < kb; p++) s += ar[p] * Bp[p * 8 + c];
                    cr[c] = s;
                }
            }
        }
    }
}

static void *trabalho(void *p) { bloco((Job *)p); return NULL; }

static void sgemm(const float *A, const float *B, float *C, int m, int k, int n, int nt) {
    Job js[16]; pthread_t th[16];
    if (nt < 1) nt = 1;
    if (nt > 16) nt = 16;
    if ((double)m * k * n < 2e6) nt = 1;
    int tiles = m / 8, faixas = (n + 7) / 8;
    int por_linha = tiles >= faixas;                    /* divide pela dimensao que tem mais blocos */
    int unid = por_linha ? tiles : faixas;
    if (nt > unid) nt = unid > 0 ? unid : 1;
    int base = unid / nt, resto = unid % nt, u = 0;
    for (int t = 0; t < nt; t++) {
        int c = base + (t < resto ? 1 : 0);
        Job J = {A, B, C, m, k, n, 0, m, 0, n};
        if (por_linha) { J.i0 = u * 8; J.i1 = (t == nt - 1) ? m : (u + c) * 8; }
        else { J.j0 = u * 8; J.j1 = (t == nt - 1) ? n : (u + c) * 8; }
        js[t] = J; u += c;
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
