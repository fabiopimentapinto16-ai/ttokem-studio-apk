import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import gptq_cubos as g
import cubo_quant as cq


def dados(seed=0, R=64, C=256, n=4000):
    r = np.random.RandomState(seed)
    A = r.randn(16, C).astype("f")
    X = r.randn(n, 16).astype("f") @ A + 0.1 * r.randn(n, C).astype("f")      # entradas correlacionadas (como nas redes reais)
    W = (r.randn(R, C) * 0.05).astype("f")
    return W, (X.T @ X / n).astype("f"), X


class T(unittest.TestCase):
    def test_rtn_igual_cubo_quant(self):
        W, H, X = dados()
        for bl, b in (((4, 16), 4), ((4, 8), 8)):
            pk, es, R, C = cq.quantizar(W, bl, b)
            ref = cq.desquantizar(pk, es, R, C, (R, C), bl, np.float32, b)
            self.assertTrue(np.array_equal(ref, g.rtn(W, bl, b)))

    def test_gptq_melhora_a_saida(self):
        W, H, X = dados()
        for bits in (3, 4):
            e_rtn = g.erro_rel(W, g.rtn(W, (4, 16), bits), H)
            e_gp = g.erro_rel(W, g.gptq(W, H, (4, 16), bits), H)
            self.assertLess(e_gp, 0.7 * e_rtn, (bits, e_rtn, e_gp))

    def test_valores_na_grade(self):
        W, H, X = dados(R=8, C=64)
        for bits in (3, 4):
            Q = g.gptq(W, H, (4, 16), bits)
            for c0 in range(0, 64, 16):
                for r0 in range(0, 8, 4):
                    b = Q[r0:r0 + 4, c0:c0 + 16]
                    self.assertLessEqual(len(np.unique(b)), 2 ** bits)           # cabe nos bits prometidos
                    nz = np.unique(np.abs(b[b != 0]))
                    if len(nz):                                                   # grade uniforme: tudo e multiplo do degrau
                        passo = nz.min()
                        k = nz / passo
                        self.assertTrue(np.allclose(k, np.rint(k), atol=1e-3) or len(nz) <= 2 ** (bits - 1))

    def test_blocos_diagonais(self):
        W, H, X = dados(C=256)
        Hb = [H[i:i + 128, i:i + 128] for i in (0, 128)]
        Q, e, S = g.gptq_blocos(W, Hb, 128, bloco=(4, 16), bits=4)
        self.assertEqual(S.shape, (16, 16))                                  # 64/4 blocos de linhas x 256/16 grupos
        self.assertEqual(Q.shape, W.shape)
        self.assertLess(e, g.erro_rel(W, g.rtn(W, (4, 16), 4), H) * 1.5)

    def test_escalas_reproduzem_q(self):
        W, H, X = dados(R=8, C=64)
        Q, S = g.gptq(W, H, (4, 16), 4, devolver_escalas=True)
        for c in range(4):
            for b in range(2):
                sc = np.float32(S[b, c])
                blk = Q[b * 4:(b + 1) * 4, c * 16:(c + 1) * 16] / sc
                self.assertTrue(np.allclose(blk, np.rint(blk), atol=1e-3))
                self.assertLessEqual(blk.max(), 7.001); self.assertGreaterEqual(blk.min(), -8.001)

    def test_pre_calculo_identico(self):
        W, H, X = dados(C=256)
        Hb = [H[i:i + 128, i:i + 128] for i in (0, 128)]
        a = g.gptq_blocos(W, Hb, 128, bloco=(4, 16), bits=3)
        pre = [g.preparar(h) for h in Hb]
        b = g.gptq_blocos(W, Hb, 128, pre=pre, bloco=(4, 16), bits=3)
        c = g.gptq_blocos(W, Hb, 128, pre=pre, bloco=(4, 16), bits=4)        # reaproveita em outra opcao
        self.assertTrue(np.array_equal(a[0], b[0]) and a[1] == b[1])
        self.assertFalse(np.array_equal(b[0], c[0]))

    def test_colunas_mortas(self):
        W, H, X = dados(C=64, n=500)
        H[:, 5] = 0; H[5, :] = 0
        Q = g.gptq(W, H, (4, 16), 4)
        self.assertTrue(np.isfinite(Q).all())


if __name__ == "__main__":
    unittest.main()
