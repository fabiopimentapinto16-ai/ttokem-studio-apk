import os, sys, tempfile, tracemalloc, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import cubo_quant as cq
import motor_cubico as mc

R = np.random.default_rng(3)


def conv_ingenua(x, w, b, stride, pad):
    C, H, W = x.shape; O, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (W + 2 * pad - kw) // stride + 1
    out = np.zeros((O, Ho, Wo), np.float32)
    for o in range(O):
        for i in range(Ho):
            for j in range(Wo):
                out[o, i, j] = (xp[:, i*stride:i*stride+kh, j*stride:j*stride+kw] * w[o]).sum() + b[o]
    return out


class Ops(unittest.TestCase):
    def test_conv_igual_a_ingenua(self):
        for stride, pad, k in ((1, 1, 3), (2, 1, 3), (1, 0, 1)):
            x = R.standard_normal((3, 9, 11)).astype(np.float32)
            w = R.standard_normal((5, 3, k, k)).astype(np.float32)
            b = R.standard_normal(5).astype(np.float32)
            np.testing.assert_allclose(mc.conv2d(x, w, b, stride, pad, linhas=2),
                                       conv_ingenua(x, w, b, stride, pad), atol=1e-4)

    def test_conv_por_partes_igual_inteira(self):
        x = R.standard_normal((4, 20, 20)).astype(np.float32); w = R.standard_normal((6, 4, 3, 3)).astype(np.float32)
        np.testing.assert_allclose(mc.conv2d(x, w, None, 1, 1, linhas=1), mc.conv2d(x, w, None, 1, 1, linhas=99), atol=1e-4)

    def test_group_norm_media_zero_variancia_um(self):
        x = R.standard_normal((8, 5, 5)).astype(np.float32) * 3 + 2
        y = mc.group_norm(x, np.ones(8, np.float32), np.zeros(8, np.float32), grupos=4)
        g = y.reshape(4, -1)
        np.testing.assert_allclose(g.mean(1), 0, atol=1e-4); np.testing.assert_allclose(g.var(1), 1, atol=1e-3)

    def test_atencao_por_blocos_igual_inteira(self):
        q = R.standard_normal((50, 16)).astype(np.float32); k = R.standard_normal((7, 16)).astype(np.float32)
        v = R.standard_normal((7, 16)).astype(np.float32)
        np.testing.assert_allclose(mc.atencao(q, k, v, 4, bloco=8), mc.atencao(q, k, v, 4, bloco=999), atol=1e-5)
        # manual, 1 cabeca
        s = q @ k.T / 4.0; p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        np.testing.assert_allclose(mc.atencao(q, k, v, 1), p @ v, atol=1e-5)


class MiniUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.est = mc.pesos_aleatorios_mini(canais=256)
        cls.d = tempfile.mkdtemp(); cls.p = os.path.join(cls.d, "mini.ttq")
        cls.info = cq.salvar(cls.est, cls.p)
        cls.x = R.standard_normal((4, 8, 8)).astype(np.float32)
        cls.ctx = R.standard_normal((8, 32)).astype(np.float32)

    def test_streaming_igual_a_tudo_na_ram(self):
        P = mc.Pesos(self.p)
        with cq.Leitor(self.p) as lt:
            tudo = mc.PesosMemoria({k: lt.ler(k) for k in lt.nomes()})
        a = mc.mini_unet(self.x, 10.0, self.ctx, P, canais=256)
        b = mc.mini_unet(self.x, 10.0, self.ctx, tudo, canais=256)
        P.fechar()
        np.testing.assert_allclose(a, b, atol=1e-5)
        self.assertEqual(a.shape, (4, 8, 8)); self.assertTrue(np.isfinite(a).all())

    def test_qualidade_4_e_8_bits(self):
        f = mc.mini_unet(self.x, 10.0, self.ctx, mc.PesosMemoria(self.est), canais=256)
        res = {}
        for bits in (4, 8):
            p = os.path.join(self.d, "m%d.ttq" % bits)
            cq.salvar(self.est, p, bits=bits)
            P = mc.Pesos(p)
            q = mc.mini_unet(self.x, 10.0, self.ctx, P, canais=256); P.fechar()
            res[bits] = cq.erro(f, q)["snr_db"]
        print("\n  SNR da saida do mini-UNet vs fp32:  4 bits = %.1f dB | 8 bits = %.1f dB" % (res[4], res[8]))
        self.assertGreater(res[8], 30)         # 8 bits: praticamente igual
        self.assertGreater(res[4], 5)          # 4 bits: perde bastante (honesto)
        self.assertGreater(res[8], res[4] + 15)

    def test_pico_de_ram_menor_que_o_modelo(self):
        modelo_mb = sum(v.nbytes for v in self.est.values()) / 2**20
        P = mc.Pesos(self.p)
        mc.mini_unet(self.x, 1.0, self.ctx, P, canais=256)           # aquece
        tracemalloc.start()
        mc.mini_unet(self.x, 1.0, self.ctx, P, canais=256)
        pico = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop(); P.fechar()
        print("\n  modelo fp32 %.1f MB | arquivo %.1f MB | pico de RAM na inferencia %.1f MB"
              % (modelo_mb, self.info["arquivo_mb"], pico))
        self.assertLess(pico, 0.5 * modelo_mb)           # nunca carrega o modelo inteiro

    def test_cache_pequeno_reduz_leituras(self):
        P0 = mc.Pesos(self.p, cache_mb=0); mc.mini_unet(self.x, 1.0, self.ctx, P0, canais=256)
        P1 = mc.Pesos(self.p, cache_mb=64); mc.mini_unet(self.x, 1.0, self.ctx, P1, canais=256); mc.mini_unet(self.x, 2.0, self.ctx, P1, canais=256)
        self.assertLess(P1.leituras, 2 * P0.leituras)
        P0.fechar(); P1.fechar()


if __name__ == "__main__":
    unittest.main()
