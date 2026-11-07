"""Fiacao do UNet SDXS e do TAESD em numpy, com pesos aleatorios de largura reduzida (formas coerentes com o SDXS real)."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ferramentas"))
import motor_cubico as mc      # noqa: E402
import unet_sdxs as us         # noqa: E402
import taesd_numpy as tn       # noqa: E402

C0, C1, C2, CTX = 32, 64, 128, 768


def _res(est, p, cin, cout, rng):
    est[p + ".norm1.weight"], est[p + ".norm1.bias"] = np.ones(cin, np.float32), np.zeros(cin, np.float32)
    est[p + ".conv1.weight"] = rng.randn(cout, cin, 3, 3).astype(np.float32) / np.sqrt(cin * 9)
    est[p + ".conv1.bias"] = np.zeros(cout, np.float32)
    est[p + ".time_emb_proj.weight"] = rng.randn(cout, 4 * C0).astype(np.float32) * 0.02
    est[p + ".time_emb_proj.bias"] = np.zeros(cout, np.float32)
    est[p + ".norm2.weight"], est[p + ".norm2.bias"] = np.ones(cout, np.float32), np.zeros(cout, np.float32)
    est[p + ".conv2.weight"] = rng.randn(cout, cout, 3, 3).astype(np.float32) / np.sqrt(cout * 9)
    est[p + ".conv2.bias"] = np.zeros(cout, np.float32)
    if cin != cout:
        est[p + ".conv_shortcut.weight"] = rng.randn(cout, cin, 1, 1).astype(np.float32) / np.sqrt(cin)
        est[p + ".conv_shortcut.bias"] = np.zeros(cout, np.float32)


def _att(est, p, c, rng):
    w = lambda *s: (rng.randn(*s) / np.sqrt(s[-1])).astype(np.float32)
    z, o = (lambda n: np.zeros(n, np.float32)), (lambda n: np.ones(n, np.float32))
    est[p + ".norm.weight"], est[p + ".norm.bias"] = o(c), z(c)
    est[p + ".proj_in.weight"], est[p + ".proj_in.bias"] = w(c, c, 1, 1), z(c)
    t = p + ".transformer_blocks.0"
    for n in ("norm1", "norm2", "norm3"):
        est["%s.%s.weight" % (t, n)], est["%s.%s.bias" % (t, n)] = o(c), z(c)
    for a, kv in (("attn1", c), ("attn2", CTX)):
        est["%s.%s.to_q.weight" % (t, a)], est["%s.%s.to_k.weight" % (t, a)] = w(c, c), w(c, kv)
        est["%s.%s.to_v.weight" % (t, a)], est["%s.%s.to_out.0.weight" % (t, a)] = w(c, kv), w(c, c)
        est["%s.%s.to_out.0.bias" % (t, a)] = z(c)
    est[t + ".ff.net.0.proj.weight"], est[t + ".ff.net.0.proj.bias"] = w(c * 8, c), z(c * 8)
    est[t + ".ff.net.2.weight"], est[t + ".ff.net.2.bias"] = w(c, c * 4), z(c)
    est[p + ".proj_out.weight"], est[p + ".proj_out.bias"] = w(c, c, 1, 1), z(c)


def _conv(est, p, cin, cout, rng):
    est[p + ".weight"] = rng.randn(cout, cin, 3, 3).astype(np.float32) / np.sqrt(cin * 9)
    est[p + ".bias"] = np.zeros(cout, np.float32)


def pesos_sdxs(rng):
    e = {}
    _conv(e, "conv_in", 4, C0, rng)
    e["time_embedding.linear_1.weight"], e["time_embedding.linear_1.bias"] = (rng.randn(4 * C0, 320) * 0.05).astype(np.float32), np.zeros(4 * C0, np.float32)
    e["time_embedding.linear_2.weight"], e["time_embedding.linear_2.bias"] = (rng.randn(4 * C0, 4 * C0) * 0.05).astype(np.float32), np.zeros(4 * C0, np.float32)
    _res(e, "down_blocks.0.resnets.0", C0, C0, rng); _conv(e, "down_blocks.0.downsamplers.0.conv", C0, C0, rng)
    _res(e, "down_blocks.1.resnets.0", C0, C1, rng); _att(e, "down_blocks.1.attentions.0", C1, rng)
    _conv(e, "down_blocks.1.downsamplers.0.conv", C1, C1, rng)
    _res(e, "down_blocks.2.resnets.0", C1, C2, rng); _att(e, "down_blocks.2.attentions.0", C2, rng)
    _res(e, "up_blocks.0.resnets.0", C2 * 2, C2, rng); _att(e, "up_blocks.0.attentions.0", C2, rng)
    _res(e, "up_blocks.0.resnets.1", C2 + C1, C2, rng); _att(e, "up_blocks.0.attentions.1", C2, rng)
    _conv(e, "up_blocks.0.upsamplers.0.conv", C2, C2, rng)
    _res(e, "up_blocks.1.resnets.0", C2 + C1, C1, rng); _att(e, "up_blocks.1.attentions.0", C1, rng)
    _res(e, "up_blocks.1.resnets.1", C1 + C0, C1, rng); _att(e, "up_blocks.1.attentions.1", C1, rng)
    _conv(e, "up_blocks.1.upsamplers.0.conv", C1, C1, rng)
    _res(e, "up_blocks.2.resnets.0", C1 + C0, C0, rng); _res(e, "up_blocks.2.resnets.1", C0 + C0, C0, rng)
    e["conv_norm_out.weight"], e["conv_norm_out.bias"] = np.ones(C0, np.float32), np.zeros(C0, np.float32)
    _conv(e, "conv_out", C0, 4, rng)
    return e


class TestConvDesl(unittest.TestCase):
    def test_igual_ao_conv2d(self):
        rng = np.random.RandomState(3)
        for C, O, H, W in ((4, 64, 8, 8), (64, 64, 17, 13), (3, 3, 5, 40), (64, 3, 9, 9)):
            x = rng.randn(C, H, W).astype(np.float32); w = (rng.randn(O, C, 3, 3) * 0.1).astype(np.float32)
            b = rng.randn(O).astype(np.float32)
            for bloco in (7, 64, 4096):
                np.testing.assert_allclose(mc.conv2d_desl(x, w, b, bloco), mc.conv2d(x, w, b, pad=1), atol=2e-5)
            np.testing.assert_allclose(mc.conv2d_desl(x, w), mc.conv2d(x, w, pad=1), atol=2e-5)


class TestSDXS(unittest.TestCase):
    def test_unet_usa_todos_os_pesos_e_forma(self):
        rng = np.random.RandomState(0)
        est = pesos_sdxs(rng)
        usados = set()

        class Rastro(mc.PesosMemoria):
            def __call__(self, n):
                usados.add(n)
                return super().__call__(n)
        x = rng.randn(4, 16, 16).astype(np.float32)
        out = us.unet_sdxs([x], 999, [rng.randn(77, CTX).astype(np.float32)], Rastro(est))[0]
        self.assertEqual(out.shape, (4, 16, 16))
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(sorted(set(est) - usados), [])

    def test_taesd_forma_e_pesos(self):
        rng = np.random.RandomState(1)
        e = {}
        def c(i, cin, cout, bias=True):
            e["decoder.layers.%d.weight" % i] = (rng.randn(cout, cin, 3, 3) / np.sqrt(cin * 9)).astype(np.float32)
            if bias:
                e["decoder.layers.%d.bias" % i] = np.zeros(cout, np.float32)
        c(0, 4, 64)
        idx = 2
        for bi, nb in enumerate((3, 3, 3, 1)):
            for _ in range(nb):
                for k in (0, 2, 4):
                    e["decoder.layers.%d.conv.%d.weight" % (idx, k)] = (rng.randn(64, 64, 3, 3) / 24).astype(np.float32)
                    e["decoder.layers.%d.conv.%d.bias" % (idx, k)] = np.zeros(64, np.float32)
                idx += 1
            if bi < 3:
                idx += 1
            c(idx, 64, 3 if bi == 3 else 64, bias=bi == 3)
            idx += 1
        self.assertEqual(idx, 19)
        y = tn.decodificar(rng.randn(4, 4, 4).astype(np.float32), mc.PesosMemoria(e))
        self.assertEqual(y.shape, (3, 32, 32))
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
