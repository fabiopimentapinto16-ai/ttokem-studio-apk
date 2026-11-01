import os, sys, unittest, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import pipeline_tinysd as p


class TestAgendador(unittest.TestCase):
    def test_timesteps_4(self):
        a = p.AgendadorDPM(); a.definir_passos(4)
        self.assertEqual(a.timesteps.tolist(), [999, 749, 500, 250])
        self.assertEqual(a.sigmas[-1], 0.0)

    def test_previsao_perfeita_recupera_x0(self):
        # se o modelo previsse o ruido exato de x0 constante, cada passo mantem x0 e o final = x0
        rng = np.random.RandomState(0)
        x0, ruido = rng.randn(4, 8, 8).astype("f"), rng.randn(4, 8, 8).astype("f")
        a = p.AgendadorDPM(); a.definir_passos(4)
        s0 = a.sigmas[0]; al, sg = a._as(s0)
        x = al * x0 + sg * ruido
        for _ in range(4):
            i = a.i; al, sg = a._as(a.sigmas[i])
            eps = (x - al * x0) / sg
            x = a.passo(eps, x)
        self.assertLess(float(np.abs(x - x0).max()), 1e-3)

    def test_passos_variados(self):
        for n in (1, 2, 4, 8):
            a = p.AgendadorDPM(); a.definir_passos(n)
            x = np.zeros((4, 4, 4), "f")
            for _ in range(n):
                x = a.passo(np.ones_like(x), x)
            self.assertTrue(np.isfinite(x).all())


class TestPNG(unittest.TestCase):
    def test_png(self):
        import tempfile
        img = (np.arange(8 * 8 * 3) % 256).astype("uint8").reshape(8, 8, 3)
        f = os.path.join(tempfile.mkdtemp(), "a.png"); p.salvar_png(img, f)
        b = open(f, "rb").read(); self.assertEqual(b[:8], b"\x89PNG\r\n\x1a\n")


TOK = "/tmp/claude-0/-home-claude/cd6353e4-5c66-59d8-961d-88ee6b5950d2/scratchpad/tok/tokenizer.json"


@unittest.skipUnless(os.path.exists(TOK), "sem tokenizer.json")
class TestTok(unittest.TestCase):
    def test_ids_conhecidos(self):
        t = p.TokenizadorCLIP(TOK)
        ids = t.codificar("a red apple on a table")
        self.assertEqual(ids[:8], [49406, 320, 736, 3055, 525, 320, 2175, 49407])
        self.assertEqual(len(ids), 77)
        self.assertEqual(t.codificar("")[:3], [49406, 49407, 49407])
        self.assertEqual(t.codificar("x " * 200)[-1], 49407)


if __name__ == "__main__":
    unittest.main()
