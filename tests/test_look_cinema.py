import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import look_cinema as lc  # noqa: E402
import roteirista as r  # noqa: E402


class TestLook(unittest.TestCase):
    def _img(self):
        rng = np.random.RandomState(1)
        return (rng.rand(96, 96, 3) * 255).astype(np.uint8)

    def test_forma_e_forca_zero(self):
        a = self._img()
        out = np.asarray(lc.aplicar(a))
        self.assertEqual(out.shape, a.shape)
        self.assertTrue(np.array_equal(np.asarray(lc.aplicar(a, forca=0)), a))
        self.assertFalse(np.array_equal(out, a))

    def test_determinista(self):
        a = self._img()
        self.assertTrue(np.array_equal(np.asarray(lc.aplicar(a)), np.asarray(lc.aplicar(a))))

    def test_vinheta_escurece_cantos(self):
        cinza = np.full((128, 128, 3), 128, np.uint8)
        out = np.asarray(lc.aplicar(cinza, graos=0)).astype(float)
        self.assertLess(out[:8, :8].mean(), out[60:68, 60:68].mean())

    def test_prompt_imagem_tem_estilo(self):
        p, aviso = r.prompt_imagem("a red apple", traduzir_ok=False)
        self.assertIn("cinematic", p)
        self.assertEqual(aviso, "")


if __name__ == "__main__":
    unittest.main()
