import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import analise_bits as a


class T(unittest.TestCase):
    def setUp(self):
        r = np.random.RandomState(0)
        self.w = {"w": (r.laplace(size=(64, 128)) * 0.02).astype("f")}

    def test_hadamard_ortogonal(self):
        H = a.hadamard(32); self.assertTrue(np.allclose(H @ H.T, np.eye(32), atol=1e-5))

    def test_entropia_nao_passa_de_base(self):
        L = {(m, b): (bpp, snr) for m, b, g, bpp, snr, x in a.varrer(self.w, [(4, 32), (8, 32)])}
        self.assertLessEqual(L[("entropia", 4)][0], L[("base", 4)][0] + 1e-9)
        self.assertAlmostEqual(L[("entropia", 4)][1], L[("base", 4)][1], 4)   # sem perda: mesmo SNR

    def test_mais_bits_mais_snr(self):
        s = [r[4] for r in a.varrer(self.w, [(3, 32), (4, 32), (6, 32)], ("base",))]
        self.assertEqual(s, sorted(s))


if __name__ == "__main__":
    unittest.main()
