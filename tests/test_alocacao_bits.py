import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import alocacao_bits as a

BPP = [3.0, 4.0, 6.0, 8.0, 16.0]


def err_modelo(ef, n=5):
    return [ef * 2.0 ** (-2 * b) * 10 for b in BPP]       # ~6 dB por bit


class T(unittest.TestCase):
    def test_casco_tira_dominadas(self):
        h = a.casco([3, 4, 5, 6], [10, 4, 5, 1])          # opcao 2 (5 bits, erro 5) e pior que a de 4 bits
        self.assertNotIn(2, h)

    def test_orcamento_respeitado_e_tensor_sensivel_ganha_bits(self):
        ef = [100.0, 1.0, 1.0, 1.0]; numel = [1000] * 4
        err = [err_modelo(e) for e in ef]
        esc, usado, e = a.alocar(BPP, err, numel, orcamento_bits=sum(numel) * 5.0)
        self.assertLessEqual(usado, sum(numel) * 5.0 + 1e-9)
        self.assertGreater(BPP[esc[0]], BPP[esc[1]])      # o sensivel recebe mais bits

    def test_mais_orcamento_menos_erro(self):
        r = np.random.RandomState(0); ef = r.rand(20) * 10; numel = (r.rand(20) * 1000 + 100).astype(int)
        err = [err_modelo(e) for e in ef]
        es = [a.alocar(BPP, err, numel, numel.sum() * b)[2] for b in (3.5, 5, 7, 10)]
        self.assertEqual(es, sorted(es, reverse=True))

    def test_orcamento_minimo(self):
        esc, usado, _ = a.alocar(BPP, [err_modelo(1.0)] * 3, [10, 10, 10], 1.0)   # nem o minimo cabe
        self.assertEqual(list(esc), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
