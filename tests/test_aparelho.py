import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aparelho as ap  # noqa: E402


def novo(gflops, ram_livre=2000, ram_total=3000, pasta=None, tempo=None):
    pasta = pasta or tempfile.mkdtemp()
    a = ap.Aparelho(pasta, medidor=lambda: gflops, relogio=(lambda: tempo[0]) if tempo else __import__("time").time)
    ap.ram_mb, orig = (lambda: (ram_total, ram_livre)), ap.ram_mb
    try:
        a.medir_se_preciso()
    finally:
        ap.ram_mb = orig
    return a, pasta


class TestAparelho(unittest.TestCase):
    def test_celular_fraco_com_pydroid(self):
        a, _ = novo(2.3)
        p = a.decidir("max", servidor_ativo=True)
        self.assertEqual(p["motor"], "pydroid")
        self.assertEqual(p["lado"], 512)                         # 95 s no Pydroid cabe no limite

    def test_celular_fraco_sem_pydroid_fica_256(self):
        a, _ = novo(2.3)
        p = a.decidir("max", servidor_ativo=False)
        self.assertEqual((p["motor"], p["lado"]), ("app", 256))

    def test_celular_forte_usa_app_512_e_cache(self):
        a, _ = novo(60.0, ram_livre=6000, ram_total=8000)
        p = a.decidir("max", servidor_ativo=True)
        self.assertEqual((p["motor"], p["lado"]), ("app", 512))     # nao precisa do Pydroid
        self.assertGreater(p["cache_mb"], 0)

    def test_qualidade_rapida_fica_256(self):
        a, _ = novo(60.0, ram_livre=6000, ram_total=8000)
        self.assertEqual(a.decidir("rapida")["lado"], 256)

    def test_aprende_com_tempo_real_e_falta_de_memoria(self):
        a, _ = novo(60.0, ram_livre=6000, ram_total=8000)
        for _ in range(3):
            a.registrar("app", 512, 600)                          # na pratica foi lento: 10 min
        self.assertEqual(a.decidir("max")["lado"], 256)
        b, _ = novo(60.0, ram_livre=6000, ram_total=8000)
        b.registrar("app", 512, 0, falta_memoria=True)
        self.assertEqual(b.decidir("max")["lado"], 256)

    def test_aparelho_novo_mede_de_novo_e_esquece(self):
        a, pasta = novo(2.3, ram_total=3000)
        a.registrar("app", 256, 500)
        b = ap.Aparelho(pasta, medidor=lambda: 60.0)
        ap.ram_mb, orig = (lambda: (8000, 6000)), ap.ram_mb
        try:
            self.assertTrue(b.medir_se_preciso())                   # RAM diferente = outro celular
        finally:
            ap.ram_mb = orig
        self.assertEqual(b.d["hist"], {})
        self.assertGreater(b.d["gflops"], 50)

    def test_mesmo_aparelho_nao_mede_de_novo(self):
        a, pasta = novo(2.3)
        b = ap.Aparelho(pasta, medidor=lambda: 99.0)
        ap.ram_mb, orig = (lambda: (3000, 2000)), ap.ram_mb
        try:
            self.assertFalse(b.medir_se_preciso())
        finally:
            ap.ram_mb = orig


if __name__ == "__main__":
    unittest.main()
