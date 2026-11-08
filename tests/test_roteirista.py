import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import roteirista as r  # noqa: E402


class _LLM:
    def __init__(self, resp=None, erro=False):
        self.resp, self.erro = resp, erro

    def chat(self, msgs, max_tokens=0):
        if self.erro:
            raise RuntimeError("sem rede")
        return self.resp, "fake/x"


class TestRoteirista(unittest.TestCase):
    def test_offline_gera_planos_diferentes(self):
        p = r.montar_cenas("a red apple on a table", usar_ia=False)
        self.assertEqual(p["origem"], "local")
        self.assertEqual(len(p["cenas"]), 4)
        self.assertEqual(len({c["prompt"] for c in p["cenas"]}), 4)
        self.assertTrue(all("cinematic" in c["prompt"] for c in p["cenas"]))

    def test_varias_linhas_viram_cenas(self):
        p = r.montar_cenas("one\ntwo cats\nthree dogs")
        self.assertEqual(p["origem"], "suas linhas")
        self.assertEqual(len(p["cenas"]), 3)

    def test_ia_valida_e_cai_para_local(self):
        ok = '{"titulo":"T","cenas":[{"prompt":"A lighthouse in waves at dusk","narracao":"a","legenda":"b"},' \
             '{"prompt":"Close up of a glowing lamp behind glass","narracao":"c","legenda":"d"}]}'
        self.assertTrue(r.montar_cenas("farol", _LLM(ok))["origem"].startswith("IA"))
        self.assertEqual(r.montar_cenas("farol", _LLM(erro=True))["origem"], "local")
        self.assertEqual(r.montar_cenas("farol", _LLM("lixo sem json"))["origem"], "local")


if __name__ == "__main__":
    unittest.main()
