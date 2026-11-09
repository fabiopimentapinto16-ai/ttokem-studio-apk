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

    def test_detecta_idioma(self):
        self.assertEqual(r.detectar_idioma("uma casa no campo com um cachorro"), "pt")
        self.assertEqual(r.detectar_idioma("a house in the countryside with a dog"), "en")
        self.assertEqual(r.detectar_idioma("una casa en el campo con un perro"), "es")
        self.assertEqual(r.detectar_idioma("maison à la campagne avec un chien"), "fr")
        self.assertEqual(r.detectar_idioma("家の庭"), "xx")
        self.assertEqual(r.locale_tts("pt"), "pt-BR")
        self.assertEqual(r.locale_tts("ja"), "ja")

    def test_qualquer_idioma_traduz_prompt_e_fala_no_idioma(self):
        antes = r.traduzir_en
        r.traduzir_en = lambda t, timeout=8: ("a house in the countryside", "pt")
        try:
            p = r.montar_cenas("uma casa no campo", usar_ia=False)
        finally:
            r.traduzir_en = antes
        self.assertIn("house in the countryside", p["cenas"][0]["prompt"])
        self.assertEqual(p["idioma"], "pt")
        self.assertEqual(p["cenas"][0]["narracao"], "Tudo comeca aqui.")

    def test_idioma_de_saida_escolhido(self):
        antes = r.traduzir
        r.traduzir = lambda t, destino="en", timeout=8: (t + "@" + destino, "en")
        try:
            p = r.montar_cenas("a red apple", idioma="Português do Brasil", usar_ia=False)
            self.assertEqual(p["idioma"], "pt")
            self.assertEqual(p["cenas"][0]["narracao"], "Tudo comeca aqui.")        # falas prontas, sem traduzir
            j = r.montar_cenas("a red apple", idioma="日本語", usar_ia=False)
            self.assertEqual(j["idioma"], "ja")
            self.assertTrue(j["cenas"][1]["narracao"].endswith("@ja"))               # sem falas prontas: traduz
        finally:
            r.traduzir = antes

    def test_codigo_idioma(self):
        self.assertEqual(r.codigo_idioma("Português do Brasil"), "pt")
        self.assertEqual(r.codigo_idioma("English"), "en")
        self.assertEqual(r.codigo_idioma("Español"), "es")
        self.assertEqual(r.codigo_idioma("Japones"), "ja")
        self.assertEqual(r.codigo_idioma("日本語"), "ja")
        self.assertEqual(r.codigo_idioma("Swahili"), "sw")
        self.assertEqual(r.codigo_idioma("xyzzy", padrao="en"), "en")

    def test_sem_tradutor_avisa(self):
        antes = r.traduzir_en
        r.traduzir_en = lambda t, timeout=8: None
        try:
            p = r.montar_cenas("uma casa no campo", usar_ia=False)
        finally:
            r.traduzir_en = antes
        self.assertIn("aviso", p)


if __name__ == "__main__":
    unittest.main()
