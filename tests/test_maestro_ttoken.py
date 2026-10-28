"""Testes do MaestroTtoken e dos servidores (sem GPU, sem rede externa)."""
import base64
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from io import BytesIO

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "servidores"))

import maestro_ttoken as mt          # noqa: E402
import servidor_colab as sc          # noqa: E402
import servidor_local as sl          # noqa: E402
from PIL import Image                # noqa: E402


def _frames(prompt, n, beta, seed):
    return [Image.new("RGB", (64, 64), (10 * i % 255, 40, 90)) for i in range(n)]


class Servidor:
    """Sobe o handler do servidor local numa porta livre (como se fosse Colab/celular)."""
    def __init__(self, motor=None, chave="sk-ttoken-2026"):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), sl.fazer_handler(
            motor or sl.MotorLocal(gerador=_frames), chave))
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def fechar(self):
        self.srv.shutdown()
        self.srv.server_close()


def porta_fechada():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return "http://127.0.0.1:%d" % p


class TestLinks(unittest.TestCase):
    def test_normalizar(self):
        self.assertEqual(mt.normalizar_link("abc.trycloudflare.com/"), "https://abc.trycloudflare.com")
        self.assertEqual(mt.normalizar_link("127.0.0.1:8000"), "http://127.0.0.1:8000")
        self.assertEqual(mt.normalizar_link("  "), "")
        self.assertEqual(mt.normalizar_link("https://x.app/"), "https://x.app")

    def test_salvar_e_carregar(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(mt.carregar_links(d)["chave"], mt.CHAVE_PADRAO)   # sem arquivo
            mt.salvar_links(d, "x.trycloudflare.com", "127.0.0.1:8000", "minha")
            dados = mt.carregar_links(d)
            self.assertEqual(dados, {"colab": "https://x.trycloudflare.com",
                                     "local": "http://127.0.0.1:8000", "chave": "minha"})
            self.assertTrue(os.path.exists(os.path.join(d, "config_links.json")))
            with open(os.path.join(d, "config_links.json"), "w") as f:
                f.write("{quebrado")
            self.assertEqual(mt.carregar_links(d)["colab"], "")                # arquivo ruim

class TestCascata(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.mkdtemp()
        self.msgs = []

    def maestro(self, **kw):
        return mt.MaestroTtoken(pasta=self.pasta, **kw)

    def test_local_primeiro(self):
        local, colab = Servidor(), Servidor()
        try:
            r = self.maestro(link_colab=colab.url, link_python_local=local.url).gerar_video(
                "carro neon na chuva", self.msgs.append)
        finally:
            local.fechar(); colab.fechar()
        self.assertEqual(r["origem"], "local")
        self.assertTrue(os.path.exists(r["arquivo"]) and r["arquivo"].endswith(".gif"))
        self.assertEqual(Image.open(r["arquivo"]).format, "GIF")
        self.assertTrue(any("Tentando gerar no Python local" in m for m in self.msgs))
        self.assertFalse(any("Colab" in m for m in self.msgs))

    def test_local_offline_cai_no_colab(self):
        colab = Servidor()
        try:
            r = self.maestro(link_colab=colab.url, link_python_local=porta_fechada()).gerar_video(
                "montanha ao amanhecer", self.msgs.append)
        finally:
            colab.fechar()
        self.assertEqual(r["origem"], "colab")
        txt = "\n".join(self.msgs)
        self.assertIn("Python local offline", txt)
        self.assertIn("Video gerado no Colab", txt)

    def test_chave_errada_e_fallback_apis(self):
        colab = Servidor(chave="outra-chave")
        chamadas = []

        def apis(prompt, cb):
            chamadas.append(prompt)
            cb("apis ok")
            return {"url": "https://exemplo/v.mp4", "texto": "legenda"}
        try:
            r = self.maestro(link_colab=colab.url, fallback_apis=apis).gerar_video(
                "dragao sobre o castelo", self.msgs.append)
        finally:
            colab.fechar()
        self.assertEqual(r["origem"], "apis")
        self.assertEqual(r["url"], "https://exemplo/v.mp4")
        self.assertEqual(chamadas, ["dragao sobre o castelo"])
        self.assertTrue(any("recusou a chave" in m for m in self.msgs))

    def test_tudo_falha(self):
        r = self.maestro(link_colab=porta_fechada()).gerar_video("qualquer coisa", self.msgs.append)
        self.assertIsNone(r)
        self.assertIn("❌ Todas as tentativas falharam.", self.msgs)

    def test_servidor_sem_modelo_e_offline(self):
        class Sem(sl.MotorLocal):
            def disponivel(self):
                return False
        srv = Servidor(motor=Sem())
        try:
            r = self.maestro(link_python_local=srv.url).gerar_video("teste de offline", self.msgs.append)
        finally:
            srv.fechar()
        self.assertIsNone(r)
        self.assertTrue(any("offline" in m for m in self.msgs))

    def test_prompt_vazio_nao_chama_nada(self):
        self.assertIsNone(self.maestro().gerar_video("  ", self.msgs.append))
        self.assertEqual(len(self.msgs), 1)

    def test_thread_e_erro_na_api_nao_mata(self):
        def apis(prompt, cb):
            raise RuntimeError("boom")
        fim = threading.Event()
        res = []
        agenda = lambda f, *a: f(*a)
        m = self.maestro(fallback_apis=apis)
        m.gerar_em_thread("ideia qualquer", self.msgs.append,
                          lambda r: (res.append(r), fim.set()), ui=agenda)
        self.assertTrue(fim.wait(10))
        self.assertEqual(res, [None])
        self.assertTrue(any("APIs gratuitas falharam" in x for x in self.msgs))

    def test_resposta_mp4_base64_e_url(self):
        m = self.maestro()
        r = m._guardar({"status": "sucesso", "mp4": base64.b64encode(b"abc").decode()})
        self.assertTrue(r["arquivo"].endswith(".mp4"))
        self.assertEqual(m._guardar({"video_url": "https://x/y.mp4"})["url"], "https://x/y.mp4")
        self.assertIsNone(m._guardar({"status": "erro"}))
        self.assertIsNone(m._guardar({"status": "sucesso"}))


class TestServidores(unittest.TestCase):
    def test_ssim_sem_estouro_de_uint8(self):
        a = Image.new("L", (32, 32), 3)
        b = Image.new("L", (32, 32), 250)          # 3 - 250 estouraria em uint8
        self.assertAlmostEqual(sc.ssim_global(a, a), 1.0, places=4)
        v = sc.ssim_global(a, b)
        self.assertTrue(-1.0 <= v < 0.5, v)

    def test_motor_colab_com_gerador_falso(self):
        m = sc.MotorVideo(gerador=_frames)
        r = m.gerar("um gato astronauta", frames=6, beta=0.05, seed=1)
        self.assertEqual(r["status"], "sucesso")
        self.assertEqual(r["frames"], 6)
        gif = Image.open(BytesIO(base64.b64decode(r["gif"])))
        self.assertEqual(gif.format, "GIF")
        self.assertEqual(m.historico, [r["ssim_medio"]])
        with self.assertRaises(ValueError):
            m.gerar("a")
        self.assertEqual(m.gerar("duas palavras", frames=999)["frames"], sc.MAX_FRAMES)

    def test_coerencia_maior_que_independente(self):
        import numpy as np
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        quase = np.clip(base.astype(np.int16) + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8)
        outra = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        self.assertGreater(sc.ssim_global(base, quase), sc.ssim_global(base, outra))

    @unittest.skipUnless(__import__("importlib").util.find_spec("fastapi"), "fastapi nao instalado aqui")
    def test_api_colab_chave_e_rotas(self):
        from fastapi.testclient import TestClient
        c = TestClient(sc.criar_app(sc.MotorVideo(gerador=_frames), chave="k"))
        self.assertEqual(c.get("/saude").json()["status"], "online")
        self.assertEqual(c.post("/gerar", json={"prompt": "uma ideia"}).status_code, 401)
        r = c.post("/gerar", json={"prompt": "uma ideia", "frames": 3}, headers={"x-api-key": "k"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["frames"], 3)
        self.assertEqual(c.post("/gerar", json={"prompt": "a"}, headers={"x-api-key": "k"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
