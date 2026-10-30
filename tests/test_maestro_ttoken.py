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
    def __init__(self, motor=None, chave="k-teste"):
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

    def test_link_e_chave_juntos(self):
        self.assertEqual(mt.separar_link_chave("abc.trycloudflare.com/#k123"),
                         ("https://abc.trycloudflare.com", "k123"))
        self.assertEqual(mt.separar_link_chave("http://127.0.0.1:8000#z", local=True),
                         ("http://127.0.0.1:8000", "z"))
        with tempfile.TemporaryDirectory() as d:
            mt.salvar_links(d, "https://x.app#segredo", "")
            self.assertEqual(mt.carregar_links(d)["chave"], "segredo")

    def test_servidor_sem_chave_embutida(self):
        self.assertGreaterEqual(len(sc.CHAVE), 8)        # aleatoria, nao embutida
        self.assertNotEqual(sc.CHAVE, "sk-ttoken-2026")
        self.assertNotEqual(sl.CHAVE, "sk-ttoken-2026")

    def test_servidor_em_thread_liga_e_desliga(self):
        srv = sl.iniciar_em_thread("k", porta=0, motor=sl.MotorLocal(gerador=_frames))
        porta = srv.server_address[1]
        import urllib.request
        self.assertEqual(urllib.request.urlopen("http://127.0.0.1:%d/saude" % porta, timeout=5).status, 200)
        srv.shutdown(); srv.server_close()
        with self.assertRaises(OSError):                   # porta ocupada -> OSError
            a = sl.iniciar_em_thread("k", porta=0, motor=sl.MotorLocal(gerador=_frames))
            try:
                sl.iniciar_em_thread("k", porta=a.server_address[1])
            finally:
                a.shutdown(); a.server_close()

    def test_salvar_e_carregar(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(mt.carregar_links(d)["chave"], "")   # sem chave embutida
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
        kw.setdefault("chave_api", "k-teste")
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

    def test_tudo_falha_cai_no_motor_matematico(self):
        r = self.maestro().gerar_video("raposa na neve", self.msgs.append)
        self.assertEqual(r["origem"], "matematico")
        self.assertTrue(r["arquivo"].endswith(".gif") and os.path.exists(r["arquivo"]))
        self.assertTrue(os.path.exists(r["imagem"]) and r["imagem"].endswith(".png"))
        self.assertEqual(Image.open(r["imagem"]).size, (256, 256))

    def test_modo_offline_nao_toca_a_rede(self):
        class Proibido:
            def get(self, *a, **k): raise AssertionError("usou a rede")
            post = get
        chamou = []
        m = self.maestro(link_colab="https://x.trycloudflare.com", link_python_local="http://127.0.0.1:9",
                         http=Proibido(), offline=True, fallback_apis=lambda *a: chamou.append(1))
        r = m.gerar_video("mar azul", self.msgs.append)
        self.assertEqual(r["origem"], "matematico")
        self.assertEqual(chamou, [])
        self.assertTrue(any("offline" in x.lower() for x in self.msgs))

    def test_gif_puro_sem_pillow_abre_e_anima(self):
        import motor_matematico as mm
        from PIL import ImageChops
        g = Image.open(__import__("io").BytesIO(mm.gerar_gif_bytes("fogo", 64, 8, 10)))
        self.assertEqual((g.format, g.n_frames, g.size), ("GIF", 8, (64, 64)))
        g.seek(0); a = g.convert("RGB").copy(); g.seek(4); b = g.convert("RGB").copy()
        self.assertIsNotNone(ImageChops.difference(a, b).getbbox())

    def test_motor_matematico_deterministico_e_png_valido(self):
        import motor_matematico as mm
        a, b = mm.gerar_bytes("mar azul", 64), mm.gerar_bytes("mar azul", 64)
        self.assertEqual(a, b)
        self.assertNotEqual(a, mm.gerar_bytes("fogo", 64))
        self.assertEqual(a[:8], b"\x89PNG\r\n\x1a\n")

    def test_tudo_falha(self):
        import motor_matematico
        antes = motor_matematico.gerar
        motor_matematico.gerar = lambda *a, **k: (_ for _ in ()).throw(OSError("disco cheio"))
        try:
            r = self.maestro(link_colab=porta_fechada()).gerar_video("qualquer coisa",
                                                                       self.msgs.append)
        finally:
            motor_matematico.gerar = antes
        self.assertIsNone(r)
        self.assertTrue(any("Todas as tentativas falharam" in m for m in self.msgs))

    def test_servidor_sem_modelo_e_offline(self):
        class Sem(sl.MotorLocal):
            def disponivel(self):
                return False
        srv = Servidor(motor=Sem())
        try:
            r = self.maestro(link_python_local=srv.url).gerar_video("teste de offline", self.msgs.append)
        finally:
            srv.fechar()
        self.assertEqual(r["origem"], "matematico")      # servidor offline -> motor matematico
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
        self.assertEqual(res[0]["origem"], "matematico")   # APIs falharam, nao morreu
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


class TestAnimador(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.mkdtemp()
        self.png = os.path.join(self.pasta, "ttoken_x.png")
        im = Image.new("RGB", (400, 300), (10, 20, 30))
        for x in range(0, 400, 40):
            for y in range(300):
                im.putpixel((x, y), (250, 120, 0))
        im.save(self.png)

    def test_todos_os_estilos_geram_gif_com_movimento(self):
        import animador
        from PIL import ImageChops
        for est in animador.ESTILOS:
            out = animador.animar(self.png, os.path.join(self.pasta, est + ".gif"),
                                  duracao=2, fps=6, lado=160, estilo=est)
            g = Image.open(out)
            self.assertEqual((g.format, g.n_frames, g.size), ("GIF", 12, (160, 160)))
            g.seek(0); a = g.convert("RGB").copy(); g.seek(11); b = g.convert("RGB").copy()
            self.assertIsNotNone(ImageChops.difference(a, b).getbbox(), est)

    def test_imagem_invalida_e_estilo_repetivel(self):
        import animador
        with self.assertRaises(ValueError):
            animador.animar(b"nao e imagem", os.path.join(self.pasta, "z.gif"))
        self.assertEqual(animador.escolher_estilo("abc"), animador.escolher_estilo("abc"))

    def test_maestro_anima_quando_servidor_devolve_so_png(self):
        m = mt.MaestroTtoken(pasta=self.pasta)
        msgs = []
        r = m.animar_imagem({"arquivo": self.png, "url": ""}, "gato", msgs.append)
        self.assertTrue(r["arquivo"].endswith(".gif") and os.path.exists(r["arquivo"]))
        self.assertEqual(r["imagem"], self.png)
        gif = {"arquivo": r["arquivo"]}                   # gif/mp4 passam sem mexer
        self.assertIs(m.animar_imagem(gif, "gato", msgs.append), gif)
        self.assertEqual(m.animar_imagem(None), None)


if __name__ == "__main__":
    unittest.main()
