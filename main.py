"""Ttoken Studio — app Kivy (visual neon/vidro).

Fluxo: ideia -> backend (Ttokem + maestro de IAs + maestro de video) -> legenda
e video prontos. O app NUNCA posta sozinho: o usuario da o ultimo toque.
"""
import base64
import json
import os
import re
import threading
import time
import webbrowser

import requests
from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import platform

from maestro import Maestro          # motor embutido: funciona sem servidor
from audio import SemTranscricao, transcrever
import trending
from pipeline import produzir
from video import VideoMaestro

SharedStorage = ShareSheet = Chooser = autoclass = None
pedir_permissao = Permission = None
if platform == "android":
    try:
        from androidstorage4kivy import Chooser, ShareSheet, SharedStorage
        from jnius import autoclass
    except Exception:                # biblioteca ausente: cai no fallback por link
        SharedStorage = ShareSheet = Chooser = autoclass = None
    try:
        from android.permissions import Permission, request_permissions as pedir_permissao
    except Exception:
        pedir_permissao = Permission = None

API = "https://ttokem-backend-8ba2a.containers.snapdeploy.app"
POLL_EVERY = 4          # segundos entre consultas
MAX_WAIT = 1200          # desiste depois de 8 minutos
WAKE_WAIT = 180         # tempo maximo esperando o servidor acordar
UA = ("Mozilla/5.0 (Linux; Android 13; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Mobile Safari/537.36")

# chaves que o usuario pode guardar so no celular (modo sem servidor)
CHAVES = [("groq", "GROQ (texto)", "GROQ_API_KEY"),
          ("openrouter", "OPENROUTER (texto, varios modelos)", "OPENROUTER_API_KEY"),
          ("gemini", "GEMINI (texto)", "GEMINI_API_KEY"),
          ("hf", "HUGGING FACE (texto)", "HF_TOKEN"),
          ("magichour", "MAGIC HOUR (video)", "MAGIC_HOUR_API_KEY"),
          ("fal", "FAL.AI (video)", "FAL_KEY"),
          ("youtube", "YOUTUBE (ver os mais assistidos)", "YOUTUBE_API_KEY"),
          ("pexels", "PEXELS (banco de videos gratis, plano B)", "PEXELS_API_KEY"),
          ("build", "SENHA DE BUILD (compilar pelo Git, igual a BUILD_SECRET)",
           "TTOKEN_BUILD_SECRET")]
LINKS = {"groq": "https://console.groq.com/keys",
         "openrouter": "https://openrouter.ai/keys",
         "gemini": "https://aistudio.google.com/apikey",
         "hf": "https://huggingface.co/settings/tokens",
         "magichour": "https://magichour.ai/developer",
         "fal": "https://fal.ai/dashboard/keys",
         "youtube": "https://console.cloud.google.com/apis/library/youtube.googleapis.com",
         "pexels": "https://www.pexels.com/api/"}
TEXTO_KEYS = ("groq", "openrouter", "gemini", "hf")

# redes: abrimos o app que o usuario ja tem logado (sem login/OAuth no nosso app)
SOCIAIS = [("Instagram", "com.instagram.android"),
           ("TikTok", "com.zhiliaoapp.musically"),
           ("YouTube", "com.google.android.youtube"),
           ("WhatsApp", "com.whatsapp"),
           ("Facebook", "com.facebook.katana"),
           ("X", "com.twitter.android"),
           ("Telegram", "org.telegram.messenger"),
           ("Outros", None)]
PRECISA_VIDEO = {"com.instagram.android", "com.zhiliaoapp.musically",
                 "com.google.android.youtube"}

# ------------------------------------------------------------------ paleta
BG_TOP = (0.030, 0.030, 0.085, 1)
BG_BOT = (0.020, 0.075, 0.115, 1)
CYAN = (0.00, 0.90, 0.76, 1)
VIOLET = (0.55, 0.36, 0.96, 1)
MAGENTA = (1.00, 0.24, 0.65, 1)
TEXT = (0.93, 0.95, 1.00, 1)
MUTED = (0.56, 0.61, 0.74, 1)
INK = (0.02, 0.03, 0.06, 1)


def alpha(c, a):
    return (c[0], c[1], c[2], a)


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(255 * v) for v in c[:3])


def gradient_texture(c1, c2, horizontal=False, steps=96):
    """Textura de degrade. Vertical: c1 em cima, c2 embaixo."""
    size = (steps, 1) if horizontal else (1, steps)
    tex = Texture.create(size=size, colorfmt="rgba")
    buf = bytearray()
    for i in range(steps):
        t = i / (steps - 1)
        if not horizontal:
            t = 1 - t               # linha 0 da textura fica embaixo
        buf += bytes(int(255 * (c1[k] * (1 - t) + c2[k] * t)) for k in range(4))
    tex.blit_buffer(bytes(buf), colorfmt="rgba", bufferfmt="ubyte")
    tex.mag_filter = "linear"
    tex.min_filter = "linear"
    return tex


def tlabel(text, sz=14, color=TEXT, bold=False, align="left", **kw):
    lb = Label(text=text, font_size=sp(sz), color=color, bold=bold, markup=True,
               halign=align, valign="middle", size_hint_y=None, **kw)
    lb.bind(width=lambda i, w: setattr(i, "text_size", (w, None)),
            texture_size=lambda i, ts: setattr(i, "height", ts[1] + dp(4)))
    return lb


# ----------------------------------------------------------------- widgets
class Backdrop(Widget):
    """Fundo em degrade com brilhos (aurora) ciano e violeta."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._tex = gradient_texture(BG_TOP, BG_BOT)
        self._glows = []
        with self.canvas:
            Color(1, 1, 1, 1)
            self._rect = Rectangle(texture=self._tex)
            for col, fx, fy, base in ((CYAN, 0.12, 0.86, 170),
                                      (VIOLET, 0.92, 0.18, 210),
                                      (MAGENTA, 0.78, 0.62, 90)):
                layers = []
                for k in range(7):
                    Color(*alpha(col, 0.028))
                    layers.append(Ellipse())
                self._glows.append((layers, fx, fy, base))
        self.bind(pos=self._u, size=self._u)

    def _u(self, *_):
        self._rect.pos, self._rect.size = self.pos, self.size
        for layers, fx, fy, base in self._glows:
            cx, cy = self.x + self.width * fx, self.y + self.height * fy
            for k, el in enumerate(layers):
                r = dp(base) * (1 - k * 0.12)
                el.pos, el.size = (cx - r, cy - r), (2 * r, 2 * r)


class Glass(BoxLayout):
    """Cartao de vidro: fundo translucido + borda neon fina."""

    def __init__(self, border=alpha(CYAN, 0.40), tint=(1, 1, 1, 0.05),
                 radius=20, **kw):
        kw.setdefault("orientation", "vertical")
        kw.setdefault("size_hint_y", None)
        kw.setdefault("padding", dp(16))
        kw.setdefault("spacing", dp(10))
        super().__init__(**kw)
        self._r = dp(radius)
        with self.canvas.before:
            Color(*tint)
            self._bg = RoundedRectangle(radius=[self._r])
            Color(*border)
            self._ln = Line(rounded_rectangle=(0, 0, 10, 10, self._r), width=dp(1))
        self.bind(pos=self._u, size=self._u,
                  minimum_height=lambda i, h: setattr(i, "height", h))

    def _u(self, *_):
        self._bg.pos, self._bg.size = self.pos, self.size
        self._ln.rounded_rectangle = (self.x, self.y, self.width, self.height, self._r)


class NeonButton(ButtonBehavior, Label):
    """Botao principal: degrade ciano->violeta com brilho."""

    def __init__(self, text, c1=CYAN, c2=VIOLET, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(56))
        super().__init__(text=text, bold=True, font_size=sp(15), color=INK, **kw)
        self._r = dp(16)
        with self.canvas.before:
            Color(*alpha(c1, 0.16))
            self._glow = RoundedRectangle(radius=[dp(22)])
            self._tint = Color(1, 1, 1, 1)
            self._bg = RoundedRectangle(texture=gradient_texture(c1, c2, True),
                                        radius=[self._r])
        self.bind(pos=self._u, size=self._u, state=self._s, disabled=self._s)

    def _u(self, *_):
        g = dp(5)
        self._glow.pos = (self.x - g, self.y - g)
        self._glow.size = (self.width + 2 * g, self.height + 2 * g)
        self._bg.pos, self._bg.size = self.pos, self.size

    def _s(self, *_):
        self._tint.a = 0.35 if self.disabled else (0.78 if self.state == "down" else 1)


class Chip(ButtonBehavior, Label):
    """Pastilha: usada como seletor (active) e como botao secundario."""
    active = BooleanProperty(False)

    def __init__(self, text, height=36, **kw):
        super().__init__(text=text, bold=True, font_size=sp(13),
                         size_hint_y=None, height=dp(height), **kw)
        self._r = dp(height / 2)
        with self.canvas.before:
            self._fc = Color(1, 1, 1, 0.06)
            self._bg = RoundedRectangle(radius=[self._r])
            self._lc = Color(1, 1, 1, 0.14)
            self._ln = Line(rounded_rectangle=(0, 0, 10, 10, self._r), width=dp(1))
        self.bind(pos=self._u, size=self._u, active=self._u, state=self._u)

    def _u(self, *_):
        self._bg.pos, self._bg.size = self.pos, self.size
        self._ln.rounded_rectangle = (self.x, self.y, self.width, self.height, self._r)
        if self.active:
            self._fc.rgba, self._lc.rgba, self.color = alpha(CYAN, 0.18), CYAN, CYAN
        else:
            self._fc.rgba = (1, 1, 1, 0.12 if self.state == "down" else 0.06)
            self._lc.rgba, self.color = (1, 1, 1, 0.14), TEXT


class NeonBar(Widget):
    """Barra de progresso com degrade; `frac` vai de 0 a 1."""
    frac = NumericProperty(0)

    def __init__(self, **kw):
        kw.setdefault("size_hint_y", None)
        kw.setdefault("height", dp(6))
        super().__init__(**kw)
        with self.canvas:
            Color(1, 1, 1, 0.09)
            self._track = RoundedRectangle(radius=[dp(3)])
            Color(1, 1, 1, 1)
            self._fill = RoundedRectangle(texture=gradient_texture(CYAN, VIOLET, True),
                                          radius=[dp(3)])
        self.bind(pos=self._u, size=self._u, frac=self._u)

    def _u(self, *_):
        self._track.pos, self._track.size = self.pos, self.size
        self._fill.pos = self.pos
        self._fill.size = (max(dp(6), self.width * self.frac), self.height)


# --------------------------------------------------------------------- app
PLATAFORMAS = [("Reels", "instagram reels"), ("TikTok", "tiktok"),
               ("Shorts", "youtube shorts")]
PASSOS = ["01  TTOKEM", "02  MAESTRO IA", "03  VIDEO"]


class TtokenApp(App):
    texto = ""
    video_url = ""
    plataforma = "instagram reels"

    def build(self):
        self.title = "Ttoken Studio"
        Window.clearcolor = BG_TOP
        Window.softinput_mode = "below_target"

        root = FloatLayout()
        root.add_widget(Backdrop(size_hint=(1, 1)))
        scroll = ScrollView(do_scroll_x=False, bar_width=0, size_hint=(1, 1))
        col = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(16),
                        padding=[dp(18), dp(30), dp(18), dp(30)])
        col.bind(minimum_height=col.setter("height"))

        col.add_widget(tlabel("[b]TTOKEN[/b] [color=%s]STUDIO[/color]" % hexc(CYAN),
                              32, align="center"))
        col.add_widget(tlabel("IDEIA  >  VIDEO  >  LEGENDA, EM UM TOQUE", 11, MUTED,
                              align="center"))

        # abas: VIDEO | CHAT | CODIGO
        abas = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        self.abas = {}
        for chave, nome in (("video", "VIDEO"), ("chat", "CHAT"), ("code", "CODIGO"),
                            ("trend", "TOP")):
            c = Chip(nome)
            c.valor, c.active = chave, chave == "video"
            c.bind(on_release=lambda w: self._aba(w.valor))
            self.abas[chave] = c
            abas.add_widget(c)
        col.add_widget(abas)
        self.painel = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(16))
        self.painel.bind(minimum_height=self.painel.setter("height"))
        col.add_widget(self.painel)
        self.p_video = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(16))
        self.p_video.bind(minimum_height=self.p_video.setter("height"))
        self.p_chat = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10))
        self.p_chat.bind(minimum_height=self.p_chat.setter("height"))
        self.p_code = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10))
        self.p_code.bind(minimum_height=self.p_code.setter("height"))
        self.p_trend = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10))
        self.p_trend.bind(minimum_height=self.p_trend.setter("height"))
        self.chat_hist = []
        self._montar_trend()
        self._montar_chat()
        self._montar_code()

        # cartao de entrada
        entrada = Glass(border=alpha(CYAN, 0.45))
        entrada.add_widget(tlabel("SUA IDEIA", 11, CYAN, bold=True))
        self.ideia = TextInput(
            hint_text="Descreva o video que voce quer criar...", multiline=True,
            size_hint_y=None, height=dp(96), font_size=sp(17),
            background_normal="", background_active="",
            background_color=(0, 0, 0, 0), foreground_color=TEXT,
            hint_text_color=alpha(MUTED, 0.8), cursor_color=CYAN, padding=[0, dp(6)])
        entrada.add_widget(self.ideia)
        entrada.add_widget(tlabel("PLATAFORMA", 11, MUTED, bold=True))
        linha = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(8))
        self.chips = []
        for nome, valor in PLATAFORMAS:
            c = Chip(nome)
            c.valor = valor
            c.active = valor == self.plataforma
            c.bind(on_release=self._escolher)
            self.chips.append(c)
            linha.add_widget(c)
        entrada.add_widget(linha)
        self.p_video.add_widget(entrada)

        self.tipo, self.foto_b64, self.meu_video = "normal", "", None
        opc = Glass(border=alpha(MAGENTA, 0.45))
        opc.add_widget(tlabel("TIPO DE VIDEO", 11, MAGENTA, bold=True))
        lt = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
        self.chips_tipo = []
        for nome, valor in (("Normal", "normal"), ("Anuncio de produto", "anuncio")):
            c = Chip(nome, height=38)
            c.valor, c.active = valor, valor == "normal"
            c.bind(on_release=self._escolher_tipo)
            self.chips_tipo.append(c)
            lt.add_widget(c)
        opc.add_widget(lt)
        opc.add_widget(tlabel("Anuncio: a IA escreve um comercial (gancho, produto em destaque, "
                              "chamada para comprar). Com FOTO DO PRODUTO o video anima a sua foto.",
                              12, MUTED))
        self.foto_chip = Chip("FOTO DO PRODUTO (opcional)", height=38)
        self.foto_chip.bind(on_release=self._escolher_foto)
        opc.add_widget(self.foto_chip)
        self.meuv_chip = Chip("USAR VIDEO MEU (Gemini, MiniMax, galeria)", height=38)
        self.meuv_chip.bind(on_release=self._escolher_meu_video)
        opc.add_widget(self.meuv_chip)
        atalhos = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
        for nome, url in (("Abrir Gemini", "https://gemini.google.com/app"),
                          ("Abrir MiniMax", "https://hailuoai.video")):
            c = Chip(nome, height=38)
            c.bind(on_release=lambda w, u=url: webbrowser.open(u))
            atalhos.add_widget(c)
        opc.add_widget(atalhos)
        self.opc_lb = tlabel("Gere o video no Gemini ou no MiniMax, toque em COMPARTILHAR e "
                             "escolha 'Ttoken Studio': ele chega aqui pronto. (Ou escolha o "
                             "arquivo no botao acima.) Depois: ideia, GERAR e postar.", 12, MUTED)
        opc.add_widget(self.opc_lb)
        self.p_video.add_widget(opc)

        self.btn = NeonButton("GERAR VIDEO + LEGENDA")
        self.btn.bind(on_release=self.iniciar)
        self.p_video.add_widget(self.btn)

        self.llm, self.videos = Maestro(), VideoMaestro()
        self.chaves, self._shared, self._dl = {}, None, None
        self._ultimo_erro = ""
        self.modo_lb = tlabel("", 11, MUTED, align="center")
        self.p_video.add_widget(self.modo_lb)
        chip_chaves = Chip("Chaves (usar sem servidor)", height=38)
        chip_chaves.bind(on_release=self._alternar_chaves)
        self.p_video.add_widget(chip_chaves)
        self.chaves_slot = BoxLayout(orientation="vertical", size_hint_y=None)
        self.chaves_slot.bind(minimum_height=self.chaves_slot.setter("height"))
        self.p_video.add_widget(self.chaves_slot)

        # area dinamica: progresso / resultado / erro
        self.slot = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(14))
        self.slot.bind(minimum_height=self.slot.setter("height"))
        self.p_video.add_widget(self.slot)
        self.painel.add_widget(self.p_video)

        col.add_widget(tlabel("O app nunca posta sozinho: voce confere e toca em postar na "
                              "rede que quiser.", 11, MUTED, align="center"))
        scroll.add_widget(col)
        root.add_widget(scroll)
        self._carregar_chaves()
        return root

    # --------------------------------------------------------- interacao
    def _escolher(self, chip):
        self.plataforma = chip.valor
        for c in self.chips:
            c.active = c is chip

    def iniciar(self, *_):
        ideia = self.ideia.text.strip()
        if len(ideia) < 3:
            self._erro("Digite uma ideia com pelo menos 3 letras.")
            return
        self.btn.disabled = True
        self.btn.text = "PROCESSANDO..."
        if pedir_permissao:
            try:
                pedir_permissao(["android.permission.POST_NOTIFICATIONS"])
            except Exception:
                pass
        self._montar_progresso()
        self._shared = self._dl = None
        if self.foto_b64 and self._tem_chaves():
            self.opc_lb.text = "Foto do produto so e usada pelo servidor (modo nuvem)."
        alvo = self._rodar_local if self._tem_chaves() else self._trabalhar
        threading.Thread(target=alvo, args=(ideia, self.plataforma), daemon=True).start()

    # --------------------------------------------------------- progresso
    def _montar_progresso(self):
        self.slot.clear_widgets()
        card = Glass(border=alpha(VIOLET, 0.5))
        card.add_widget(tlabel("EM ANDAMENTO", 11, VIOLET, bold=True))
        self.status = tlabel("Conectando...", 15)
        card.add_widget(self.status)
        linha = BoxLayout(size_hint_y=None, height=dp(24), spacing=dp(6))
        self.passos = []
        for nome in PASSOS:
            lb = Label(text=nome, font_size=sp(11), bold=True, color=MUTED)
            self.passos.append(lb)
            linha.add_widget(lb)
        card.add_widget(linha)
        self.barra = NeonBar()
        card.add_widget(self.barra)
        self._t0, self._eta = time.time(), "Video: de 1 a 8 min conforme o gerador"
        self.tempo_lb = tlabel(self._eta, 12, MUTED)
        card.add_widget(self.tempo_lb)
        card.add_widget(tlabel("Pode sair do app: avisamos com uma notificacao quando ficar pronto.",
                               12, MUTED))
        self._acompanhando = True
        Clock.unschedule(self._tique)
        Clock.schedule_interval(self._tique, 1)
        self.slot.add_widget(card)
        self._marcar(0)
        Animation(frac=0.08, d=0.6).start(self.barra)

    def _marcar(self, ativo):
        for i, lb in enumerate(self.passos):
            Animation.cancel_all(lb, "opacity")
            lb.opacity = 1
            if i < ativo:
                lb.color = CYAN
            elif i == ativo:
                lb.color = TEXT
                anim = Animation(opacity=0.35, d=0.7) + Animation(opacity=1, d=0.7)
                anim.repeat = True
                anim.start(lb)
            else:
                lb.color = MUTED

    def _progresso(self, passo, etapa, eta=""):
        if not hasattr(self, "status"):
            return
        if eta:
            self._eta = eta
        self.status.text = etapa
        self._marcar(min(passo, 2))
        Animation(frac=min(0.97, (passo + 0.6) / 3), d=0.8).start(self.barra)

    # ------------------------------------------------------------- rede
    def _trabalhar(self, ideia, plataforma):
        s = requests.Session()
        s.max_redirects = 10           # servidor dormindo pode redirecionar em loop
        s.headers.update({"User-Agent": UA, "Accept": "application/json"})
        try:
            acordou = self._acordar(s)
            if acordou is None:          # ja mostrou o erro especifico
                return
            if not acordou:
                self._ui(self._erro, "O servidor nao respondeu (ultimo erro: %s).\n\n"
                         "Tente de novo em 1 minuto ou toque em \"Chaves\" para usar o "
                         "app sem servidor." % (self._ultimo_erro or "sem resposta"))
                return
            job, ultimo = None, ""
            for _ in range(3):
                try:
                    r = s.post(API + "/api/gerar", timeout=30,
                               json={"ideia": ideia, "plataforma": plataforma,
                                     "tipo": self.tipo, "imagem_b64": self.foto_b64,
                                     "sem_video": bool(self.meu_video)})
                    if r.status_code == 429:
                        self._ui(self._erro, "Voce atingiu o limite de pedidos por hora. "
                                 "Tente de novo mais tarde.")
                        return
                    if r.status_code >= 500:
                        raise RuntimeError("servidor ocupado (HTTP %d)" % r.status_code)
                    r.raise_for_status()
                    job = r.json()["job_id"]
                    break
                except Exception as e:
                    ultimo = str(e)
                    time.sleep(3)
            if not job:
                self._ui(self._erro, "Nao consegui enviar a ideia: " + ultimo)
                return
            self._guardar_job(job)
            self._acompanhar(s, job)
        except Exception as e:
            self._ui(self._erro, "Falha inesperada: %s" % e)

    def _acompanhar(self, s, job):
        limite, falhas = time.time() + MAX_WAIT, 0
        try:
            while time.time() < limite:
                time.sleep(POLL_EVERY)
                try:
                    r = s.get(API + "/api/status/" + job, timeout=20)
                    if r.status_code == 404:
                        self._limpar_job()
                        self._ui(self._erro, "Esse pedido expirou no servidor. Gere de novo.")
                        return
                    st = r.json()
                    falhas = 0
                except Exception:
                    falhas += 1
                    if falhas >= 15:      # ~1 min sem rede; o job fica guardado para retomar
                        self._ui(self._erro, "Perdi a conexao. Abra o app de novo para retomar.")
                        return
                    continue
                if st.get("status") == "processando":
                    eta = ""
                    if st.get("eta_min"):
                        eta = "Estimativa: %s a %s min (%s)" % (
                            st["eta_min"], st["eta_max"], st.get("gerador", "video"))
                    self._ui(self._progresso, int(st.get("passo", 0)),
                             st.get("etapa", "Processando"), eta)
                elif st.get("status") == "pronto":
                    self._limpar_job()
                    self._ui(self._pronto, st)
                    return
                else:
                    self._limpar_job()
                    self._ui(self._erro, st.get("erro", "Erro desconhecido"))
                    return
            self._limpar_job()
            self._ui(self._erro, "Tempo esgotado. Tente de novo em instantes.")
        except Exception as e:
            self._ui(self._erro, "Falha inesperada: %s" % e)

    def _acordar(self, s):
        """O SnapDeploy dorme quando ocioso: insiste ate ele responder /health.
        Retorna True (acordou), False (nao respondeu) ou None (sem chave de IA)."""
        limite = time.time() + WAKE_WAIT
        primeira = True
        while time.time() < limite:
            try:
                r = s.get(API + "/health", timeout=15)
                try:
                    dados = r.json()
                except Exception:
                    dados = None
                if r.status_code == 200 and isinstance(dados, dict) and dados.get("ok"):
                    if not dados.get("ias"):
                        self._ui(self._erro, "O servidor esta sem nenhuma chave de IA "
                                 "configurada (GROQ_API_KEY, GEMINI_API_KEY...).")
                        return None
                    return True
                self._ultimo_erro = "HTTP %d (%s)" % (
                    r.status_code, (r.headers.get("content-type") or "?")[:30])
            except Exception as e:
                self._ultimo_erro = "%s: %s" % (type(e).__name__, str(e)[:80])
            if primeira:
                self._ui(self._progresso, 0, "Acordando o servidor (ate 1 minuto)...")
                primeira = False
            time.sleep(4)
        return False

    # ------------------------------------------- notificacao e retomada
    def _notificar(self, titulo, texto):
        """Notificacao do Android (so se o app estiver em segundo plano)."""
        if platform != "android" or autoclass is None or not getattr(self, "_fundo", False):
            return
        try:
            from jnius import cast
            act = autoclass("org.kivy.android.PythonActivity").mActivity
            Context = autoclass("android.content.Context")
            nm = act.getSystemService(Context.NOTIFICATION_SERVICE)
            String = autoclass("java.lang.String")

            def cs(t):
                return cast("java.lang.CharSequence", String(t))
            NB = autoclass("android.app.Notification$Builder")
            if autoclass("android.os.Build$VERSION").SDK_INT >= 26:
                NC = autoclass("android.app.NotificationChannel")
                nm.createNotificationChannel(NC("ttoken", cs("Ttoken Studio"), 3))
                b = NB(act, "ttoken")
            else:
                b = NB(act)
            intent = act.getPackageManager().getLaunchIntentForPackage(act.getPackageName())
            pi = autoclass("android.app.PendingIntent").getActivity(act, 0, intent, 67108864)
            b.setSmallIcon(act.getApplicationInfo().icon)
            b.setContentTitle(cs(titulo))
            b.setContentText(cs(texto))
            b.setContentIntent(pi)
            b.setAutoCancel(True)
            nm.notify(1, b.build())
        except Exception:
            pass

    def on_pause(self):
        self._fundo = True
        return True

    def on_resume(self):
        self._fundo = False
        self._retomar()

    def on_start(self):
        self._fundo = False
        self._ligar_compartilhar()
        self._retomar()

    def _arq_job(self):
        return os.path.join(getattr(self, "user_data_dir", None) or ".", "job.json")

    def _guardar_job(self, job):
        try:
            with open(self._arq_job(), "w", encoding="utf-8") as f:
                json.dump({"job": job, "t0": time.time()}, f)
        except Exception:
            pass

    def _limpar_job(self):
        try:
            os.remove(self._arq_job())
        except Exception:
            pass

    def _retomar(self):
        """Se o app foi fechado/reaberto no meio de um video, continua acompanhando."""
        if getattr(self, "_acompanhando", False):
            return
        try:
            with open(self._arq_job(), encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        if time.time() - d.get("t0", 0) > 3300:
            self._limpar_job()
            return
        self._montar_progresso()
        self.btn.disabled = True
        self.btn.text = "PROCESSANDO..."
        self._t0 = d["t0"]
        self._eta = ""
        threading.Thread(target=self._acompanhar_novo, args=(d["job"],), daemon=True).start()

    def _acompanhar_novo(self, job):
        s = requests.Session()
        s.max_redirects = 10
        s.headers.update({"User-Agent": UA, "Accept": "application/json"})
        self._acompanhar(s, job)

    def _tique(self, dt):
        """Atualiza 'Estimativa: X a Y min - passou M:SS' a cada segundo."""
        if not hasattr(self, "tempo_lb") or not getattr(self, "_acompanhando", False):
            return
        seg = int(time.time() - self._t0)
        self.tempo_lb.text = ("%s  |  passou %d:%02d" % (self._eta, seg // 60, seg % 60)
                              if self._eta else "passou %d:%02d" % (seg // 60, seg % 60))

    # ------------------------------------------------------------ abas
    def _aba(self, chave):
        for k, c in self.abas.items():
            c.active = (k == chave)
        self.painel.clear_widgets()
        self.painel.add_widget({"video": self.p_video, "chat": self.p_chat,
                                "code": self.p_code, "trend": self.p_trend}[chave])

    def _perguntar(self, mensagem, historico, max_tokens=1500):
        """Chat/codigo: usa suas chaves no celular, senao o servidor (/api/chat)."""
        if self._tem_chaves():
            msgs = [{"role": "system", "content": "Voce e o assistente do Ttoken Studio: "
                     "roteiros, legendas, ideias de video e codigo (Python/Kivy). "
                     "Responda em portugues, claro e pratico."}]
            msgs += [h for h in historico[-10:]]
            msgs.append({"role": "user", "content": mensagem})
            return self.llm.chat(msgs, max_tokens=max_tokens)[0]
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        if self._acordar(s) is not True:
            raise RuntimeError("servidor indisponivel (%s)" % (self._ultimo_erro or "sem resposta"))
        r = s.post(API + "/api/chat", json={"mensagem": mensagem, "historico": historico[-10:]},
                   timeout=120)
        if r.status_code == 429:
            raise RuntimeError("limite de pedidos por hora atingido")
        r.raise_for_status()
        return r.json().get("resposta") or "(sem resposta)"

    def _montar_chat(self):
        self.chat_lista = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(8))
        self.chat_lista.bind(minimum_height=self.chat_lista.setter("height"))
        self.p_chat.add_widget(self.chat_lista)
        self._balao("Ola! Sou o Ttoken. Peca roteiro, legenda, ideia de video, codigo "
                    "Python/Kivy ou so converse.", False)
        card = Glass(border=alpha(VIOLET, 0.5))
        self.chat_inp = TextInput(hint_text="Digite sua mensagem...", multiline=True,
                                  size_hint_y=None, height=dp(72), font_size=sp(15),
                                  background_normal="", background_active="",
                                  background_color=(0, 0, 0, 0), foreground_color=TEXT,
                                  hint_text_color=alpha(MUTED, 0.8), cursor_color=CYAN)
        card.add_widget(self.chat_inp)
        self.anexo = None                      # (nome, texto) do arquivo escolhido
        self.anexo_lb = tlabel("", 12, CYAN)
        card.add_widget(self.anexo_lb)
        ferr = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        self.mic = Chip("MIC (falar)", height=40)
        self.mic.bind(on_release=self._alternar_mic)
        arq = Chip("ARQUIVO", height=40)
        arq.bind(on_release=self._escolher_arquivo)
        ferr.add_widget(self.mic)
        ferr.add_widget(arq)
        card.add_widget(ferr)
        env = NeonButton("ENVIAR", height=dp(46))
        env.bind(on_release=self._enviar_chat)
        card.add_widget(env)
        self.p_chat.add_widget(card)

    def _balao(self, texto, meu):
        g = Glass(border=alpha(CYAN if meu else VIOLET, 0.35))
        g.add_widget(tlabel(texto, 14, TEXT))
        self.chat_lista.add_widget(g)
        return g

    def _enviar_chat(self, *_):
        msg = (self.chat_inp.text or "").strip()
        if not msg:
            return
        self.chat_inp.text = ""
        completo = msg
        if self.anexo:
            nome, texto = self.anexo
            completo = "%s\n\n[Arquivo anexado: %s]\n%s" % (msg, nome, texto)
            msg = "%s\n(anexo: %s)" % (msg, nome)
            self.anexo, self.anexo_lb.text = None, ""
        self._balao(msg, True)
        espera = self._balao("...", False)
        hist = list(self.chat_hist)
        self.chat_hist.append({"role": "user", "content": completo})

        def trabalho():
            try:
                resp = self._perguntar(completo, hist)
            except Exception as e:
                resp = "Falha na conversa: %s" % str(e)[:160]
            self.chat_hist.append({"role": "assistant", "content": resp})
            self._ui(self._trocar_balao, espera, resp)
        threading.Thread(target=trabalho, daemon=True).start()

    def _trocar_balao(self, balao, texto):
        self.chat_lista.remove_widget(balao)
        self._balao(texto, False)

    def _montar_code(self):
        g = Glass(border=alpha(CYAN, 0.45))
        g.add_widget(tlabel("GERAR CODIGO", 11, CYAN, bold=True))
        g.add_widget(tlabel("Descreva o app ou trecho. A IA escreve o codigo; para virar APK, "
                            "envie ao GitHub e o Actions compila na nuvem.", 12, MUTED))
        self.code_inp = TextInput(hint_text="Ex: tela Kivy com botao que conta cliques...",
                                  multiline=True, size_hint_y=None, height=dp(96),
                                  font_size=sp(15), background_normal="", background_active="",
                                  background_color=(0, 0, 0, 0), foreground_color=TEXT,
                                  hint_text_color=alpha(MUTED, 0.8), cursor_color=CYAN)
        g.add_widget(self.code_inp)
        b = NeonButton("GERAR CODIGO", height=dp(48))
        b.bind(on_release=self._gerar_codigo)
        g.add_widget(b)
        self.p_code.add_widget(g)
        self.code_slot = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10))
        self.code_slot.bind(minimum_height=self.code_slot.setter("height"))
        self.p_code.add_widget(self.code_slot)

    def _gerar_codigo(self, *_):
        pedido = (self.code_inp.text or "").strip()
        self.code_slot.clear_widgets()
        if not pedido:
            self.code_slot.add_widget(tlabel("Descreva o que voce quer.", 13, MAGENTA))
            return
        self.code_slot.add_widget(tlabel("Gerando codigo...", 13, MUTED))
        prompt = ("Gere codigo completo e funcional. Prefira Python/Kivy se for app Android. "
                  "So o codigo e comentarios curtos.\n\nPedido: " + pedido)

        def trabalho():
            try:
                resp = self._perguntar(prompt, [], max_tokens=2500)
            except Exception as e:
                resp = "Erro: %s" % str(e)[:160]
            self._ui(self._mostrar_codigo, resp)
        threading.Thread(target=trabalho, daemon=True).start()

    def _mostrar_codigo(self, resp):
        self.code_slot.clear_widgets()
        card = Glass(border=alpha(VIOLET, 0.5))
        card.add_widget(tlabel(resp, 12, TEXT))
        cp = Chip("Copiar codigo", height=38)
        cp.bind(on_release=lambda w: self._copiar(w, resp, "Copiar codigo"))
        card.add_widget(cp)
        self.code_slot.add_widget(card)
        self.codigo_puro = self._extrair_codigo(resp)

        git = Glass(border=alpha(CYAN, 0.45))
        git.add_widget(tlabel("COMPILAR PELO GIT", 11, CYAN, bold=True))
        git.add_widget(tlabel("Envia este codigo para o branch de teste 'ia-build' (o app "
                              "principal fica intacto) e compila o APK no GitHub Actions.",
                              12, MUTED))
        self.arq_alvo = TextInput(text="main.py", multiline=False, size_hint_y=None,
                                  height=dp(40), font_size=sp(14), background_normal="",
                                  background_active="", background_color=(1, 1, 1, 0.07),
                                  foreground_color=TEXT, cursor_color=CYAN,
                                  padding=[dp(10), dp(9)])
        git.add_widget(tlabel("ARQUIVO NO REPOSITORIO", 11, MUTED, bold=True))
        git.add_widget(self.arq_alvo)
        b1 = NeonButton("ENVIAR AO GIT", height=dp(46))
        b1.bind(on_release=self._git_enviar)
        b2 = NeonButton("COMPILAR APK", c1=VIOLET, c2=MAGENTA, height=dp(46))
        b2.bind(on_release=self._git_compilar)
        git.add_widget(b1)
        git.add_widget(b2)
        self.git_msg = tlabel("", 12, MUTED)
        git.add_widget(self.git_msg)
        self.git_links = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(8))
        self.git_links.bind(minimum_height=self.git_links.setter("height"))
        git.add_widget(self.git_links)
        self.code_slot.add_widget(git)

    # ------------------------------------------- TOP: mais assistidos (inspiracao)
    def _montar_trend(self):
        g = Glass(border=alpha(MAGENTA, 0.5))
        g.add_widget(tlabel("MAIS ASSISTIDOS AGORA", 11, MAGENTA, bold=True))
        g.add_widget(tlabel("Veja o que esta bombando no YouTube (dados oficiais) e use o tema "
                            "como INSPIRACAO para criar a sua ideia original. O app nao baixa "
                            "videos de terceiros.", 12, MUTED))
        self.trend_q = TextInput(hint_text="Tema (opcional): ex. cachorro, receita, futebol",
                                 multiline=False, size_hint_y=None, height=dp(42),
                                 font_size=sp(14), background_normal="", background_active="",
                                 background_color=(1, 1, 1, 0.07), foreground_color=TEXT,
                                 hint_text_color=alpha(MUTED, 0.7), cursor_color=CYAN,
                                 padding=[dp(10), dp(11)])
        g.add_widget(self.trend_q)
        b = NeonButton("VER MAIS VISTOS", c1=MAGENTA, c2=VIOLET, height=dp(48))
        b.bind(on_release=self._buscar_trend)
        g.add_widget(b)
        self.trend_msg = tlabel("Precisa da chave do YouTube (Chaves, aba VIDEO) ou do "
                                "servidor com YOUTUBE_API_KEY.", 12, MUTED)
        g.add_widget(self.trend_msg)
        self.p_trend.add_widget(g)
        self.trend_lista = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10))
        self.trend_lista.bind(minimum_height=self.trend_lista.setter("height"))
        self.p_trend.add_widget(self.trend_lista)

    @staticmethod
    def _fmt_views(n):
        if n >= 1000000:
            return "%.1f mi" % (n / 1000000.0)
        if n >= 1000:
            return "%d mil" % (n // 1000)
        return str(n)

    def _buscar_trend(self, *_):
        q = (self.trend_q.text or "").strip()
        self.trend_msg.text = "Buscando..."
        self.trend_lista.clear_widgets()

        def trabalho():
            try:
                yk = (self.chaves.get("youtube") or "").strip()
                if yk:                                   # direto do celular, com a sua chave
                    itens = trending.mais_vistos("BR", q, 15, yk)
                else:
                    s = requests.Session()
                    s.headers.update({"User-Agent": UA})
                    if self._acordar(s) is not True:
                        raise RuntimeError("servidor indisponivel (%s)" %
                                           (self._ultimo_erro or "sem resposta"))
                    r = s.get(API + "/api/tendencias", params={"q": q}, timeout=60)
                    if r.status_code != 200:
                        det = r.json().get("detail", r.text) if "json" in r.headers.get(
                            "content-type", "") else r.text
                        raise RuntimeError(str(det)[:160])
                    itens = r.json()["itens"]
                self._ui(self._mostrar_trend, itens)
            except Exception as e:
                self._ui(setattr, self.trend_msg, "text", "Nao deu: %s" % str(e)[:180])
        threading.Thread(target=trabalho, daemon=True).start()

    def _mostrar_trend(self, itens):
        self.trend_lista.clear_widgets()
        self.trend_msg.text = ("%d videos, do mais visto para o menos." % len(itens)
                               if itens else "Nada encontrado para esse tema.")
        for n, it in enumerate(itens, 1):
            c = Glass(border=alpha(CYAN if n <= 3 else VIOLET, 0.4))
            c.add_widget(tlabel("[b]%d. %s[/b]" % (n, it["titulo"]), 14))
            c.add_widget(tlabel("%s  |  %s views" % (it["canal"], self._fmt_views(it["views"])),
                                12, MUTED))
            linha = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(8))
            usar = Chip("Usar como ideia", height=38)
            usar.bind(on_release=lambda w, t=it["titulo"]: self._usar_ideia(t))
            ver = Chip("Abrir", height=38)
            ver.bind(on_release=lambda w, u=it["url"]: webbrowser.open(u))
            linha.add_widget(usar)
            linha.add_widget(ver)
            c.add_widget(linha)
            self.trend_lista.add_widget(c)

    def _usar_ideia(self, titulo):
        self.ideia.text = "Video original inspirado no tema: " + titulo
        self._aba("video")

    # ------------------------------- tipo, foto do produto e video do usuario
    def _escolher_tipo(self, chip):
        self.tipo = chip.valor
        for c in self.chips_tipo:
            c.active = c is chip

    def _pegar_arquivo(self, mime, destino):
        if platform != "android" or Chooser is None:
            self.opc_lb.text = "Escolher arquivo so funciona no Android."
            return
        self._chooser2 = Chooser(lambda uris: self._arquivo_pego(uris, destino))
        self._chooser2.choose_content(mime)

    def _arquivo_pego(self, uris, destino):
        if not uris:
            return
        try:
            caminho = SharedStorage().copy_from_shared(uris[0])
        except Exception as e:
            self._ui(setattr, self.opc_lb, "text", "Nao consegui abrir: %s" % str(e)[:60])
            return
        if destino == "foto":
            threading.Thread(target=self._preparar_foto, args=(caminho,), daemon=True).start()
        else:
            self._ui(self._definir_meu_video, caminho)

    def _escolher_foto(self, *_):
        if self.foto_b64:                    # 2o toque remove a foto
            self.foto_b64 = ""
            self.foto_chip.text = "FOTO DO PRODUTO (opcional)"
            self.opc_lb.text = "Foto removida."
            return
        self._pegar_arquivo("image/*", "foto")

    @staticmethod
    def _reduzir_foto(caminho):
        """Reduz a foto (lado maior ~1400 px, JPEG 85) para caber no envio."""
        if autoclass is None:
            with open(caminho, "rb") as f:
                dados = f.read()
            if len(dados) > 3000000:
                raise RuntimeError("foto grande demais")
            return dados
        BF = autoclass("android.graphics.BitmapFactory")
        Opt = autoclass("android.graphics.BitmapFactory$Options")
        o = Opt()
        o.inJustDecodeBounds = True
        BF.decodeFile(caminho, o)
        amostra = 1
        while max(o.outWidth, o.outHeight) / float(amostra) > 1400:
            amostra *= 2
        o2 = Opt()
        o2.inSampleSize = amostra
        bmp = BF.decodeFile(caminho, o2)
        if bmp is None:
            raise RuntimeError("nao consegui ler a imagem")
        saida = autoclass("java.io.ByteArrayOutputStream")()
        bmp.compress(autoclass("android.graphics.Bitmap$CompressFormat").JPEG, 85, saida)
        return bytes(bytearray(saida.toByteArray()))

    def _preparar_foto(self, caminho):
        try:
            dados = self._reduzir_foto(caminho)
            self.foto_b64 = base64.b64encode(dados).decode()
            self._ui(setattr, self.foto_chip, "text", "FOTO OK (toque para remover)")
            self._ui(setattr, self.opc_lb, "text", "Foto pronta (%d KB). Escolha 'Anuncio de "
                     "produto' e descreva o produto." % (len(dados) // 1024))
        except Exception as e:
            self._ui(setattr, self.opc_lb, "text", "Foto nao usada: %s" % str(e)[:70])

    def _escolher_meu_video(self, *_):
        if self.meu_video:                   # 2o toque remove
            self.meu_video = None
            self.meuv_chip.text = "USAR VIDEO MEU (Gemini, MiniMax, galeria)"
            self.opc_lb.text = "Video removido: o app volta a gerar o video."
            return
        self._pegar_arquivo("video/*", "video")

    def _definir_meu_video(self, caminho):
        self.meu_video = caminho
        self.meuv_chip.text = "VIDEO ESCOLHIDO (toque para remover)"
        self.opc_lb.text = ("Seu video: %s. Descreva a ideia e toque em GERAR: o app escreve a "
                            "legenda e voce posta." % os.path.basename(caminho))

    # ----------------------------- recebe "Compartilhar" de outros apps (Gemini etc.)
    def _receber(self, mime, valor):
        """valor = caminho local (video/imagem) ou texto compartilhado."""
        mime = mime or ""
        if mime.startswith("video/"):
            self._definir_meu_video(valor)
            self.opc_lb.text = "Video recebido: " + self.opc_lb.text
        elif mime.startswith("image/"):
            self._escolher_tipo(self.chips_tipo[1])          # foto = anuncio de produto
            threading.Thread(target=self._preparar_foto, args=(valor,), daemon=True).start()
        elif mime.startswith("text/"):
            self.ideia.text = str(valor)[:2000]
            self.opc_lb.text = "Texto recebido na ideia."
        else:
            self.opc_lb.text = "Tipo de arquivo nao suportado: %s" % mime
            return
        self._aba("video")

    def _nova_intent(self, intent):
        """Chamado pelo Android quando outro app compartilha algo com o Ttoken Studio."""
        try:
            if intent is None or intent.getAction() != "android.intent.action.SEND":
                return
            mime = intent.getType() or ""
            if mime.startswith("text/"):
                txt = intent.getStringExtra("android.intent.extra.TEXT") or ""
                Clock.schedule_once(lambda dt: self._receber(mime, txt))
                return
            uri = intent.getParcelableExtra("android.intent.extra.STREAM")
            if uri is not None:
                threading.Thread(target=self._copiar_compartilhado,
                                 args=(mime, str(uri.toString())), daemon=True).start()
        except Exception:
            pass

    def _copiar_compartilhado(self, mime, uri):
        try:
            caminho = SharedStorage().copy_from_shared(uri)
            self._ui(self._receber, mime, caminho)
        except Exception as e:
            self._ui(setattr, self.opc_lb, "text", "Nao consegui abrir o que foi compartilhado: %s" % str(e)[:60])

    def _ligar_compartilhar(self):
        if platform != "android" or autoclass is None or SharedStorage is None:
            return
        try:
            from android import activity as _act
            _act.bind(on_new_intent=self._nova_intent)
            self._nova_intent(autoclass("org.kivy.android.PythonActivity").mActivity.getIntent())
        except Exception:
            pass

    def _tem_video(self):
        return bool(self.video_url or self.meu_video)

    # --------------------------------------------------- salvar na galeria
    def _salvar_galeria(self, *_):
        if not self._tem_video():
            self._aviso("Nao ha video para salvar.")
            return
        if platform != "android" or SharedStorage is None:
            if self.video_url:
                webbrowser.open(self.video_url)
            self._aviso("Abri o video no navegador: use 'baixar' de la.")
            return
        self._aviso("Baixando para a galeria...")

        def trabalho():
            try:
                self._baixar_video()
                self._ui(self._aviso, "Salvo na galeria (pasta Filmes).")
            except Exception as e:
                self._ui(self._aviso, "Nao consegui salvar (%s)." % str(e)[:60])
        threading.Thread(target=trabalho, daemon=True).start()

    # --------------------------------------------------- arquivo e audio no chat
    def _escolher_arquivo(self, *_):
        if platform != "android" or Chooser is None:
            self.anexo_lb.text = "Escolher arquivo so funciona no Android."
            return
        self._chooser = Chooser(self._arquivo_escolhido)
        self._chooser.choose_content("*/*")

    def _arquivo_escolhido(self, uris):
        if not uris:
            return
        try:
            caminho = SharedStorage().copy_from_shared(uris[0])
            self._ui(self._anexar, caminho)
        except Exception as e:
            self._ui(setattr, self.anexo_lb, "text", "Nao consegui abrir o arquivo: %s" % str(e)[:60])

    def _anexar(self, caminho):
        """Texto/codigo vai junto da proxima mensagem. Imagem/PDF/audio: nao suportado aqui."""
        try:
            with open(caminho, "rb") as f:
                bruto = f.read(300000)
            texto = bruto.decode("utf-8")
        except Exception:
            self.anexo, self.anexo_lb.text = None, ("So arquivos de texto/codigo por enquanto "
                                                    "(para voz use o MIC).")
            return
        nome = os.path.basename(caminho)
        self.anexo = (nome, texto[:12000])
        self.anexo_lb.text = "Anexo: %s (%d caracteres)" % (nome, min(len(texto), 12000))

    def _alternar_mic(self, *_):
        if getattr(self, "_rec", None):
            return self._parar_mic()
        if platform != "android" or autoclass is None:
            self.anexo_lb.text = "Gravar voz so funciona no Android."
            return
        try:
            if pedir_permissao:
                pedir_permissao([Permission.RECORD_AUDIO])
            ctx = autoclass("org.kivy.android.PythonActivity").mActivity
            self._rec_arq = os.path.join(str(ctx.getCacheDir().getAbsolutePath()), "voz.m4a")
            MR = autoclass("android.media.MediaRecorder")
            rec = MR()
            rec.setAudioSource(autoclass("android.media.MediaRecorder$AudioSource").MIC)
            rec.setOutputFormat(autoclass("android.media.MediaRecorder$OutputFormat").MPEG_4)
            rec.setAudioEncoder(autoclass("android.media.MediaRecorder$AudioEncoder").AAC)
            rec.setOutputFile(self._rec_arq)
            rec.prepare()
            rec.start()
            self._rec = rec
            self.mic.text = "PARAR e enviar"
        except Exception as e:
            self._rec = None
            self.anexo_lb.text = ("Permita o microfone e toque de novo (%s)" % str(e)[:50])

    def _parar_mic(self):
        rec, self._rec = self._rec, None
        self.mic.text = "MIC (falar)"
        try:
            rec.stop()
            rec.release()
        except Exception:
            pass
        self.anexo_lb.text = "Transcrevendo..."
        threading.Thread(target=self._transcrever_arq, args=(self._rec_arq,), daemon=True).start()

    def _transcrever_arq(self, caminho):
        try:
            with open(caminho, "rb") as f:
                dados = f.read()
            texto = self._transcrever(dados)
            self._ui(self._voz_pronta, texto)
        except Exception as e:
            self._ui(setattr, self.anexo_lb, "text", "Falha na voz: %s" % str(e)[:120])

    def _transcrever(self, dados):
        gk = (self.chaves.get("groq") or "").strip()
        if gk:                                   # modo celular: fala direto com o Groq
            return transcrever(dados, "voz.m4a", [gk])
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        if self._acordar(s) is not True:
            raise RuntimeError("servidor indisponivel")
        r = s.post(API + "/api/transcrever",
                   json={"b64": base64.b64encode(dados).decode(), "nome": "voz.m4a"}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:80]))
        return r.json().get("texto", "")

    def _voz_pronta(self, texto):
        self.anexo_lb.text = "Voz transcrita - confira e toque em ENVIAR." if texto else "Nao ouvi nada."
        self.chat_inp.text = (self.chat_inp.text + " " + texto).strip()

    # ---------------------------------------------------- compilar pelo Git
    @staticmethod
    def _extrair_codigo(resp):
        m = re.search(r"```[\w+-]*\n(.*?)```", resp, re.S)
        return (m.group(1) if m else resp).strip() + "\n"

    def _build_req(self, metodo, caminho, **kw):
        senha = (self.chaves.get("build") or "").strip()
        if not senha:
            raise RuntimeError("Coloque a SENHA DE BUILD em Chaves (aba VIDEO) - a mesma "
                               "BUILD_SECRET do servidor.")
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "X-Build-Secret": senha})
        if self._acordar(s) is not True:
            raise RuntimeError("servidor indisponivel (%s)" % (self._ultimo_erro or "sem resposta"))
        r = s.request(metodo, API + caminho, timeout=60, **kw)
        if r.status_code != 200:
            try:
                det = r.json().get("detail", r.text)
            except Exception:
                det = r.text
            raise RuntimeError(str(det)[:200])
        return r.json()

    def _git_msg(self, txt):
        self.git_msg.text = txt

    def _git_enviar(self, *_):
        alvo = (self.arq_alvo.text or "").strip()
        self._git_msg("Enviando %s..." % alvo)

        def trabalho():
            try:
                r = self._build_req("POST", "/api/build/arquivo",
                                    json={"path": alvo, "conteudo": self.codigo_puro})
                self._ui(self._git_msg, "Enviado: commit %s no branch %s. Agora toque em COMPILAR APK."
                         % (r["commit"], r["branch"]))
            except Exception as e:
                self._ui(self._git_msg, "Falhou: %s" % e)
        threading.Thread(target=trabalho, daemon=True).start()

    def _git_compilar(self, *_):
        self.git_links.clear_widgets()
        self._git_msg("Iniciando o build...")
        threading.Thread(target=self._git_trabalho, daemon=True).start()

    def _git_trabalho(self):
        try:
            run = self._build_req("POST", "/api/build/iniciar")["run_id"]
            t0 = time.time()
            while time.time() - t0 < 25 * 60:
                st = self._build_req("GET", "/api/build/status/%d" % run)
                if st["status"] == "completed":
                    self._ui(self._git_fim, run, st)
                    return
                self._ui(self._git_msg, "Compilando no GitHub... (%d min)" % ((time.time() - t0) // 60))
                time.sleep(10)
            self._ui(self._git_msg, "Demorou demais; acompanhe no GitHub Actions.")
        except Exception as e:
            self._ui(self._git_msg, "Falhou: %s" % e)

    def _git_fim(self, run, st):
        if st.get("conclusao") != "success":
            self._git_msg("O build falhou. Veja o log no GitHub.")
        else:
            self._git_msg("APK pronto!")
            senha = (self.chaves.get("build") or "").strip()
            b = Chip("Baixar APK", height=40)
            b.bind(on_release=lambda w: webbrowser.open("%s/api/build/apk/%d?s=%s" % (API, run, senha)))
            self.git_links.add_widget(b)
        v = Chip("Ver no GitHub", height=40)
        v.bind(on_release=lambda w: webbrowser.open(st["url"]))
        self.git_links.add_widget(v)

    # ------------------------------------------- modo sem servidor (no celular)
    def _rodar_local(self, ideia, plataforma):
        """Mesmo pipeline do servidor, rodando aqui com as chaves do usuario."""
        resultado = {}

        def report(**kw):
            resultado.update(kw)
            if kw.get("status") == "pronto":
                self._dl = resultado.get("_dl")
                self._ui(self._pronto, dict(resultado))
            elif kw.get("status") == "erro":
                self._ui(self._erro, kw.get("erro", "Erro desconhecido"))
            elif "etapa" in kw:
                self._ui(self._progresso, int(kw.get("passo", 0)), kw["etapa"])

        try:
            self._ui(self._progresso, 0, "Preparando (modo sem servidor)...")
            produzir(ideia, plataforma, self.llm, self.videos, report,
                     tipo=self.tipo, sem_video=bool(self.meu_video))
        except Exception as e:
            self._ui(self._erro, "Falha inesperada: %s" % e)

    def _arq_chaves(self):
        return os.path.join(getattr(self, "user_data_dir", None) or ".", "chaves.json")

    def _carregar_chaves(self):
        try:
            with open(self._arq_chaves(), encoding="utf-8") as f:
                self.chaves = json.load(f)
        except Exception:
            self.chaves = {}
        self._aplicar_chaves()

    def _aplicar_chaves(self):
        for campo, _rot, var in CHAVES:
            v = (self.chaves.get(campo) or "").strip()
            if v:
                os.environ[var] = v
            else:
                os.environ.pop(var, None)
        self.modo_lb.text = ("Modo: no celular (suas chaves, sem servidor)"
                             if self._tem_chaves() else "Modo: servidor na nuvem")

    def _tem_chaves(self):
        return any((self.chaves.get(c) or "").strip() for c in TEXTO_KEYS)

    def _alternar_chaves(self, *_):
        if self.chaves_slot.children:
            self.chaves_slot.clear_widgets()
            return
        card = Glass(border=alpha(VIOLET, 0.5))
        card.add_widget(tlabel("SUAS CHAVES (ficam so neste celular)", 11, VIOLET, bold=True))
        card.add_widget(tlabel("1) Toque em \"Pegar chave gratis\", crie a chave no site e copie. "
                               "2) Volte aqui, toque em \"Colar\". 3) SALVAR e depois TESTAR. "
                               "Com uma chave de texto o app funciona sem servidor; a de video e "
                               "opcional (sem ela voce recebe so a legenda).", 12, MUTED))
        self._campos = {}
        for campo, rotulo, _var in CHAVES:
            card.add_widget(tlabel(rotulo, 11, MUTED, bold=True))
            ti = TextInput(text=self.chaves.get(campo, ""), password=True, multiline=False,
                           hint_text="cole a chave aqui", hint_text_color=alpha(MUTED, 0.6),
                           size_hint_y=None, height=dp(42), font_size=sp(14),
                           background_normal="", background_active="",
                           background_color=(1, 1, 1, 0.07), foreground_color=TEXT,
                           cursor_color=CYAN, padding=[dp(10), dp(11)])
            self._campos[campo] = ti
            card.add_widget(ti)
            linha = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(8))
            colar = Chip("Colar", height=34)
            colar.bind(on_release=lambda w, t=ti: self._colar(t))
            linha.add_widget(colar)
            if campo in LINKS:
                pegar = Chip("Pegar chave gratis", height=34)
                pegar.bind(on_release=lambda w, u=LINKS[campo]: webbrowser.open(u))
                linha.add_widget(pegar)
            card.add_widget(linha)
        salvar = NeonButton("SALVAR CHAVES", height=dp(48))
        salvar.bind(on_release=self._salvar_chaves)
        card.add_widget(salvar)
        testar = Chip("TESTAR MINHAS CHAVES", height=40)
        testar.bind(on_release=self._testar_chaves)
        card.add_widget(testar)
        srv = Chip("TESTAR O SERVIDOR", height=40)
        srv.bind(on_release=self._testar_servidor)
        card.add_widget(srv)
        self.teste_lb = tlabel("", 12, MUTED)
        card.add_widget(self.teste_lb)
        self.chaves_slot.add_widget(card)

    def _colar(self, campo):
        try:
            campo.text = (Clipboard.paste() or "").strip()
        except Exception:
            pass

    def _gravar_campos(self):
        self.chaves = {c: self._campos[c].text.strip() for c, _r, _v in CHAVES}
        try:
            with open(self._arq_chaves(), "w", encoding="utf-8") as f:
                json.dump(self.chaves, f)
        except Exception:
            pass
        self._aplicar_chaves()

    def _testar_chaves(self, *_):
        self._gravar_campos()
        if not self._tem_chaves():
            self.teste_lb.text = "Cole pelo menos uma chave de texto (Groq e a mais rapida)."
            return
        self.teste_lb.text = "Testando suas chaves..."

        def trabalho():
            try:
                linhas = []
                for r in self.llm.testar():
                    linhas.append(("OK   " if r["ok"] else "ERRO ") + r["provedor"] +
                                  ("" if r["ok"] else ": " + r["erro"]))
                if (self.chaves.get("magichour") or self.chaves.get("fal")):
                    linhas.append("Video: a chave so e testada ao gerar um video.")
                self._ui(setattr, self.teste_lb, "text", "\n".join(linhas))
            except Exception as e:
                self._ui(setattr, self.teste_lb, "text", "Falha no teste: %s" % e)
        threading.Thread(target=trabalho, daemon=True).start()

    def _testar_servidor(self, *_):
        self.teste_lb.text = "Acordando o servidor e testando as IAs (pode levar 1 min)..."

        def trabalho():
            try:
                s = requests.Session()
                s.headers.update({"User-Agent": UA})
                if self._acordar(s) is not True:
                    self._ui(setattr, self.teste_lb, "text",
                             "Servidor fora do ar (%s)." % (self._ultimo_erro or "sem resposta"))
                    return
                dados = s.get(API + "/api/testar", timeout=90).json()
                linhas = [("OK   " if r["ok"] else "ERRO ") + r["provedor"] +
                          ("" if r["ok"] else ": " + r["erro"]) for r in dados.get("textos", [])]
                vids = [v["tipo"] for v in dados.get("videos", [])]
                linhas.append("Videos no servidor: " + (", ".join(vids) or "nenhum"))
                self._ui(setattr, self.teste_lb, "text", "\n".join(linhas))
            except Exception as e:
                self._ui(setattr, self.teste_lb, "text", "Falha no teste: %s" % str(e)[:140])
        threading.Thread(target=trabalho, daemon=True).start()

    def _salvar_chaves(self, *_):
        self._gravar_campos()
        self.teste_lb.text = ("Salvo. Modo: no celular." if self._tem_chaves()
                              else "Salvo. Sem chave de texto: o app usa o servidor.")

    # ------------------------------------------------- postar nas redes
    def _aviso(self, msg):
        if hasattr(self, "aviso_post"):
            self.aviso_post.text = msg

    def _compartilhar(self, botao, nome, pacote):
        """Abre o app da rede (ja logado no celular) com o video; legenda na area
        de transferencia. O usuario so cola e toca em postar."""
        Clipboard.copy(self.texto)
        if not self._tem_video() and (pacote in PRECISA_VIDEO):
            self._aviso("%s precisa do video e ele nao ficou pronto. Gere de novo." % nome)
            return
        if platform != "android" or ShareSheet is None:
            if self.video_url:
                webbrowser.open(self.video_url)
            self._aviso("Legenda copiada. (Compartilhar direto so funciona no Android.)")
            return
        self._aviso("Preparando o video para %s..." % nome)
        threading.Thread(target=self._preparar_envio, args=(nome, pacote),
                         daemon=True).start()

    def _preparar_envio(self, nome, pacote):
        try:
            arq = self._baixar_video() if self._tem_video() else None
        except Exception as e:
            self._ui(self._aviso, "Nao consegui baixar o video (%s)." % str(e)[:60])
            return
        self._ui(self._abrir_rede, nome, pacote, arq)

    def _baixar_video(self):
        """Baixa o mp4 e salva em Filmes (galeria) para poder compartilhar."""
        if self._shared:
            return self._shared
        if self.meu_video and not self.video_url:        # video que o usuario escolheu
            env = autoclass("android.os.Environment")
            self._shared = SharedStorage().copy_to_shared(
                self.meu_video, collection=env.DIRECTORY_MOVIES)
            if not self._shared:
                raise RuntimeError("nao consegui preparar o video")
            return self._shared
        destino = os.path.join(getattr(self, "user_data_dir", None) or ".", "ttoken_video.mp4")
        r = requests.get(self.video_url, headers=self._dl, stream=True, timeout=120)
        r.raise_for_status()
        with open(destino, "wb") as f:
            for pedaco in r.iter_content(65536):
                f.write(pedaco)
        env = autoclass("android.os.Environment")
        self._shared = SharedStorage().copy_to_shared(destino, collection=env.DIRECTORY_MOVIES)
        if not self._shared:
            raise RuntimeError("nao consegui salvar na galeria")
        return self._shared

    def _abrir_rede(self, nome, pacote, arq):
        self._sheet = ShareSheet()           # precisa continuar vivo enquanto compartilha
        try:
            if arq:
                self._sheet.share_file(arq, app=pacote) if pacote else self._sheet.share_file(arq)
            else:
                (self._sheet.share_plain_text(self.texto, app=pacote) if pacote
                 else self._sheet.share_plain_text(self.texto))
            self._aviso("Legenda copiada: cole ao postar no %s." % nome)
        except Exception:
            try:                            # app nao instalado: deixa o usuario escolher
                (self._sheet.share_file(arq) if arq else self._sheet.share_plain_text(self.texto))
                self._aviso("%s nao abriu direto. Escolha o app na lista." % nome)
            except Exception as e:
                self._aviso("Nao consegui abrir o compartilhamento: %s" % str(e)[:60])

    @staticmethod
    def _ui(fn, *args):
        Clock.schedule_once(lambda dt: fn(*args))

    # ---------------------------------------------------------- resultado
    def _fim(self):
        self._acompanhando = False
        Clock.unschedule(self._tique)
        self.btn.disabled = False
        self.btn.text = "GERAR VIDEO + LEGENDA"

    def _erro(self, msg):
        self._notificar("Ttoken Studio", "Nao deu certo: " + str(msg)[:80])
        self.slot.clear_widgets()
        card = Glass(border=alpha(MAGENTA, 0.6))
        card.add_widget(tlabel("ALGO DEU ERRADO", 11, MAGENTA, bold=True))
        card.add_widget(tlabel(str(msg), 15))
        self.slot.add_widget(card)
        self._fim()

    def _pronto(self, s):
        self._notificar("Seu video esta pronto!", "Abra o Ttoken Studio para postar.")
        self.texto = s.get("texto", "")
        self.video_url = s.get("video_url", "")
        self._shared = None
        self.slot.clear_widgets()
        card = Glass(border=alpha(CYAN, 0.55))
        card.add_widget(tlabel("PRONTO PARA POSTAR", 11, CYAN, bold=True))
        if s.get("titulo"):
            card.add_widget(tlabel("[b]%s[/b]" % s["titulo"], 20))
        card.add_widget(tlabel("LEGENDA", 11, MUTED, bold=True))
        card.add_widget(tlabel(self.texto or "(sem legenda)", 15))

        t = s.get("tokens") or {}
        if t.get("modo") and t["modo"] != "passthrough":
            card.add_widget(tlabel("Ttokem: %s para %s tokens ([color=%s]-%s%%[/color])" % (
                t.get("antes"), t.get("depois"), hexc(CYAN), t.get("economia_pct")),
                12, MUTED))
        else:
            card.add_widget(tlabel("Ttokem: ideia curta, sem compressao necessaria", 12, MUTED))
        card.add_widget(tlabel("IA: %s" % s.get("api_llm", "-"), 12, MUTED))

        if self.video_url:
            card.add_widget(tlabel("VIDEO", 11, MUTED, bold=True))
            card.add_widget(tlabel("Gerado por %s" % s.get("api_video", "-"), 12, MUTED))
        elif self.meu_video:
            card.add_widget(tlabel("VIDEO", 11, MUTED, bold=True))
            card.add_widget(tlabel("O seu arquivo: %s" % os.path.basename(self.meu_video), 12, MUTED))
            bg = Chip("Salvar na galeria", height=46)
            bg.bind(on_release=self._salvar_galeria)
            card.add_widget(bg)
        else:
            card.add_widget(tlabel(s.get("aviso", "Video indisponivel agora."),
                                   13, alpha(MAGENTA, 1)))
        if self.video_url and s.get("aviso"):
            card.add_widget(tlabel(s["aviso"], 12, alpha(MAGENTA, 1)))

        b1 = NeonButton("COPIAR LEGENDA")
        b1.bind(on_release=lambda *_: self._copiar(b1, self.texto, "COPIAR LEGENDA"))
        card.add_widget(b1)
        if self.video_url:
            linha = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
            b2 = Chip("Copiar link", height=46)
            b2.bind(on_release=lambda *_: self._copiar(b2, self.video_url, "Copiar link"))
            b3 = Chip("Abrir video", height=46)
            b3.bind(on_release=lambda *_: webbrowser.open(self.video_url))
            linha.add_widget(b2)
            linha.add_widget(b3)
            card.add_widget(linha)
            b4 = Chip("Salvar na galeria", height=46)
            b4.bind(on_release=self._salvar_galeria)
            card.add_widget(b4)

        card.add_widget(tlabel("POSTAR EM", 11, CYAN, bold=True))
        card.add_widget(tlabel("Abre o app da rede que ja esta logado no seu celular, com o "
                               "video pronto. A legenda vai copiada: e so colar.", 12, MUTED))
        for i in range(0, len(SOCIAIS), 3):
            linha = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
            for nome, pacote in SOCIAIS[i:i + 3]:
                b = Chip(nome, height=40)
                b.bind(on_release=lambda w, n=nome, p=pacote: self._compartilhar(w, n, p))
                linha.add_widget(b)
            card.add_widget(linha)
        self.aviso_post = tlabel("", 12, MUTED)
        card.add_widget(self.aviso_post)
        self.slot.add_widget(card)
        self._fim()

    @staticmethod
    def _copiar(botao, conteudo, rotulo):
        Clipboard.copy(conteudo)
        botao.text = "COPIADO"
        Clock.schedule_once(lambda dt: setattr(botao, "text", rotulo), 1.6)


if __name__ == "__main__":
    TtokenApp().run()
