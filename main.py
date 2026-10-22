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
MAX_WAIT = 480          # desiste depois de 8 minutos
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
          ("build", "SENHA DE BUILD (compilar pelo Git, igual a BUILD_SECRET)",
           "TTOKEN_BUILD_SECRET")]
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
        for chave, nome in (("video", "VIDEO"), ("chat", "CHAT"), ("code", "CODIGO")):
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
        self.chat_hist = []
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
        self._montar_progresso()
        self._shared = self._dl = None
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

    def _progresso(self, passo, etapa):
        if not hasattr(self, "status"):
            return
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
                               json={"ideia": ideia, "plataforma": plataforma})
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
            limite, falhas = time.time() + MAX_WAIT, 0
            while time.time() < limite:
                time.sleep(POLL_EVERY)
                try:
                    st = s.get(API + "/api/status/" + job, timeout=20).json()
                    falhas = 0
                except Exception:
                    falhas += 1
                    if falhas >= 6:
                        self._ui(self._erro, "Perdi a conexao com o servidor.")
                        return
                    continue
                if st.get("status") == "processando":
                    self._ui(self._progresso, int(st.get("passo", 0)),
                             st.get("etapa", "Processando"))
                elif st.get("status") == "pronto":
                    self._ui(self._pronto, st)
                    return
                else:
                    self._ui(self._erro, st.get("erro", "Erro desconhecido"))
                    return
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

    # ------------------------------------------------------------ abas
    def _aba(self, chave):
        for k, c in self.abas.items():
            c.active = (k == chave)
        self.painel.clear_widgets()
        self.painel.add_widget({"video": self.p_video, "chat": self.p_chat,
                                "code": self.p_code}[chave])

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
            produzir(ideia, plataforma, self.llm, self.videos, report)
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
        card.add_widget(tlabel("CHAVES (ficam so neste celular)", 11, VIOLET, bold=True))
        card.add_widget(tlabel("Com pelo menos uma chave de IA de texto o app funciona sem "
                               "servidor. Chave de video e opcional (sem ela voce recebe a "
                               "legenda). Deixe vazio para usar o servidor.", 12, MUTED))
        self._campos = {}
        for campo, rotulo, _var in CHAVES:
            card.add_widget(tlabel(rotulo, 11, MUTED, bold=True))
            ti = TextInput(text=self.chaves.get(campo, ""), password=True, multiline=False,
                           size_hint_y=None, height=dp(42), font_size=sp(14),
                           background_normal="", background_active="",
                           background_color=(1, 1, 1, 0.07), foreground_color=TEXT,
                           cursor_color=CYAN, padding=[dp(10), dp(11)])
            self._campos[campo] = ti
            card.add_widget(ti)
        salvar = NeonButton("SALVAR CHAVES", height=dp(48))
        salvar.bind(on_release=self._salvar_chaves)
        card.add_widget(salvar)
        self.chaves_slot.add_widget(card)

    def _salvar_chaves(self, *_):
        self.chaves = {c: self._campos[c].text.strip() for c, _r, _v in CHAVES}
        try:
            with open(self._arq_chaves(), "w", encoding="utf-8") as f:
                json.dump(self.chaves, f)
        except Exception:
            pass
        self._aplicar_chaves()
        self.chaves_slot.clear_widgets()

    # ------------------------------------------------- postar nas redes
    def _aviso(self, msg):
        if hasattr(self, "aviso_post"):
            self.aviso_post.text = msg

    def _compartilhar(self, botao, nome, pacote):
        """Abre o app da rede (ja logado no celular) com o video; legenda na area
        de transferencia. O usuario so cola e toca em postar."""
        Clipboard.copy(self.texto)
        if not self.video_url and (pacote in PRECISA_VIDEO):
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
            arq = self._baixar_video() if self.video_url else None
        except Exception as e:
            self._ui(self._aviso, "Nao consegui baixar o video (%s)." % str(e)[:60])
            return
        self._ui(self._abrir_rede, nome, pacote, arq)

    def _baixar_video(self):
        """Baixa o mp4 e salva em Filmes (galeria) para poder compartilhar."""
        if self._shared:
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
        self.btn.disabled = False
        self.btn.text = "GERAR VIDEO + LEGENDA"

    def _erro(self, msg):
        self.slot.clear_widgets()
        card = Glass(border=alpha(MAGENTA, 0.6))
        card.add_widget(tlabel("ALGO DEU ERRADO", 11, MAGENTA, bold=True))
        card.add_widget(tlabel(str(msg), 15))
        self.slot.add_widget(card)
        self._fim()

    def _pronto(self, s):
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
        else:
            card.add_widget(tlabel(s.get("aviso", "Video indisponivel agora."),
                                   13, alpha(MAGENTA, 1)))

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
