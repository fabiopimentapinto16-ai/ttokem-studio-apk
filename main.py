"""Ttoken Studio — app Kivy (visual neon/vidro).

Fluxo: ideia -> backend (Ttokem + maestro de IAs + maestro de video) -> legenda
e video prontos. O app NUNCA posta sozinho: o usuario da o ultimo toque.
"""
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

API = "https://ttoken-backend-8ba2a.containers.snapdeploy.app"
POLL_EVERY = 4          # segundos entre consultas
MAX_WAIT = 480          # desiste depois de 8 minutos
WAKE_WAIT = 150         # tempo maximo esperando o servidor acordar

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
        col.add_widget(entrada)

        self.btn = NeonButton("GERAR VIDEO + LEGENDA")
        self.btn.bind(on_release=self.iniciar)
        col.add_widget(self.btn)

        # area dinamica: progresso / resultado / erro
        self.slot = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(14))
        self.slot.bind(minimum_height=self.slot.setter("height"))
        col.add_widget(self.slot)

        col.add_widget(tlabel("O app nunca posta por voce: voce confere e toca em "
                              "postar na rede que quiser.", 11, MUTED, align="center"))
        scroll.add_widget(col)
        root.add_widget(scroll)
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
        threading.Thread(target=self._trabalhar, args=(ideia, self.plataforma),
                         daemon=True).start()

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
        s.max_redirects = 5            # servidor dormindo pode redirecionar em loop
        try:
            acordou = self._acordar(s)
            if acordou is None:          # ja mostrou o erro especifico
                return
            if not acordou:
                self._ui(self._erro, "O servidor nao respondeu. Tente de novo em "
                         "1 minuto (ele dorme quando fica parado).")
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
                dados = r.json() if r.status_code == 200 else None
                if isinstance(dados, dict) and dados.get("ok"):
                    if not dados.get("ias"):
                        self._ui(self._erro, "O servidor esta sem nenhuma chave de IA "
                                 "configurada (GROQ_API_KEY, GEMINI_API_KEY...).")
                        return None
                    return True
            except Exception:
                pass
            if primeira:
                self._ui(self._progresso, 0, "Acordando o servidor (ate 1 minuto)...")
                primeira = False
            time.sleep(4)
        return False

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
        self.slot.add_widget(card)
        self._fim()

    @staticmethod
    def _copiar(botao, conteudo, rotulo):
        Clipboard.copy(conteudo)
        botao.text = "COPIADO"
        Clock.schedule_once(lambda dt: setattr(botao, "text", rotulo), 1.6)


if __name__ == "__main__":
    TtokenApp().run()
