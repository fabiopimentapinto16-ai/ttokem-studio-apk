import threading
import time
import webbrowser

import requests
from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

API = "https://ttoken-backend-8ba2a.containers.snapdeploy.app"
POLL_EVERY = 4        # segundos entre consultas ao backend
MAX_WAIT = 420        # desiste depois de 7 minutos
VERDE = (0, 0.82, 0.62, 1)


class TtokenApp(App):
    texto = ""
    video_url = ""

    def build(self):
        self.title = "Ttoken Studio"
        Window.clearcolor = (0.04, 0.04, 0.06, 1)
        root = BoxLayout(orientation="vertical", padding=20, spacing=12)

        root.add_widget(Label(text="Ttoken Studio", font_size="26sp", bold=True,
                              color=VERDE, size_hint_y=None, height=50))

        self.ideia_input = TextInput(
            hint_text="Digite sua ideia de video...", multiline=True,
            size_hint_y=None, height=100,
            background_color=(0.08, 0.08, 0.12, 1),
            foreground_color=(0.95, 0.95, 0.95, 1),
            hint_text_color=(0.5, 0.5, 0.55, 1), padding=[12, 12])
        root.add_widget(self.ideia_input)

        self.btn_gerar = self._botao("Gerar Video + Legenda", VERDE, self.iniciar)
        root.add_widget(self.btn_gerar)

        self.scroll = ScrollView()
        self.resultado = Label(text="Aguardando sua ideia...", size_hint_y=None,
                               halign="left", valign="top",
                               color=(0.9, 0.9, 0.9, 1), padding=(8, 8))
        self.resultado.bind(
            width=lambda i, w: setattr(i, "text_size", (w - 16, None)),
            texture_size=lambda i, s: setattr(i, "height", s[1]))
        self.scroll.add_widget(self.resultado)
        root.add_widget(self.scroll)

        # Acoes pos-geracao: o usuario sempre da o ultimo clique para postar
        self.acoes = BoxLayout(size_hint_y=None, height=0, spacing=8, opacity=0)
        self.acoes.add_widget(self._botao("Copiar legenda", (0.15, 0.4, 0.9, 1),
                                          self.copiar_legenda))
        self.acoes.add_widget(self._botao("Copiar link", (0.35, 0.35, 0.45, 1),
                                          self.copiar_link))
        self.acoes.add_widget(self._botao("Abrir video", (0.35, 0.35, 0.45, 1),
                                          self.abrir_video))
        root.add_widget(self.acoes)
        return root

    @staticmethod
    def _botao(texto, cor, cb):
        b = Button(text=texto, size_hint_y=None, height=52, bold=True,
                   background_normal="", background_color=cor)
        b.bind(on_press=cb)
        return b

    # ------------------------------------------------------------ fluxo
    def iniciar(self, *_):
        ideia = self.ideia_input.text.strip()
        if len(ideia) < 3:
            self.resultado.text = "Digite uma ideia primeiro."
            return
        self.btn_gerar.disabled = True
        self.btn_gerar.text = "Processando..."
        self._acoes(False)
        self.resultado.text = "Enviando sua ideia..."
        threading.Thread(target=self._trabalhar, args=(ideia,), daemon=True).start()

    def _trabalhar(self, ideia):
        try:
            r = requests.post(f"{API}/api/gerar", json={"ideia": ideia}, timeout=30)
            r.raise_for_status()
            job = r.json()["job_id"]
            limite = time.time() + MAX_WAIT
            while time.time() < limite:
                time.sleep(POLL_EVERY)
                s = requests.get(f"{API}/api/status/{job}", timeout=20).json()
                if s.get("status") == "processando":
                    self._ui(self._mostrar_progresso, s.get("etapa", "Processando"))
                    continue
                if s.get("status") == "pronto":
                    self._ui(self._mostrar_pronto, s)
                else:
                    self._ui(self._mostrar_erro, s.get("erro", "Erro desconhecido"))
                return
            self._ui(self._mostrar_erro, "Tempo esgotado. Tente de novo em instantes.")
        except Exception as e:      # rede, JSON, HTTP
            self._ui(self._mostrar_erro, f"Falha de conexao: {e}")

    @staticmethod
    def _ui(fn, *args):
        Clock.schedule_once(lambda dt: fn(*args))

    # --------------------------------------------------------------- UI
    def _mostrar_progresso(self, etapa):
        self.resultado.text = f"{etapa}...\n\nIsso pode levar ate 2 minutos."

    def _mostrar_pronto(self, s):
        self.texto = s.get("texto", "")
        self.video_url = s.get("video_url", "")
        partes = [f"Pronto! IA: {s.get('api_llm', '-')}"]
        if s.get("titulo"):
            partes.append(f"\nTitulo: {s['titulo']}")
        partes.append(f"\nLegenda:\n{self.texto}")
        if self.video_url:
            partes.append(f"\nVideo:\n{self.video_url}")
        else:
            partes.append(f"\n{s.get('aviso', 'Video indisponivel agora.')}")
        partes.append("\nCopie a legenda e poste na rede que quiser.")
        self.resultado.text = "\n".join(partes)
        self._fim()
        self._acoes(True)

    def _mostrar_erro(self, msg):
        self.resultado.text = f"Erro: {msg}"
        self._fim()

    def _fim(self):
        self.btn_gerar.disabled = False
        self.btn_gerar.text = "Gerar Video + Legenda"

    def _acoes(self, mostrar):
        self.acoes.height = 52 if mostrar else 0
        self.acoes.opacity = 1 if mostrar else 0

    def copiar_legenda(self, *_):
        Clipboard.copy(self.texto)

    def copiar_link(self, *_):
        Clipboard.copy(self.video_url)

    def abrir_video(self, *_):
        if self.video_url:
            webbrowser.open(self.video_url)


if __name__ == "__main__":
    TtokenApp().run()
