import threading
import requests
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.clock import Clock

BACKEND_URL = "https://ttoken-backend-8ba2a.containers.snapdeploy.app/api/gerar"

class TtokenApp(App):
    def build(self):
        Window.clearcolor = (0.04, 0.04, 0.06, 1)
        layout = BoxLayout(orientation='vertical', padding=30, spacing=20)

        layout.add_widget(Label(
            text="Ttoken Studio",
            font_size='24sp',
            color=(0, 0.82, 0.62, 1),
            size_hint_y=None,
            height=60
        ))

        self.ideia_input = TextInput(
            hint_text="Digite sua ideia...",
            multiline=False,
            size_hint_y=None,
            height=60,
            background_color=(0.07, 0.07, 0.1, 1),
            foreground_color=(0.9, 0.9, 0.9, 1)
        )
        layout.add_widget(self.ideia_input)

        self.btn_gerar = Button(
            text="Gerar Vídeo",
            size_hint_y=None,
            height=60,
            background_color=(0, 0.82, 0.62, 1),
            background_normal='',
            font_size='18sp'
        )
        self.btn_gerar.bind(on_press=self.iniciar_geracao)
        layout.add_widget(self.btn_gerar)

        self.scroll = ScrollView()
        self.resultado_label = Label(
            text="Aguardando...",
            size_hint_y=None,
            text_size=(Window.width - 60, None),
            halign="left",
            valign="top",
            color=(0.9, 0.9, 0.9, 1)
        )
        self.resultado_label.bind(texture_size=self.resultado_label.setter('size'))
        self.scroll.add_widget(self.resultado_label)
        layout.add_widget(self.scroll)
        return layout

    def iniciar_geracao(self, instance):
        self.btn_gerar.disabled = True
        self.btn_gerar.text = "Processando..."
        self.resultado_label.text = "O servidor está processando, aguarde (pode levar até 90s)..."
        threading.Thread(target=self.fazer_requisicao, daemon=True).start()

    def fazer_requisicao(self):
        ideia = self.ideia_input.text
        try:
            res = requests.post(BACKEND_URL, json={"ideia": ideia}, timeout=90)
            data = res.json()
            if data.get("success"):
                msg = (f"OK IA: {data.get('api_llm')} | Video: {data.get('api_video')}\n\n"
                       f"{data.get('texto')}\n\nLink do Vídeo:\n{data.get('video_url')}")
            else:
                msg = f"ERRO: {data.get('error')}"
        except Exception:
            msg = "Erro de conexão com o servidor ou tempo esgotado."
        Clock.schedule_once(lambda dt: self.atualizar_tela(msg))

    def atualizar_tela(self, mensagem):
        self.resultado_label.text = mensagem
        self.btn_gerar.disabled = False
        self.btn_gerar.text = "Gerar Vídeo"

if __name__ == '__main__':
    TtokenApp().run()
