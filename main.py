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
        layout = BoxLayout(orientation='vertical', padding=25, spacing=15)

        # Título
        layout.add_widget(Label(
            text="Ttoken Studio",
            font_size='26sp',
            color=(0, 0.82, 0.62, 1),
            size_hint_y=None,
            height=55,
            bold=True
        ))

        # Campo de ideia
        self.ideia_input = TextInput(
            hint_text="Digite sua ideia de vídeo...",
            multiline=True,
            size_hint_y=None,
            height=100,
            background_color=(0.08, 0.08, 0.12, 1),
            foreground_color=(0.95, 0.95, 0.95, 1),
            padding=[12, 12]
        )
        layout.add_widget(self.ideia_input)

        # Botão Gerar
        self.btn_gerar = Button(
            text="Gerar Vídeo + Legenda",
            size_hint_y=None,
            height=55,
            background_color=(0, 0.82, 0.62, 1),
            background_normal='',
            font_size='17sp',
            bold=True
        )
        self.btn_gerar.bind(on_press=self.iniciar_geracao)
        layout.add_widget(self.btn_gerar)

        # Área de resultado
        self.scroll = ScrollView()
        self.resultado_label = Label(
            text="Aguardando sua ideia...",
            size_hint_y=None,
            text_size=(Window.width - 50, None),
            halign="left",
            valign="top",
            color=(0.9, 0.9, 0.9, 1),
            padding=[10, 10]
        )
        self.resultado_label.bind(texture_size=self.resultado_label.setter('size'))
        self.scroll.add_widget(self.resultado_label)
        layout.add_widget(self.scroll)

        return layout

    def iniciar_geracao(self, instance):
        ideia = self.ideia_input.text.strip()
        if not ideia:
            self.resultado_label.text = "Digite uma ideia primeiro."
            return

        self.btn_gerar.disabled = True
        self.btn_gerar.text = "Processando..."
        self.resultado_label.text = "Comprimindo prompt com Ttokem + gerando vídeo...\nIsso pode levar até 90 segundos."
        threading.Thread(target=self.fazer_requisicao, args=(ideia,), daemon=True).start()

    def fazer_requisicao(self, ideia):
        try:
            res = requests.post(
                BACKEND_URL,
                json={"ideia": ideia},
                timeout=100
            )
            data = res.json()

            if data.get("success"):
                msg = (
                    f"✅ Gerado com sucesso!\n\n"
                    f"IA usada: {data.get('api_llm', 'N/A')}\n"
                    f"Vídeo: {data.get('api_video', 'N/A')}\n\n"
                    f"📝 Texto / Legenda:\n{data.get('texto', '')}\n\n"
                    f"🎬 Link do Vídeo:\n{data.get('video_url', '')}\n\n"
                    f"→ Agora você pode copiar a legenda e postar manualmente na rede que quiser."
                )
            else:
                msg = f"❌ Erro: {data.get('error', 'Erro desconhecido')}"
        except Exception as e:
            msg = f"❌ Erro de conexão ou timeout.\nDetalhe: {str(e)}"

        Clock.schedule_once(lambda dt: self.atualizar_tela(msg))

    def atualizar_tela(self, mensagem):
        self.resultado_label.text = mensagem
        self.btn_gerar.disabled = False
        self.btn_gerar.text = "Gerar Vídeo + Legenda"

if __name__ == '__main__':
    TtokenApp().run()
