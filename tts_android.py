"""Narracao offline pela voz do Android (TextToSpeech -> WAV). So roda no celular (pyjnius).
Sem TTS instalado/idioma baixado, devolve None e o video sai so com legenda."""
import os
import threading


def disponivel():
    try:
        from jnius import autoclass  # noqa: F401
        return True
    except Exception:
        return False


def sintetizar(textos, pasta, idioma="pt-BR", espera=20):
    """Gera um WAV por texto em `pasta`. Devolve a lista de caminhos (None onde falhou)."""
    try:
        from jnius import PythonJavaClass, autoclass, java_method
    except Exception:
        return [None] * len(textos)
    pronto = threading.Event()
    estado = {"ok": False}

    class _Init(PythonJavaClass):
        __javainterfaces__ = ["android/speech/tts/TextToSpeech$OnInitListener"]
        __javacontext__ = "app"

        @java_method("(I)V")
        def onInit(self, status):
            estado["ok"] = (status == 0)
            pronto.set()

    saidas = [None] * len(textos)
    try:
        PA = autoclass("org.kivy.android.PythonActivity")
        TTS = autoclass("android.speech.tts.TextToSpeech")
        Locale = autoclass("java.util.Locale")
        File = autoclass("java.io.File")
        init = _Init()
        tts = TTS(PA.mActivity, init)
        if not pronto.wait(espera) or not estado["ok"]:
            return saidas
        partes = idioma.replace("_", "-").split("-")
        tts.setLanguage(Locale(partes[0], partes[1]) if len(partes) > 1 else Locale(partes[0]))
        os.makedirs(pasta, exist_ok=True)
        for i, t in enumerate(textos):
            if not (t or "").strip():
                continue
            cam = os.path.join(pasta, "narra_%02d.wav" % i)
            if os.path.exists(cam):
                os.remove(cam)
            r = tts.synthesizeToFile(t, None, File(cam), "ttk%d" % i)
            if r != 0:
                continue
            for _ in range(int(espera * 10)):              # espera o arquivo ficar pronto
                if os.path.exists(cam) and os.path.getsize(cam) > 1000:
                    break
                threading.Event().wait(0.1)
            if os.path.exists(cam) and os.path.getsize(cam) > 1000:
                saidas[i] = cam
        try:
            tts.shutdown()
        except Exception:
            pass
    except Exception:
        pass
    return saidas
