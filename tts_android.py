"""Narracao offline pela voz do Android (TextToSpeech -> WAV). So roda no celular (pyjnius).
Sem TTS instalado/idioma baixado, devolve None e o video sai so com legenda."""
import os
import threading

ULTIMO_ERRO = ""       # por que a voz falhou (mostrado no chat do app para diagnostico)


def disponivel():
    try:
        from jnius import autoclass  # noqa: F401
        return True
    except Exception:
        return False


def sintetizar(textos, pasta, idioma="pt-BR", espera=20):
    """Gera um WAV por texto em `pasta`. Devolve a lista de caminhos (None onde falhou)."""
    global ULTIMO_ERRO
    ULTIMO_ERRO = ""
    try:
        from jnius import PythonJavaClass, autoclass, java_method
    except Exception as e:
        ULTIMO_ERRO = "pyjnius indisponivel (%s)" % str(e)[:60]
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
        if not pronto.wait(espera):
            ULTIMO_ERRO = "o Android nao respondeu ao iniciar a voz em %ds (sem motor de voz?)" % espera
            return saidas
        if not estado["ok"]:
            ULTIMO_ERRO = "o motor de voz do Android nao iniciou (instale/ative 'Sintese de voz do Google' em Ajustes > Idioma)"
            return saidas
        partes = idioma.replace("_", "-").split("-")
        res = tts.setLanguage(Locale(partes[0], partes[1]) if len(partes) > 1 else Locale(partes[0]))
        if res < 0:                                    # -1 = faltam os dados da voz, -2 = idioma nao suportado
            ULTIMO_ERRO = "voz '%s' indisponivel (codigo %d); usei a voz padrao do celular" % (idioma, res)
            tts.setLanguage(Locale.getDefault())
        os.makedirs(pasta, exist_ok=True)
        for i, t in enumerate(textos):
            if not (t or "").strip():
                continue
            cam = os.path.join(pasta, "narra_%02d.wav" % i)
            if os.path.exists(cam):
                os.remove(cam)
            r = tts.synthesizeToFile(t, None, File(cam), "ttk%d" % i)
            if r != 0:
                ULTIMO_ERRO = ULTIMO_ERRO or "synthesizeToFile falhou (codigo %d)" % r
                continue
            for _ in range(int(espera * 10)):              # espera o arquivo ficar pronto
                if os.path.exists(cam) and os.path.getsize(cam) > 1000:
                    break
                threading.Event().wait(0.1)
            if os.path.exists(cam) and os.path.getsize(cam) > 1000:
                saidas[i] = cam
            else:
                ULTIMO_ERRO = ULTIMO_ERRO or "a voz nao gerou o arquivo de audio da cena %d" % (i + 1)
        try:
            tts.shutdown()
        except Exception:
            pass
    except Exception as e:
        ULTIMO_ERRO = ULTIMO_ERRO or "erro na voz: %s" % str(e)[:80]
    return saidas
