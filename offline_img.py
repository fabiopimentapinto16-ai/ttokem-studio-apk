"""Imagem 100% offline no celular (SDXS 1 passo, numpy puro). Modelos: Download/sdxs_cubo."""
import os
import sys
import time

_AQUI = os.path.dirname(os.path.abspath(__file__))
_MOT = os.path.join(_AQUI, "motor_offline")


def _completa(p):
    f = lambda n: os.path.isfile(os.path.join(p, n))
    return (f("sdxs_texto_q8.ttq") and f("sdxs_vae_fp16.ttq") and (f("sdxs_unet_q8.ttq") or f("sdxs_unet_curva.ttq"))
            and (f("tokenizer.json") or os.path.isfile(os.path.join(p, "tok", "tokenizer.json"))))


def achar_pasta():
    """Procura a pasta COMPLETA (3 .ttq + tokenizer) nos lugares comuns do celular."""
    base = ["/storage/emulated/0/Download", "/sdcard/Download", "/storage/emulated/0",
            os.path.expanduser("~"), os.getcwd()]
    for b in base:
        for sub in ("sdxs_cubo", "sdxs", ""):
            p = os.path.join(b, sub) if sub else b
            if _completa(p):
                return p
    return None


def tem_acesso():
    """Android 11+: ler arquivos da pasta Download exige 'acesso a todos os arquivos'. Fora do Android: True."""
    try:
        from jnius import autoclass
        if autoclass("android.os.Build$VERSION").SDK_INT < 30:
            return True
        return bool(autoclass("android.os.Environment").isExternalStorageManager())
    except Exception:
        return True


def abrir_ajustes_acesso():
    """Abre a tela do Android onde o usuario liga 'Permitir acesso a todos os arquivos' para este app."""
    try:
        from jnius import autoclass
        PA = autoclass("org.kivy.android.PythonActivity")
        Intent = autoclass("android.content.Intent")
        Settings = autoclass("android.provider.Settings")
        Uri = autoclass("android.net.Uri")
        act = PA.mActivity
        try:
            it = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
            it.setData(Uri.parse("package:" + act.getPackageName()))
            act.startActivity(it)
        except Exception:
            act.startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
        return True
    except Exception:
        return False


PORTA_RAPIDA = 8765


def servidor_ativo():
    """True se o motor rapido (sdxs_servidor.py rodando no Pydroid) esta ligado neste celular."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:%d/" % PORTA_RAPIDA, timeout=1.5) as r:
            return b"ttoken-sdxs" in r.read(64)
    except Exception:
        return False


def _gerar_no_servidor(prompt, saida, semente, lado):
    import json
    import urllib.request
    req = urllib.request.Request("http://127.0.0.1:%d/imagem" % PORTA_RAPIDA, method="POST",
                                 data=json.dumps({"prompt": prompt, "semente": semente, "lado": lado}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        dados = r.read()
    if not dados.startswith(b"\x89PNG"):
        raise RuntimeError("motor rapido respondeu: %s" % dados[:80].decode("utf-8", "replace"))
    with open(saida, "wb") as f:
        f.write(dados)
    return saida


def disponivel():
    """(ok, motivo). ok=True se o motor rapido esta ligado, ou se numpy e os modelos estao no celular."""
    if servidor_ativo():
        return True, ""
    try:
        import numpy  # noqa: F401
    except Exception:
        return False, "numpy nao esta neste app"
    if not tem_acesso():
        return False, "sem_acesso"
    if achar_pasta() is None:
        return False, "pasta sdxs_cubo nao encontrada em Download"
    return True, ""


def gerar_png(prompt, saida, semente=None, lado=256, log=print, look=1.0):
    """Gera a imagem e salva em PNG. Devolve o caminho. look = forca do acabamento de cinema (0 = desliga)."""
    _gerar_png_bruto(prompt, saida, semente, lado, log)
    if look and look > 0:
        try:
            import look_cinema
            from PIL import Image
            look_cinema.aplicar(Image.open(saida), forca=look).save(saida)
        except Exception as e:
            log("acabamento de cinema nao aplicado: %s" % str(e)[:60])
    return saida


def _gerar_png_bruto(prompt, saida, semente, lado, log):
    if _MOT not in sys.path:
        sys.path.insert(0, _MOT)
    if semente is None:
        semente = int(time.time()) % 100000
    if servidor_ativo():
        log("motor rapido do Pydroid ligado")
        return _gerar_no_servidor(prompt, saida, semente, lado)
    log("motor rapido desligado: usando o motor do app (lento, uns 12 min)")
    pasta = achar_pasta()
    if pasta is None:
        raise RuntimeError("Nao achei a pasta sdxs_cubo em Download.")
    import pipeline_sdxs as ps
    from PIL import Image
    if semente is None:
        semente = int(time.time()) % 100000
    img = ps.gerar(pasta, prompt, semente=semente, lado=lado, log=log)
    Image.fromarray(img).save(saida)
    return saida
