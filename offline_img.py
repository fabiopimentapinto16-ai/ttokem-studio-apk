"""Imagem 100% offline no celular (SDXS 1 passo, numpy puro). Modelos: Download/sdxs_cubo."""
import os
import sys
import time

_AQUI = os.path.dirname(os.path.abspath(__file__))
_MOT = os.path.join(_AQUI, "motor_offline")


def achar_pasta():
    """Procura a pasta com sdxs_unet_q8.ttq nos lugares comuns do celular."""
    base = ["/storage/emulated/0/Download", "/sdcard/Download", "/storage/emulated/0",
            os.path.expanduser("~"), os.getcwd()]
    for b in base:
        for sub in ("sdxs_cubo", "sdxs", ""):
            p = os.path.join(b, sub) if sub else b
            for ar in ("sdxs_unet_q8.ttq", "sdxs_unet_curva.ttq"):
                if os.path.isfile(os.path.join(p, ar)):
                    return p
    return None


def disponivel():
    """(ok, motivo). ok=True se numpy existe e os modelos estao no celular."""
    try:
        import numpy  # noqa: F401
    except Exception:
        return False, "numpy nao esta neste app"
    if achar_pasta() is None:
        return False, "pasta sdxs_cubo nao encontrada em Download"
    return True, ""


def gerar_png(prompt, saida, semente=None, lado=256, log=print):
    """Gera a imagem e salva em PNG. Devolve o caminho."""
    if _MOT not in sys.path:
        sys.path.insert(0, _MOT)
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
