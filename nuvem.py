"""Refino na nuvem (opcional): o celular faz o rascunho offline e, quando ha link do Colab e internet,
a GPU gratis refina so o detalhe (img2img). Sem link/internet/erro: devolve None e fica o rascunho."""
import base64
import json
import os
import urllib.request

FORCA_PADRAO = 0.7          # 0.7 = refaz bastante (corrige objeto que faltou); 0.4 = so afina o detalhe


def _base(url):
    return (url or "").strip().rstrip("/")


def pronto(url, timeout=4):
    """True se o Colab refinador responde."""
    u = _base(url)
    if not u.startswith("http"):
        return False
    try:
        req = urllib.request.Request(u + "/saude", headers={"ngrok-skip-browser-warning": "1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return b"ttoken-refinador" in r.read(80)
    except Exception:
        return False


def refinar(png, prompt, url, saida=None, forca=FORCA_PADRAO, timeout=120, chave=""):
    """Manda o rascunho (png) + prompt e grava a imagem refinada em `saida` (padrao: <png>_nuvem.png).
    Devolve o caminho, ou None se nao deu (o rascunho continua valendo)."""
    u = _base(url)
    if not pronto(u):
        return None
    try:
        with open(png, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        corpo = json.dumps({"prompt": prompt, "imagem_b64": b64, "forca": forca}).encode("utf-8")
        req = urllib.request.Request(u + "/refinar", data=corpo, method="POST", headers={
            "Content-Type": "application/json", "X-Chave": chave or "", "ngrok-skip-browser-warning": "1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            dados = r.read()
        if not dados.startswith(b"\x89PNG"):
            return None
        saida = saida or (os.path.splitext(png)[0] + "_nuvem.png")
        with open(saida, "wb") as f:
            f.write(dados)
        return saida
    except Exception:
        return None
