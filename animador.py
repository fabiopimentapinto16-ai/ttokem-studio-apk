"""Animador local do Ttoken: transforma UMA imagem em um GIF com movimento de camera
(efeito Ken Burns: zoom e pan suaves). Python puro + Pillow, sem ffmpeg, sem internet,
sem GPU: roda dentro do APK e no Pydroid 3.

Honestidade: isto NAO e video gerado por IA; e a imagem com camera em movimento. O que a IA
(Colab/APIs) faz e criar a imagem; aqui so damos vida a ela, de graca e offline.
"""
import hashlib
import io
import os
import time

from PIL import Image

ESTILOS = ("zoom_in", "zoom_out", "pan_dir", "pan_esq")


def _suave(t):
    """smoothstep: comeca e termina devagar (movimento de camera natural)."""
    return t * t * (3.0 - 2.0 * t)


def _caixa(estilo, t, w, h):
    """Retangulo (x0,y0,x1,y1), em float, que a 'camera' enquadra no instante t (0..1)."""
    s = _suave(t)
    if estilo == "zoom_out":
        z = 1.18 - 0.18 * s
        cx, cy = w / 2.0, h / 2.0
    elif estilo in ("pan_dir", "pan_esq"):
        z = 1.15
        folga = w - w / z
        cx = w / z / 2.0 + folga * (s if estilo == "pan_dir" else 1.0 - s)
        cy = h / 2.0
    else:                                           # zoom_in
        z = 1.0 + 0.18 * s
        cx, cy = w / 2.0, h / 2.0
    cw, ch = w / z, h / z
    return (cx - cw / 2.0, cy - ch / 2.0, cx + cw / 2.0, cy + ch / 2.0)


def escolher_estilo(chave=""):
    """Estilo repetivel a partir de um texto (o mesmo prompt gera a mesma camera)."""
    n = int(hashlib.md5((chave or str(time.time())).encode("utf-8")).hexdigest(), 16)
    return ESTILOS[n % len(ESTILOS)]


def animar(origem, saida, duracao=6, fps=12, lado=480, estilo="auto", semente=""):
    """origem: caminho ou bytes da imagem. Grava `saida` (.gif) e devolve o caminho.
    Lanca ValueError se a imagem nao abrir."""
    try:
        img = Image.open(io.BytesIO(origem) if isinstance(origem, (bytes, bytearray))
                         else origem)
        img = img.convert("RGB")
    except Exception as e:
        raise ValueError("nao consegui abrir a imagem: %s" % e)
    duracao = max(1, min(int(duracao), 12))
    fps = max(4, min(int(fps), 20))
    lado = max(128, min(int(lado), 720))
    if estilo == "auto" or estilo not in ESTILOS:
        estilo = escolher_estilo(semente)
    # quadrado central: corta o excesso do lado maior e limita o tamanho (memoria no celular)
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    base = min(m, lado * 2)                          # folga para o zoom nao perder nitidez
    if base != m:
        img = img.resize((base, base), Image.LANCZOS)
    n = duracao * fps
    quadros = []
    for i in range(n):
        t = i / float(n - 1) if n > 1 else 0.0
        box = _caixa(estilo, t, base, base)
        quadros.append(img.transform((lado, lado), Image.EXTENT, box, Image.BILINEAR))
    pasta = os.path.dirname(os.path.abspath(saida))
    os.makedirs(pasta, exist_ok=True)
    tmp = saida + ".tmp"
    quadros[0].save(tmp, format="GIF", save_all=True, append_images=quadros[1:],
                    duration=int(1000 / fps), loop=0, optimize=False)
    os.replace(tmp, saida)
    return saida
