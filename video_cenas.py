"""Video de cenas offline: varias imagens -> GIF com camera em movimento, transicao suave e legenda.
Python puro + Pillow (sem ffmpeg). A narracao vai a parte (WAV) e toca junto no app.
Honestidade: nao e video gerado quadro a quadro; sao cenas (imagens) com camera e legenda."""
import os
import wave

from PIL import Image, ImageDraw, ImageFont

import animador


def _fonte(tam):
    for nome in ("DejaVuSans-Bold.ttf", "/system/fonts/Roboto-Bold.ttf", "/system/fonts/DroidSans-Bold.ttf",
                 "/system/fonts/Roboto-Regular.ttf", "/system/fonts/NotoSansCJK-Regular.ttc",
                 "/system/fonts/NotoSansCJKjp-Regular.otf", "/system/fonts/NotoSansArabic-Regular.ttf",
                 "/system/fonts/NotoSansDevanagari-Regular.ttf", "/system/fonts/DroidSansFallback.ttf"):
        try:
            return ImageFont.truetype(nome, tam)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=tam)
    except Exception:
        return ImageFont.load_default()


def _quebrar(draw, texto, fonte, largura):
    linhas, atual = [], ""
    for pal in texto.split():
        t = (atual + " " + pal).strip()
        if draw.textlength(t, font=fonte) <= largura or not atual:
            atual = t
        else:
            linhas.append(atual)
            atual = pal
    if atual:
        linhas.append(atual)
    return linhas[:4]


def legendar(img, texto):
    """Desenha a legenda embaixo, com faixa escura translucida."""
    if not texto:
        return img
    img = img.convert("RGBA")
    w, h = img.size
    fonte = _fonte(max(14, w // 18))
    cam = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(cam)
    linhas = _quebrar(d, texto, fonte, w * 0.9)
    alt = fonte.size if hasattr(fonte, "size") else 16
    bloco = int(len(linhas) * alt * 1.25 + alt * 0.8)
    d.rectangle((0, h - bloco, w, h), fill=(0, 0, 0, 150))
    y = h - bloco + int(alt * 0.4)
    for ln in linhas:
        d.text(((w - d.textlength(ln, font=fonte)) / 2, y), ln, font=fonte, fill=(255, 255, 255, 255))
        y += int(alt * 1.25)
    return Image.alpha_composite(img, cam).convert("RGB")


def duracao_wav(caminho):
    try:
        with wave.open(caminho, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def juntar_wavs(wavs, saida, folga=0.4):
    """Concatena WAVs (mesmo formato) com uma pausa curta; devolve a lista de duracoes (s) por cena.
    Cena sem WAV valido vira silencio de 3 s no formato dos demais."""
    base = None
    for c in wavs:
        if c and os.path.isfile(c):
            try:
                with wave.open(c, "rb") as w:
                    base = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                break
            except Exception:
                pass
    if base is None:
        return [3.0] * len(wavs)
    ch, sw, fr = base
    sil = lambda s: b"\x00" * int(s * fr) * ch * sw
    durs = []
    with wave.open(saida, "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(sw)
        o.setframerate(fr)
        for c in wavs:
            dados = b""
            if c and os.path.isfile(c):
                try:
                    with wave.open(c, "rb") as w:
                        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) == base:
                            dados = w.readframes(w.getnframes())
                except Exception:
                    pass
            if not dados:
                dados = sil(3.0 - folga)
            dados += sil(folga)
            o.writeframes(dados)
            durs.append(len(dados) / float(fr * ch * sw))
    return durs


def montar(imagens, legendas, saida, duracoes=None, fps=10, lado=384, transicao=0.5, semente=""):
    """imagens: lista de caminhos; legendas: lista de textos (mesmo tamanho); duracoes: s por cena.
    Grava o GIF em `saida` e devolve (saida, duracao_total_s)."""
    if not imagens:
        raise ValueError("sem imagens")
    legendas = list(legendas) + [""] * (len(imagens) - len(legendas))
    duracoes = list(duracoes or []) + [4.0] * (len(imagens) - len(duracoes or []))
    fps = max(4, min(int(fps), 15))
    lado = max(128, min(int(lado), 512))
    quadros, ant = [], None
    for k, cam in enumerate(imagens):
        img = Image.open(cam).convert("RGB")
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
        base = min(m, lado * 2)
        if base != m:
            img = img.resize((base, base), Image.LANCZOS)
        estilo = animador.ESTILOS[(k + sum(map(ord, semente))) % len(animador.ESTILOS)]
        n = max(2, int(round(max(1.5, duracoes[k]) * fps)))
        cena = []
        for i in range(n):
            box = animador._caixa(estilo, i / float(n - 1), base, base)
            q = img.transform((lado, lado), Image.EXTENT, box, Image.BILINEAR)
            cena.append(legendar(q, legendas[k]))
        nt = min(int(transicao * fps), n // 2) if ant is not None else 0
        for i in range(nt):                               # fusao suave com o fim da cena anterior
            cena[i] = Image.blend(ant, cena[i], (i + 1) / float(nt + 1))
        ant = cena[-1]
        quadros.extend(q.quantize(colors=128, method=Image.FASTOCTREE) for q in cena)
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    tmp = saida + ".tmp"
    quadros[0].save(tmp, format="GIF", save_all=True, append_images=quadros[1:],
                    duration=int(1000 / fps), loop=0, optimize=False)
    os.replace(tmp, saida)
    return saida, len(quadros) / float(fps)
