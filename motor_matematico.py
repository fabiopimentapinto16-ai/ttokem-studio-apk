"""Motor MATEMATICO do Ttoken: gera uma imagem so com Python puro (biblioteca padrao: hashlib,
math, zlib, struct). Sem Pillow, sem numpy, sem torch, sem internet, sem GPU.

Como funciona: o texto vira uma semente (hash). A semente escolhe o fractal de Julia, a paleta
(cores por palavra-chave: fogo, mar, noite, neon, floresta...) e o enquadramento. O mesmo texto
sempre gera a mesma imagem.

Honestidade: isto e ARTE GENERATIVA (matematica), nao uma foto do que o texto descreve.
Uma IA de verdade (SDXL no Colab, APIs) entende o prompt; este motor e o ultimo recurso que
nunca falha e funciona offline em qualquer celular.
"""
import hashlib
import math
import os
import struct
import zlib

# palavra-chave -> (a, b, c, d) da paleta cosseno  cor(t) = a + b*cos(2*pi*(c*t + d))
_BASE = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
PALETAS = (
    (("fogo", "fire", "lava", "sol", "sun", "quente", "vermelho", "red", "raposa", "fox"),
     ((0.5, 0.3, 0.2), (0.5, 0.4, 0.3), (1.0, 0.8, 0.6), (0.0, 0.1, 0.2))),
    (("mar", "oceano", "ocean", "agua", "water", "azul", "blue", "chuva", "rain", "gelo", "neve", "snow"),
     ((0.2, 0.4, 0.6), (0.3, 0.4, 0.4), (0.8, 0.9, 1.0), (0.5, 0.6, 0.7))),
    (("floresta", "forest", "verde", "green", "natureza", "folha", "planta"),
     ((0.2, 0.5, 0.2), (0.3, 0.4, 0.3), (0.7, 1.0, 0.7), (0.3, 0.1, 0.4))),
    (("neon", "cyber", "carro", "car", "futuro", "future", "cidade", "city"),
     ((0.5, 0.2, 0.6), (0.5, 0.5, 0.5), (1.0, 1.0, 0.7), (0.8, 0.0, 0.3))),
    (("noite", "night", "escuro", "dark", "lua", "moon", "espaco", "space", "estrela", "star"),
     ((0.1, 0.1, 0.25), (0.35, 0.3, 0.5), (1.0, 0.7, 1.0), (0.6, 0.3, 0.9))),
)


def semente(texto):
    return int(hashlib.sha256((texto or "ttoken").lower().encode("utf-8")).hexdigest(), 16)


def _paleta(texto, s):
    t = (texto or "").lower()
    for chaves, cores in PALETAS:
        if any(c in t for c in chaves):
            return cores
    r = [((s >> (8 * i)) & 255) / 255.0 for i in range(12)]    # sem palavra-chave: pela semente
    return ((0.3 + 0.4 * r[0], 0.3 + 0.4 * r[1], 0.3 + 0.4 * r[2]),
            (0.4 + 0.2 * r[3], 0.4 + 0.2 * r[4], 0.4 + 0.2 * r[5]),
            (0.6 + 0.8 * r[6], 0.6 + 0.8 * r[7], 0.6 + 0.8 * r[8]),
            (r[9], r[10], r[11]))


def _png(larg, alt, linhas_rgb):
    """PNG RGB 8 bits escrito a mao (zlib + struct)."""
    def bloco(tipo, dados):
        c = struct.pack(">I", len(dados)) + tipo + dados
        return c + struct.pack(">I", zlib.crc32(tipo + dados) & 0xFFFFFFFF)
    cru = b"".join(b"\x00" + bytes(l) for l in linhas_rgb)
    return (b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", struct.pack(">IIBBBBB", larg, alt, 8, 2, 0, 0, 0))
            + bloco(b"IDAT", zlib.compress(cru, 6)) + bloco(b"IEND", b""))


def gerar_bytes(prompt, lado=256, iteracoes=40):
    """Devolve os bytes de um PNG lado x lado."""
    lado = max(32, min(int(lado), 720))
    s = semente(prompt)
    a, b, c, d = _paleta(prompt, s)
    ang = ((s >> 16) % 3600) / 3600.0 * 2 * math.pi              # constante de Julia: c = 0.7885*e^(i*ang)
    raio = 0.70 + ((s >> 40) % 200) / 1000.0
    cr, ci = raio * math.cos(ang), raio * math.sin(ang)
    zoom = 1.25 + ((s >> 60) % 100) / 200.0
    desl = (((s >> 70) % 200) - 100) / 1000.0
    metade = lado / 2.0
    esc = 2.0 * zoom / lado
    log2 = math.log(2.0)
    linhas = []
    for py in range(lado):
        y0 = (py - metade) * esc + desl
        linha = bytearray()
        vy = ((py - metade) / metade) ** 2
        for px in range(lado):
            x, y = (px - metade) * esc, y0
            i = 0
            while i < iteracoes and x * x + y * y <= 16.0:
                x, y = x * x - y * y + cr, 2.0 * x * y + ci
                i += 1
            if i >= iteracoes:
                t = 0.02 + 0.02 * math.sin((px + py) * 0.05)     # miolo: tom escuro
            else:                                                 # cor suave, sem faixas
                t = (i + 1 - math.log(math.log(x * x + y * y) / 2.0 + 1e-9) / log2) / iteracoes
            v = 1.0 - 0.45 * (vy + ((px - metade) / metade) ** 2) / 2.0   # vinheta
            for k in range(3):
                cor = a[k] + b[k] * math.cos(6.2831853 * (c[k] * t * 2.5 + d[k]))
                linha.append(max(0, min(255, int(255 * cor * v))))
        linhas.append(linha)
    return _png(lado, lado, linhas)


# ------------------------------------------------------------------ GIF animado (tudo a mao)
def _lzw(indices, bits=8):
    """Compressao LZW do formato GIF (indices: bytes de 0..255)."""
    clear, fim = 1 << bits, (1 << bits) + 1
    saida, acc, nacc = bytearray(), 0, 0
    tabela, prox, tam = {}, fim + 1, bits + 1

    def emitir(codigo, n):
        nonlocal acc, nacc
        acc |= codigo << nacc
        nacc += n
        while nacc >= 8:
            saida.append(acc & 255)
            acc >>= 8
            nacc -= 8

    emitir(clear, tam)
    prefixo = indices[0]
    for b in indices[1:]:
        chave = (prefixo << 8) | b
        achado = tabela.get(chave)
        if achado is not None:
            prefixo = achado
            continue
        emitir(prefixo, tam)
        if prox < 4096:
            tabela[chave] = prox
            if prox == (1 << tam):
                tam += 1
            prox += 1
        else:
            emitir(clear, tam)
            tabela, prox, tam = {}, fim + 1, bits + 1
        prefixo = b
    emitir(prefixo, tam)
    emitir(fim, tam)
    if nacc:
        saida.append(acc & 255)
    return bytes(saida)


def _gif(lado, paleta, quadros, atraso_cs):
    cab = b"GIF89a" + struct.pack("<HHBBB", lado, lado, 0xF7, 0, 0) + bytes(paleta)
    cab += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"          # repete para sempre
    partes = [cab]
    for q in quadros:
        partes.append(b"\x21\xF9\x04\x00" + struct.pack("<H", atraso_cs) + b"\x00\x00")
        partes.append(b"\x2C" + struct.pack("<HHHHB", 0, 0, lado, lado, 0) + b"\x08")
        dados = _lzw(q)
        for i in range(0, len(dados), 255):
            bloco = dados[i:i + 255]
            partes.append(bytes([len(bloco)]) + bloco)
        partes.append(b"\x00")
    partes.append(b"\x3B")
    return b"".join(partes)


def _indices(lado, cr, ci, esc, desl, iteracoes):
    metade = lado / 2.0
    log2 = math.log(2.0)
    buf = bytearray()
    for py in range(lado):
        y0 = (py - metade) * esc + desl
        for px in range(lado):
            x, y = (px - metade) * esc, y0
            i = 0
            while i < iteracoes and x * x + y * y <= 16.0:
                x, y = x * x - y * y + cr, 2.0 * x * y + ci
                i += 1
            if i >= iteracoes:
                buf.append(0)
            else:
                t = (i + 1 - math.log(math.log(x * x + y * y) / 2.0 + 1e-9) / log2) / iteracoes
                buf.append(1 + max(0, min(253, int(t * 254))))
    return bytes(buf)


def gerar_gif_bytes(prompt, lado=160, quadros=24, fps=12, iteracoes=36):
    """GIF animado (o fractal respira: zoom e constante oscilam e o loop fecha sem salto).
    Tudo em Python puro; nao precisa de Pillow."""
    lado = max(32, min(int(lado), 400))
    quadros = max(4, min(int(quadros), 60))
    s = semente(prompt)
    a, b, c, d = _paleta(prompt, s)
    paleta = bytearray(3 * 256)
    paleta[0:3] = bytes((6, 6, 12))
    for n in range(1, 256):
        t = (n - 1) / 254.0
        for k in range(3):
            cor = a[k] + b[k] * math.cos(6.2831853 * (c[k] * t * 2.5 + d[k]))
            paleta[3 * n + k] = max(0, min(255, int(255 * cor)))
    ang = ((s >> 16) % 3600) / 3600.0 * 2 * math.pi
    raio = 0.70 + ((s >> 40) % 200) / 1000.0
    zoom = 1.25 + ((s >> 60) % 100) / 200.0
    desl = (((s >> 70) % 200) - 100) / 1000.0
    lista = []
    for f in range(quadros):
        fase = 2 * math.pi * f / quadros
        a2 = ang + 0.12 * math.sin(fase)
        z = zoom * (1.0 + 0.10 * math.sin(fase + 1.0))
        lista.append(_indices(lado, raio * math.cos(a2), raio * math.sin(a2), 2.0 * z / lado,
                              desl, iteracoes))
    return _gif(lado, paleta, lista, max(2, int(100 / fps)))


def gerar_gif(prompt, saida, **kw):
    dados = gerar_gif_bytes(prompt, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    tmp = saida + ".tmp"
    with open(tmp, "wb") as f:
        f.write(dados)
    os.replace(tmp, saida)
    return saida


def gerar(prompt, saida, lado=256, iteracoes=40):
    """Grava o PNG em `saida` (de forma atomica) e devolve o caminho."""
    dados = gerar_bytes(prompt, lado, iteracoes)
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    tmp = saida + ".tmp"
    with open(tmp, "wb") as f:
        f.write(dados)
    os.replace(tmp, saida)
    return saida


if __name__ == "__main__":                 # Pydroid 3 / Termux:  python motor_matematico.py "sua ideia"
    import sys
    ideia = " ".join(sys.argv[1:]) or "raposa vermelha na neve a noite"
    pasta = "/sdcard/Download/ttoken" if os.path.isdir("/sdcard/Download") else "."
    base = os.path.join(pasta, "ttoken_" + hashlib.md5(ideia.encode()).hexdigest()[:8])
    print("PNG:", gerar(ideia, base + ".png"))
    print("GIF:", gerar_gif(ideia, base + ".gif"))
