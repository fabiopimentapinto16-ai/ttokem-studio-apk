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


def gerar(prompt, saida, lado=256, iteracoes=40):
    """Grava o PNG em `saida` (de forma atomica) e devolve o caminho."""
    dados = gerar_bytes(prompt, lado, iteracoes)
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    tmp = saida + ".tmp"
    with open(tmp, "wb") as f:
        f.write(dados)
    os.replace(tmp, saida)
    return saida
