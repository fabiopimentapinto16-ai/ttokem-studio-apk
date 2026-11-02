# TTOKEN - QUANTOS BITS POR PESO CADA TECNICA GASTA PARA A MESMA FIDELIDADE (pesos reais do tiny-sd). Rode no Colab. Arquivo UNICO.
# Compara: base (o que o .ttq faz), entropia (codificador sem perda por cima), hadamard (rotacao + entropia).
# Mede SNR nos PESOS (proxy). Se uma tecnica ganha muito aqui, depois eu testo na IMAGEM (UNet) antes de adotar.
import sys, types, time, json
import numpy as np
_SRC = '"""Quantos BITS POR PESO cada tecnica gasta para chegar a mesma fidelidade (SNR nos pesos).\n\nTecnicas (todas com blocos 1 x g ao longo das colunas, escala fp16 por bloco, simetrico):\n  base      : inteiro de b bits, tamanho = b + 16/g bits/peso (o que o .ttq faz, com bloco menor)\n  entropia  : mesmos inteiros, mas contados pela ENTROPIA empirica (limite de um codificador Huffman/aritmetico)\n  hadamard  : rotacao de Hadamard (g x g) em cada bloco ANTES de quantizar (espalha valores extremos), + entropia\nA rotacao e ortogonal: a leitura aplica a inversa, custo O(g log g) por bloco. Medida = SNR nos pesos (proxy;\na fidelidade da IMAGEM continua sendo decidida pelo teste com UNet).\n"""\nimport numpy as np\n\n\ndef hadamard(n):\n    h = np.ones((1, 1), np.float32)\n    while h.shape[0] < n:\n        h = np.block([[h, h], [h, -h]])\n    return h / np.sqrt(n)\n\n\ndef _blocos(m, g):\n    R, C = m.shape\n    Cp = (C + g - 1) // g * g\n    p = np.zeros((R, Cp), np.float32); p[:, :C] = m\n    return p.reshape(R, Cp // g, g), (R, C)\n\n\ndef quantizar_bloco(b, bits):\n    """b: (..., g). Devolve (inteiros, reconstrucao). Escala = absmax/nivel, arredondada para fp16."""\n    nivel = 2 ** (bits - 1) - 1\n    mx = np.abs(b).max(-1, keepdims=True)\n    esc = np.where(mx == 0, 1.0, mx / nivel).astype(np.float16).astype(np.float32)\n    q = np.clip(np.rint(b / esc), -nivel - 1, nivel)\n    return q.astype(np.int16), q * esc\n\n\ndef entropia_bits(q):\n    """Entropia empirica (bits/simbolo) dos inteiros q."""\n    v, c = np.unique(q, return_counts=True)\n    p = c / c.sum()\n    return float(-(p * np.log2(p)).sum())\n\n\ndef avaliar(m, bits, g, metodo):\n    """Devolve (bits_por_peso, ruido, sinal) para uma matriz m (R,C)."""\n    b, (R, C) = _blocos(m, g)\n    rot = metodo == "hadamard"\n    if rot:\n        H = hadamard(g)\n        b = b @ H.T\n    q, rec = quantizar_bloco(b, bits)\n    if rot:\n        rec = rec @ H\n    rec, b = rec.reshape(R, -1)[:, :C], b.reshape(R, -1)[:, :C]\n    ruido, sinal = float(((m - rec) ** 2).sum()), float((m ** 2).sum())\n    custo = (bits if metodo == "base" else entropia_bits(q)) + 16.0 / g\n    return custo, ruido, sinal\n\n\ndef varrer(tensores, grade, metodos=("base", "entropia", "hadamard")):\n    """tensores: dict nome->ndarray (>=2D). grade: lista de (bits, g).\n    Devolve linhas (metodo, bits, g, bits_por_peso, snr_db, vezes_menor_que_fp16)."""\n    linhas = []\n    for metodo in metodos:\n        for bits, g in grade:\n            tot_bits = tot_n = tot_r = tot_s = 0.0\n            for arr in tensores.values():\n                m = np.asarray(arr, np.float32).reshape(arr.shape[0], -1)\n                custo, r, s = avaliar(m, bits, g, metodo)\n                tot_bits += custo * m.size; tot_n += m.size; tot_r += r; tot_s += s\n            bpp = tot_bits / tot_n\n            snr = 10 * np.log10(tot_s / max(tot_r, 1e-30))\n            linhas.append((metodo, bits, g, bpp, snr, 16.0 / bpp))\n    return linhas\n'

ab = types.ModuleType("analise_bits"); sys.modules["analise_bits"] = ab; exec(_SRC, ab.__dict__)
import torch
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32, safety_checker=None)
T = {k: v.numpy() for k, v in pipe.unet.state_dict().items() if v.ndim >= 2 and v.numel() >= 256}
n = sum(v.size for v in T.values())
print("UNet: %d tensores 2D+, %.1f M pesos (fp16 = %.0f MB)" % (len(T), n / 1e6, n * 2 / 2**20))
grade = [(8, 16), (8, 32), (7, 32), (6, 32), (5, 32), (4, 32), (3, 32), (6, 64), (5, 64), (4, 64), (3, 64), (5, 16), (4, 16), (3, 16)]
t0 = time.time()
L = ab.varrer(T, grade)
print("tempo %.0f s\n" % (time.time() - t0))
print("%-9s %4s %4s %10s %8s %12s" % ("tecnica", "bits", "g", "bits/peso", "SNR dB", "vs fp16"))
for m, b, g, bpp, snr, x in sorted(L, key=lambda r: (r[0], -r[4])):
    mark = "  <== o que o .ttq usa hoje (8 bits, bloco 4x4=16)" if (m, b, g) == ("base", 8, 16) else ""
    print("%-9s %4d %4d %10.2f %8.1f %11.2fx%s" % (m, b, g, bpp, snr, x, mark))
json.dump([list(map(float, r[1:])) + [r[0]] for r in L], open("tinysd_bits.json", "w"))
print("\nMe mande este texto inteiro.")
