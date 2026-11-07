"""TAESD (VAE leve, 2,45 M pesos) em numpy puro: so o DECODIFICADOR (latente -> imagem), como o DecoderTiny do diffusers.

decodificar(z, P)   z (4,H,W) (scaling_factor = 1.0) -> (3, 8H, 8W) em [-1, 1]
Estrutura (pesos decoder.layers.N): clamp tanh(x/3)*3 | conv 4->64 + ReLU | [3 blocos, x2, conv] x3 | 1 bloco | conv 64->3 | *2-1
Bloco: conv-ReLU-conv-ReLU-conv + entrada, depois ReLU. Subida: vizinho mais proximo.
STATUS: igualdade com o torch so se prova no Colab (colab_sdxs_validar.py).
"""
import numpy as np

try:
    import motor_cubico as mc
except ImportError:
    from . import motor_cubico as mc


def _relu(x):
    return np.maximum(x, 0.0, out=x) if x.dtype == np.float32 else np.maximum(x, 0.0)


def _conv(x, P, nome):
    b = P(nome + ".bias") if P.tem(nome + ".bias") else None
    return mc.conv2d(x, P(nome + ".weight"), b, pad=1)


def _bloco(x, P, p):
    h = _relu(_conv(x, P, p + ".conv.0"))
    h = _relu(_conv(h, P, p + ".conv.2"))
    h = _conv(h, P, p + ".conv.4")
    return _relu(h + x)


def decodificar(z, P, estrutura=(3, 3, 3, 1)):
    h = (np.tanh(np.asarray(z, np.float32) / 3.0) * 3.0).astype(np.float32)
    h = _relu(_conv(h, P, "decoder.layers.0"))
    idx = 2
    for bi, nb in enumerate(estrutura):
        for _ in range(nb):
            h = _bloco(h, P, "decoder.layers.%d" % idx)
            idx += 1
        if bi < len(estrutura) - 1:
            h = mc.sobe2x(h)
            idx += 1                                           # a camada de subida nao tem pesos
        h = _conv(h, P, "decoder.layers.%d" % idx)
        idx += 1
    return (h * 2.0 - 1.0).astype(np.float32)
