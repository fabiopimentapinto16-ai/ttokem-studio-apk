"""UNet do segmind/tiny-sd em numpy puro, montado com os blocos do motor_cubico.

Arquitetura (lida de tinysd_config.json e do cabecalho do .ttq):
  conv_in 4->320 | down0 (320, resnet+atencao, reduz) | down1 (640, resnet+atencao, reduz)
  | down2 (1280, resnet+atencao) | SEM bloco do meio | up0 (1280, 2x[resnet+atencao], sobe)
  | up1 (640, 2x, sobe) | up2 (320, 2x) | norm -> silu -> conv_out 320->4
  8 cabecas de atencao, GEGLU exato (erf), tempo: seno/cosseno 320 -> 1280.

Cada bloco carrega seus pesos UMA vez e processa todas as amostras (condicional e
incondicional) antes de soltar; assim o pico de RAM e (ativacoes + pesos de UM bloco).
STATUS: forma e fiacao testadas com pesos aleatorios; a igualdade com o torch so se prova
rodando validar_unet_tinysd contra tinysd_ref.npz (Colab).
"""
import numpy as np

try:
    import motor_cubico as mc
except ImportError:
    from . import motor_cubico as mc

CABECAS = 8


def erf(x):
    """erf vetorizado (Abramowitz-Stegun 7.1.26, erro < 1.5e-7)."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return (s * y).astype(np.float32)


def gelu_exato(x):
    return (0.5 * x * (1.0 + erf(x * np.float32(0.70710678)))).astype(np.float32)


class _Memo:
    """Carrega cada peso uma vez durante UM bloco (varias amostras reaproveitam)."""

    def __init__(self, P):
        self.P, self.m = P, {}

    def tem(self, nome):
        return self.P.tem(nome)

    def __call__(self, nome):
        if nome not in self.m:
            self.m[nome] = self.P(nome)
        return self.m[nome]


def transformer(x, ctx, P, p, cabecas=CABECAS):
    C, H, W = x.shape
    h = mc.group_norm(x, P(p + ".norm.weight"), P(p + ".norm.bias"), 32, eps=1e-6)
    h = mc._proj(h.reshape(C, H * W).T, P, p + ".proj_in")
    t = p + ".transformer_blocks.0"
    n1 = mc.layer_norm(h, P(t + ".norm1.weight"), P(t + ".norm1.bias"))
    h = h + mc._atn(n1, None, P, t + ".attn1", cabecas)
    n2 = mc.layer_norm(h, P(t + ".norm2.weight"), P(t + ".norm2.bias"))
    h = h + mc._atn(n2, ctx, P, t + ".attn2", cabecas)
    f = mc.layer_norm(h, P(t + ".norm3.weight"), P(t + ".norm3.bias"))
    f = mc.linear(f, P(t + ".ff.net.0.proj.weight"), P(t + ".ff.net.0.proj.bias"))
    a, gate = np.split(f, 2, axis=-1)
    f = mc.linear(a * gelu_exato(gate), P(t + ".ff.net.2.weight"), P(t + ".ff.net.2.bias"))
    h = mc._proj(h + f, P, p + ".proj_out")
    return (x + h.T.reshape(C, H, W)).astype(np.float32)


def _res(xs, tembs, P, p):
    M = _Memo(P)
    return [mc.bloco_resnet(x, te, M, p) for x, te in zip(xs, tembs)]


def _trf(xs, ctxs, P, p):
    M = _Memo(P)
    return [transformer(x, c, M, p) for x, c in zip(xs, ctxs)]


def _conv(xs, P, nome, stride=1, pad=1):
    M = _Memo(P)
    w, b = M(nome + ".weight"), M(nome + ".bias")
    return [mc.conv2d(x, w, b, stride=stride, pad=pad) for x in xs]


def unet_tinysd(xs, t, ctxs, P, captura=None):
    """xs: lista de latentes (4,H,W); ctxs: lista de (77,768), um por amostra; t: passo (numero).
    Devolve a lista de previsoes de ruido (4,H,W). `captura` (dict) recebe as ativacoes da
    amostra 0 com os mesmos nomes dos ganchos do script de exportacao."""
    cap = (lambda k, v: captura.__setitem__(k, np.array(v[0], np.float32))) if captura is not None \
        else (lambda k, v: None)
    temb = mc.embedding_tempo(float(t), 320, P, "time_embedding")
    cap("time_embedding", [temb])
    tembs = [temb] * len(xs)
    hs = _conv(xs, P, "conv_in")
    cap("conv_in", hs)
    pilha = [hs]
    for i in range(3):                                      # descida
        b = "down_blocks.%d" % i
        hs = _res(hs, tembs, P, b + ".resnets.0")
        hs = _trf(hs, ctxs, P, b + ".attentions.0")
        pilha.append(hs)
        if i < 2:
            hs = _conv(hs, P, b + ".downsamplers.0.conv", stride=2)
            pilha.append(hs)
        cap("down_%d" % i, hs)
    for i in range(3):                                      # subida
        b = "up_blocks.%d" % i
        for j in range(2):
            pula = pilha.pop()
            hs = [np.concatenate([h, s]) for h, s in zip(hs, pula)]
            hs = _res(hs, tembs, P, b + ".resnets.%d" % j)
            hs = _trf(hs, ctxs, P, b + ".attentions.%d" % j)
        if i < 2:
            hs = [mc.sobe2x(h) for h in hs]
            hs = _conv(hs, P, b + ".upsamplers.0.conv")
        cap("up_%d" % i, hs)
    M = _Memo(P)
    g, bb = M("conv_norm_out.weight"), M("conv_norm_out.bias")
    hs = [mc.silu(mc.group_norm(h, g, bb, 32)) for h in hs]
    out = _conv(hs, P, "conv_out")
    cap("conv_out", out)
    return out
