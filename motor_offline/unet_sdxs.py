"""UNet do SDXS-512 (1 passo, sem guia) em numpy puro. Mesma fiacao do unet_tinysd, mas a estrutura vem dos NOMES dos pesos:

  conv_in 4->320 | down0 (320: so resnet, reduz) | down1 (640: resnet+atencao, reduz) | down2 (1280: resnet+atencao)
  | SEM bloco do meio | up0 (1280: 2x[resnet+atencao], sobe) | up1 (640: 2x[resnet+atencao], sobe) | up2 (320: 2x resnet)
  | norm -> silu -> conv_out 320->4.   (arquitetura lida de sdxs_arquitetura.json; 315,7 M pesos)

Tambem serve para o tiny-sd (la todos os blocos tem atencao): o bloco so aplica atencao/reducao/subida se o peso existe.
STATUS: fiacao conferida pelas formas do sdxs_arquitetura.json; a igualdade com o torch so se prova no Colab (colab_sdxs_validar.py).
"""
import numpy as np

try:
    import motor_cubico as mc
    import unet_tinysd as ut
except ImportError:
    from . import motor_cubico as mc
    from . import unet_tinysd as ut


def unet_sdxs(xs, t, ctxs, P, captura=None):
    """xs: lista de latentes (4,H,W); ctxs: lista de (77,768); t: passo. Devolve a lista de previsoes de ruido."""
    cap = (lambda k, v: captura.__setitem__(k, np.array(v[0], np.float32))) if captura is not None else (lambda k, v: None)
    temb = mc.embedding_tempo(float(t), 320, P, "time_embedding")
    tembs = [temb] * len(xs)
    hs = ut._conv(xs, P, "conv_in")
    cap("conv_in", hs)
    pilha = [hs]
    for i in range(3):                                         # descida
        b = "down_blocks.%d" % i
        hs = ut._res(hs, tembs, P, b + ".resnets.0")
        if P.tem(b + ".attentions.0.norm.weight"):
            hs = ut._trf(hs, ctxs, P, b + ".attentions.0")
        pilha.append(hs)
        if P.tem(b + ".downsamplers.0.conv.weight"):
            hs = ut._conv(hs, P, b + ".downsamplers.0.conv", stride=2)
            pilha.append(hs)
        cap("down_%d" % i, hs)
    for i in range(3):                                         # subida
        b = "up_blocks.%d" % i
        for j in range(2):
            pula = pilha.pop()
            hs = [np.concatenate([h, s]) for h, s in zip(hs, pula)]
            hs = ut._res(hs, tembs, P, b + ".resnets.%d" % j)
            if P.tem(b + ".attentions.%d.norm.weight" % j):
                hs = ut._trf(hs, ctxs, P, b + ".attentions.%d" % j)
        if P.tem(b + ".upsamplers.0.conv.weight"):
            hs = [mc.sobe2x(h) for h in hs]
            hs = ut._conv(hs, P, b + ".upsamplers.0.conv")
        cap("up_%d" % i, hs)
    M = ut._Memo(P)
    g, bb = M("conv_norm_out.weight"), M("conv_norm_out.bias")
    hs = [mc.silu(mc.group_norm(h, g, bb, 32)) for h in hs]
    out = ut._conv(hs, P, "conv_out")
    cap("conv_out", out)
    return out
