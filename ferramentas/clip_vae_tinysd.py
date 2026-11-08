"""CLIP (texto) e VAE (decodificador) do tiny-sd em numpy puro, lendo os pesos do .ttq por partes.

clip_texto(ids, P)       ids (77,) inteiros -> (77, 768)   [= text_encoder(ids).last_hidden_state]
vae_decodificar(z, P)    z (4,H,W) JA dividido por scaling_factor (0.18215) -> (3, 8H, 8W) em [-1, 1]

STATUS: forma e fiacao testadas com pesos aleatorios; a igualdade com o torch so se prova rodando
colab_validar_clip_vae.py contra tinysd_ref.npz (Colab).
"""
import numpy as np

try:
    import motor_cubico as mc
except ImportError:
    from . import motor_cubico as mc


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _linhas(P, nome, idx):
    lt = getattr(P, "lt", None)
    if lt is not None and hasattr(lt, "ler_filas"):
        return lt.ler_filas(nome, idx)
    return np.asarray(P(nome))[idx]


# ------------------------------------------------------------------ CLIP
def _atencao_causal(h, P, p, cabecas):
    N, D = h.shape
    d = D // cabecas
    q = mc.linear(h, P(p + ".q_proj.weight"), P(p + ".q_proj.bias")) * (d ** -0.5)
    k = mc.linear(h, P(p + ".k_proj.weight"), P(p + ".k_proj.bias"))
    v = mc.linear(h, P(p + ".v_proj.weight"), P(p + ".v_proj.bias"))
    qh, kh, vh = [a.reshape(N, cabecas, d).transpose(1, 0, 2) for a in (q, k, v)]
    s = mc.mm(qh, kh.transpose(0, 2, 1))
    s = s + np.triu(np.full((N, N), -np.inf, np.float32), 1)       # cada token so ve os anteriores
    s -= s.max(-1, keepdims=True)
    e = np.exp(s)
    e /= e.sum(-1, keepdims=True)
    o = mc.mm(e, vh).transpose(1, 0, 2).reshape(N, D)
    return mc.linear(o, P(p + ".out_proj.weight"), P(p + ".out_proj.bias"))


def clip_texto(ids, P, camadas=12, cabecas=12):
    ids = np.asarray(ids, np.int64).reshape(-1)
    N = len(ids)
    h = _linhas(P, "embeddings.token_embedding.weight", ids) + np.asarray(P("embeddings.position_embedding.weight"))[:N]
    h = h.astype(np.float32)
    for i in range(camadas):
        p = "encoder.layers.%d" % i
        a = mc.layer_norm(h, P(p + ".layer_norm1.weight"), P(p + ".layer_norm1.bias"))
        h = h + _atencao_causal(a, P, p + ".self_attn", cabecas)
        m = mc.layer_norm(h, P(p + ".layer_norm2.weight"), P(p + ".layer_norm2.bias"))
        m = mc.linear(m, P(p + ".mlp.fc1.weight"), P(p + ".mlp.fc1.bias"))
        m = m * _sig(1.702 * m)                                     # quick_gelu
        h = h + mc.linear(m, P(p + ".mlp.fc2.weight"), P(p + ".mlp.fc2.bias"))
    return mc.layer_norm(h, P("final_layer_norm.weight"), P("final_layer_norm.bias")).astype(np.float32)


# ------------------------------------------------------------------ VAE
EPS_VAE = 1e-6


def _resnet_vae(x, P, p):
    h = mc.silu(mc.group_norm(x, P(p + ".norm1.weight"), P(p + ".norm1.bias"), 32, EPS_VAE))
    h = mc.conv2d(h, P(p + ".conv1.weight"), P(p + ".conv1.bias"), pad=1)
    h = mc.silu(mc.group_norm(h, P(p + ".norm2.weight"), P(p + ".norm2.bias"), 32, EPS_VAE))
    h = mc.conv2d(h, P(p + ".conv2.weight"), P(p + ".conv2.bias"), pad=1)
    if P.tem(p + ".conv_shortcut.weight"):
        x = mc.conv2d(x, P(p + ".conv_shortcut.weight"), P(p + ".conv_shortcut.bias"))
    return x + h


def _atencao_vae(x, P, p):
    C, H, W = x.shape
    h = mc.group_norm(x, P(p + ".group_norm.weight"), P(p + ".group_norm.bias"), 32, EPS_VAE)
    h = h.reshape(C, H * W).T
    q = mc.linear(h, P(p + ".to_q.weight"), P(p + ".to_q.bias"))
    k = mc.linear(h, P(p + ".to_k.weight"), P(p + ".to_k.bias"))
    v = mc.linear(h, P(p + ".to_v.weight"), P(p + ".to_v.bias"))
    o = mc.atencao(q, k, v, 1)                                       # uma cabeca de dimensao C
    o = mc.linear(o, P(p + ".to_out.0.weight"), P(p + ".to_out.0.bias"))
    return x + o.T.reshape(C, H, W)


def vae_decodificar(z, P):
    z = np.asarray(z, np.float32)
    h = mc.conv2d(z, P("post_quant_conv.weight"), P("post_quant_conv.bias"))
    h = mc.conv2d(h, P("decoder.conv_in.weight"), P("decoder.conv_in.bias"), pad=1)
    h = _resnet_vae(h, P, "decoder.mid_block.resnets.0")
    h = _atencao_vae(h, P, "decoder.mid_block.attentions.0")
    h = _resnet_vae(h, P, "decoder.mid_block.resnets.1")
    for i in range(4):
        b = "decoder.up_blocks.%d" % i
        for j in range(3):
            h = _resnet_vae(h, P, "%s.resnets.%d" % (b, j))
        if P.tem(b + ".upsamplers.0.conv.weight"):
            h = mc.sobe2x(h)
            h = mc.conv2d(h, P(b + ".upsamplers.0.conv.weight"), P(b + ".upsamplers.0.conv.bias"), pad=1)
    h = mc.silu(mc.group_norm(h, P("decoder.conv_norm_out.weight"), P("decoder.conv_norm_out.bias"), 32, EPS_VAE))
    return mc.conv2d(h, P("decoder.conv_out.weight"), P("decoder.conv_out.bias"), pad=1)


def para_imagem(x):
    """(3,H,W) em [-1,1] -> uint8 (H,W,3)."""
    return np.clip((x.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
