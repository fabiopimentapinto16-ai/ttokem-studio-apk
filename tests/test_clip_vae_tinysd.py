import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import cubo_quant as cq
import motor_cubico as mc
import clip_vae_tinysd as cv


def pesos_clip(D=64, camadas=2, vocab=300, N=77, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s: (r.standard_normal(s) * 0.05).astype(np.float32)
    s = {"embeddings.token_embedding.weight": n(vocab, D), "embeddings.position_embedding.weight": n(N, D),
         "final_layer_norm.weight": np.ones(D, np.float32), "final_layer_norm.bias": np.zeros(D, np.float32)}
    for i in range(camadas):
        p = "encoder.layers.%d" % i
        for q in ("q", "k", "v", "out"):
            s["%s.self_attn.%s_proj.weight" % (p, q)], s["%s.self_attn.%s_proj.bias" % (p, q)] = n(D, D), n(D)
        for l in ("layer_norm1", "layer_norm2"):
            s["%s.%s.weight" % (p, l)], s["%s.%s.bias" % (p, l)] = np.ones(D, np.float32), np.zeros(D, np.float32)
        s[p + ".mlp.fc1.weight"], s[p + ".mlp.fc1.bias"] = n(4 * D, D), n(4 * D)
        s[p + ".mlp.fc2.weight"], s[p + ".mlp.fc2.bias"] = n(D, 4 * D), n(D)
    return s


def test_clip_causal_e_forma():
    s = pesos_clip()
    P = mc.PesosMemoria(s)
    ids = np.arange(77) % 300
    a = cv.clip_texto(ids, P, camadas=2, cabecas=4)
    assert a.shape == (77, 64) and np.isfinite(a).all()
    ids2 = ids.copy(); ids2[40:] = 7                       # mudar tokens FUTUROS nao pode mudar os anteriores
    b = cv.clip_texto(ids2, P, camadas=2, cabecas=4)
    assert np.allclose(a[:40], b[:40], atol=1e-5) and not np.allclose(a[40:], b[40:])


def test_clip_lendo_do_ttq_por_filas():
    s = pesos_clip(seed=1)
    cam = os.path.join(tempfile.mkdtemp(), "t.ttq")
    cq.salvar(s, cam, (4, 4), 8)
    P = mc.Pesos(cam)
    ids = np.random.default_rng(0).integers(0, 300, 77)
    a = cv.clip_texto(ids, P, camadas=2, cabecas=4)
    b = cv.clip_texto(ids, mc.PesosMemoria({k: P(k) for k in s}), camadas=2, cabecas=4)   # mesmos pesos, tudo em RAM
    assert np.allclose(a, b, atol=1e-5)
    P.fechar()


def pesos_vae(canais=(64, 64, 32, 32), lat=4, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s: (r.standard_normal(s) * 0.03).astype(np.float32)
    um = lambda c: np.ones(c, np.float32)
    z = lambda c: np.zeros(c, np.float32)
    s = {"post_quant_conv.weight": n(lat, lat, 1, 1), "post_quant_conv.bias": z(lat)}
    def res(p, cin, cout):
        s[p + ".norm1.weight"], s[p + ".norm1.bias"] = um(cin), z(cin)
        s[p + ".conv1.weight"], s[p + ".conv1.bias"] = n(cout, cin, 3, 3), z(cout)
        s[p + ".norm2.weight"], s[p + ".norm2.bias"] = um(cout), z(cout)
        s[p + ".conv2.weight"], s[p + ".conv2.bias"] = n(cout, cout, 3, 3), z(cout)
        if cin != cout:
            s[p + ".conv_shortcut.weight"], s[p + ".conv_shortcut.bias"] = n(cout, cin, 1, 1), z(cout)
    s["decoder.conv_in.weight"], s["decoder.conv_in.bias"] = n(canais[0], lat, 3, 3), z(canais[0])
    c = canais[0]
    res("decoder.mid_block.resnets.0", c, c); res("decoder.mid_block.resnets.1", c, c)
    p = "decoder.mid_block.attentions.0"
    s[p + ".group_norm.weight"], s[p + ".group_norm.bias"] = um(c), z(c)
    for q in ("to_q", "to_k", "to_v", "to_out.0"):
        s["%s.%s.weight" % (p, q)], s["%s.%s.bias" % (p, q)] = n(c, c), z(c)
    atual = c
    for i, saida in enumerate(canais):
        for j in range(3):
            res("decoder.up_blocks.%d.resnets.%d" % (i, j), atual, saida); atual = saida
        if i < 3:
            s["decoder.up_blocks.%d.upsamplers.0.conv.weight" % i], s["decoder.up_blocks.%d.upsamplers.0.conv.bias" % i] = n(saida, saida, 3, 3), z(saida)
    s["decoder.conv_norm_out.weight"], s["decoder.conv_norm_out.bias"] = um(atual), z(atual)
    s["decoder.conv_out.weight"], s["decoder.conv_out.bias"] = n(3, atual, 3, 3), z(3)
    return s


def test_vae_forma_e_finito():
    P = mc.PesosMemoria(pesos_vae())
    z = np.random.default_rng(0).standard_normal((4, 4, 4)).astype(np.float32)
    y = cv.vae_decodificar(z, P)
    assert y.shape == (3, 32, 32) and np.isfinite(y).all()           # 4x4 -> 32x32 (x8)
    im = cv.para_imagem(y)
    assert im.shape == (32, 32, 3) and im.dtype == np.uint8


def test_vae_do_ttq_igual_aos_mesmos_pesos():
    s = pesos_vae(seed=2)
    cam = os.path.join(tempfile.mkdtemp(), "v.ttq")
    cq.salvar(s, cam, (4, 4), 8)
    P = mc.Pesos(cam)
    z = np.random.default_rng(1).standard_normal((4, 4, 4)).astype(np.float32)
    a = cv.vae_decodificar(z, P)
    b = cv.vae_decodificar(z, mc.PesosMemoria({k: P(k) for k in s}))
    assert np.allclose(a, b, atol=1e-5)
    P.fechar()
