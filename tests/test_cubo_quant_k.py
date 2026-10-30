import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import cubo_quant as cq
import cubo_quant_k as ck


def snr(a, b):
    return 10 * np.log10((a ** 2).mean() / (((a - b) ** 2).mean() + 1e-30))


def ida_volta(w, bloco, bits, busca=True):
    p, e, m, R, C = ck.quantizar_k(w, bloco, bits, busca)
    return ck.desquantizar_k(p, e, m, R, C, w.shape, bloco, bits), (p, e, m)


def test_forma_e_enchimento_conv_4d():
    w = np.random.default_rng(0).standard_normal((37, 5, 3, 3)).astype(np.float32)   # C=45 nao e multiplo
    for bloco in ((1, 32), (8, 8), (4, 4)):
        for bits in (4, 8):
            r, _ = ida_volta(w, bloco, bits)
            assert r.shape == w.shape and np.isfinite(r).all()
    r, _ = ida_volta(w, (1, 32), 8)
    assert snr(w, r) > 40


def test_8bits_quase_exato_e_4bits_razoavel():
    w = np.random.default_rng(1).standard_normal((64, 256)).astype(np.float32)
    assert snr(w, ida_volta(w, (1, 32), 8)[0]) > 45
    assert snr(w, ida_volta(w, (1, 32), 4)[0]) > 17


def test_busca_de_faixa_nunca_piora_e_ganha_um_pouco():
    r = np.random.default_rng(2)
    for w in (r.standard_normal((64, 256)), r.standard_t(3, (64, 256)), r.laplace(size=(64, 256))):
        w = w.astype(np.float32)
        s_sem = snr(w, ida_volta(w, (1, 32), 4, busca=False)[0])
        s_com = snr(w, ida_volta(w, (1, 32), 4, busca=True)[0])
        assert s_com >= s_sem and s_com < s_sem + 3      # ganho real, mas modesto (~0,2-0,3 dB)


def test_assimetrico_vence_simetrico_em_dados_deslocados():
    w = np.abs(np.random.default_rng(3).standard_normal((64, 256))).astype(np.float32) + 0.5
    pk, es, R, C = cq.quantizar(w, (8, 8), 4)
    sim = cq.desquantizar(pk, es, R, C, w.shape, (8, 8), bits=4)
    assert snr(w, ida_volta(w, (8, 8), 4)[0]) > snr(w, sim) + 3


def test_tamanho_bate_com_bits_por_peso():
    w = np.random.default_rng(4).standard_normal((128, 256)).astype(np.float32)
    for bloco in ((1, 32), (1, 64), (8, 8)):
        _, (p, e, m) = ida_volta(w, bloco, 4)
        bpp = 8.0 * ck.bytes_k(p, e, m) / w.size
        assert abs(bpp - ck.bits_por_peso(bloco, 4)) < 0.01


def test_bloco_constante_nao_quebra():
    w = np.full((16, 64), 0.37, np.float32)
    r, _ = ida_volta(w, (1, 32), 4)
    assert np.allclose(r, w, atol=1e-3)
