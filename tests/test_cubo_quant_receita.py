import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import cubo_quant as cq
import motor_cubico as mc


def snr(a, b):
    return 10 * np.log10((a ** 2).mean() / (((a - b) ** 2).mean() + 1e-30))


def estado():
    r = np.random.default_rng(0)
    return {"a.weight": r.standard_normal((40, 70)).astype(np.float32),
            "b.weight": r.standard_normal((64, 8, 3, 3)).astype(np.float32),
            "c.weight": r.standard_normal((50, 33)).astype(np.float32),
            "d.weight": r.standard_normal((32, 64)).astype(np.float32),
            "e.bias": r.standard_normal(64).astype(np.float32)}


def test_receita_mista_ida_e_volta_e_tamanhos():
    e = estado()
    rec = {"a.weight": ((2, 2), 8), "b.weight": ((4, 4), 8), "c.weight": "fp16"}
    cam = os.path.join(tempfile.mkdtemp(), "m.ttq")
    cq.salvar(e, cam, (8, 8), 8, receita=rec)
    with cq.Leitor(cam) as L:
        out = {k: L.ler(k) for k in e}
        modos = {k: L.meta[k]["modo"] for k in e}
    for k in e:
        assert out[k].shape == e[k].shape
    assert modos["c.weight"] == "f16" and modos["a.weight"] == "q8"
    assert snr(e["c.weight"], out["c.weight"]) > 60            # fp16
    assert snr(e["a.weight"], out["a.weight"]) > snr(e["d.weight"], out["d.weight"]) + 3   # 2x2 > 8x8 (padrao)
    assert snr(e["b.weight"], out["b.weight"]) > snr(e["d.weight"], out["d.weight"])       # 4x4 > 8x8


def test_sem_receita_formato_antigo_identico():
    e = estado()
    d = tempfile.mkdtemp()
    cq.salvar(e, d + "/a.ttq", (8, 8), 4)
    cq.salvar(e, d + "/b.ttq", (8, 8), 4, receita={})
    assert open(d + "/a.ttq", "rb").read() == open(d + "/b.ttq", "rb").read()
    with cq.Leitor(d + "/a.ttq") as L:
        assert all("bloco" not in t for t in L.meta.values())


def test_motor_le_receita_mista():
    e = estado()
    cam = os.path.join(tempfile.mkdtemp(), "m.ttq")
    cq.salvar(e, cam, (4, 4), 8, receita={"a.weight": "fp16", "b.weight": ((2, 2), 8)})
    P = mc.Pesos(cam)
    assert P("a.weight").shape == (40, 70) and P("b.weight").shape == (64, 8, 3, 3)
    P.fechar()
