import os, sys, tempfile, unittest
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


def test_ler_filas_igual_a_ler_completo():
    r = np.random.default_rng(5)
    for forma, bloco, bits in (((97, 70), (4, 4), 8), ((50, 33), (8, 8), 8), ((64, 64), (4, 4), 4), ((40, 20), (2, 2), 8)):
        w = r.standard_normal(forma).astype(np.float32)
        cam = os.path.join(tempfile.mkdtemp(), "f.ttq")
        cq.salvar({"w": w}, cam, bloco, bits)
        with cq.Leitor(cam) as L:
            completo = L.ler("w")
            for filas in ([0], [forma[0] - 1], [3, 3, 60 % forma[0], 1], list(range(forma[0]))[::7]):
                assert np.array_equal(L.ler_filas("w", filas), completo[filas])
    cam = os.path.join(tempfile.mkdtemp(), "f.ttq")
    cq.salvar({"b": np.random.randn(10, 4).astype(np.float32)}, cam, (4, 4), 8, receita={"b": "fp16"})
    with cq.Leitor(cam) as L:
        assert np.array_equal(L.ler_filas("b", [2, 5]), L.ler("b")[[2, 5]])


class TestDesquantizarRapido(unittest.TestCase):
    def test_identico_ao_lento(self):
        import numpy as np, cubo_quant as cq
        rng = np.random.RandomState(0)
        for forma in ((64, 64), (37, 51), (320, 4, 3, 3), (7, 9)):
            w = rng.randn(*forma).astype("f")
            for bl in ((2, 2), (4, 4), (8, 8), (1, 32)):
                pack, esc, R, C = cq.quantizar(w, bl, 8)
                a = cq.desquantizar(pack, esc, R, C, forma, bl, np.float32, 8)
                b = cq.desquantizar_rapido(pack, esc, R, C, forma, bl, np.float32)
                self.assertTrue(np.array_equal(a, b), (forma, bl))
