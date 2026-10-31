import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import evolucao_cubo as ev


def mundo(seed, N=151):
    rng = np.random.default_rng(seed)
    numel = (rng.lognormal(12, 1.5, N)).astype(np.int64) + 1000
    sens = rng.pareto(1.5, N) + 0.01
    sens /= sens.sum()
    err = np.array([1.0, 0.35, 0.08, 0.0])
    avaliar = lambda g: float((sens * err[g]).sum() * 1e-3) + 1e-9
    medido = sens * 0.35e-3 * rng.lognormal(0, 0.3, N)
    imp = np.argsort(np.argsort(medido / numel)) / N
    return numel, avaliar, medido, imp


def test_tamanho_frac_extremos():
    numel = np.array([100, 300])
    assert abs(ev.tamanho_frac(np.array([3, 3]), numel) - 1.0) < 1e-12
    assert abs(ev.tamanho_frac(np.array([1, 1]), numel) - 9 / 16) < 1e-12


def test_guloso_respeita_orcamento_e_melhora_o_4x4():
    for sd in range(3):
        numel, avaliar, medido, imp = mundo(sd)
        gi = ev.guloso_inicial(medido, numel, 0.62)
        assert ev.tamanho_frac(gi, numel) <= 0.62 + 1e-9
        assert avaliar(gi) < avaliar(np.ones(len(numel), int)) / 4          # >= 6 dB melhor


def test_evolucao_respeita_orcamento_nunca_piora_e_para_na_meta():
    numel, avaliar, medido, imp = mundo(1)
    gi = ev.guloso_inicial(medido, numel, 0.62)
    (e, g, t), hist = ev.evoluir(avaliar, numel, imp, 0.62, 999, pop=10, geracoes=8, iniciais=[gi], log=lambda *a: None)
    assert t <= 0.62 + 1e-9 and e <= avaliar(gi) + 1e-15 and len(hist) == 8
    (e2, g2, t2), hist2 = ev.evoluir(avaliar, numel, imp, 0.62, 30.0, pop=10, geracoes=8, iniciais=[gi], log=lambda *a: None)
    assert len(hist2) == 1                                                  # meta de 30 dB ja atingida na geracao 0


def test_individuos_acima_do_orcamento_morrem():
    numel, avaliar, medido, imp = mundo(2)
    chamadas = []
    def av(g):
        chamadas.append(ev.tamanho_frac(g, numel)); return avaliar(g)
    ev.evoluir(av, numel, imp, 0.60, 999, pop=10, geracoes=5, iniciais=[np.ones(len(numel), int)], log=lambda *a: None)
    assert all(t <= 0.60 + 1e-9 for t in chamadas)                          # nem chegam a ser avaliados


def test_mutacao_tende_a_subir_importantes():
    rng = np.random.default_rng(0)
    g = np.ones(2000, int)
    imp = np.concatenate([np.ones(1000), np.zeros(1000)])
    f = ev.mutar(g, imp, 1.0, rng)
    assert (f[:1000] > 1).mean() > 0.8 and (f[1000:] < 1).mean() > 0.6
    assert f.min() >= 0 and f.max() <= 3


def test_piso_proibe_precisao_baixa():
    numel, avaliar, medido, imp = mundo(3)
    gi = ev.guloso_inicial(medido, numel, 0.62, piso=1)
    assert gi.min() >= 1
    (e, g, t), _ = ev.evoluir(avaliar, numel, imp, 0.62, 999, pop=10, geracoes=6, iniciais=[gi], piso=1, log=lambda *a: None)
    assert g.min() >= 1 and t <= 0.62 + 1e-9
    f = ev.mutar(np.ones(500, int), np.zeros(500), 1.0, np.random.default_rng(0), piso=1)
    assert f.min() >= 1
