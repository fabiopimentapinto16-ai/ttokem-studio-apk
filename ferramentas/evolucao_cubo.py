"""Nucleo evolutivo da quantizacao em cubos (numpy puro): populacao de receitas, quem estoura o tamanho
morre, os melhores geram filhos. Usado por evolucao_cubo_unico.py (Colab) e testado aqui."""
import numpy as np

# ------------------------------------------------------------ NUCLEO EVOLUTIVO (numpy puro, testavel sem torch)
OPCOES_BITS = np.array([8.25, 9.0, 12.0, 16.0])      # 8x8, 4x4, 2x2, fp16 (bits por peso, com a escala)
ERRO_REL_4x4 = np.array([2.9, 1.0, 0.23, 0.0])        # estimativa do erro de cada opcao relativo ao 4x4 (refinado pela evolucao)
PM0 = 0.12

def tamanho_frac(g, numel):
    return float((numel * OPCOES_BITS[g]).sum() / (numel.sum() * 16.0))

def mutar(g, imp, pm, rng, piso=0):
    """Cada gene muda com prob. pm. Tensores IMPORTANTES tendem a subir de precisao, os outros a descer."""
    f = g.copy()
    mud = rng.random(len(g)) < pm
    sobe = rng.random(len(g)) < np.clip(0.15 + 0.7 * imp, 0.05, 0.95)
    passo = np.where(sobe, 1, -1)
    f[mud] = np.clip(f[mud] + passo[mud], piso, len(OPCOES_BITS) - 1)
    return f

def cruzar(a, b, rng):
    return np.where(rng.random(len(a)) < 0.5, a, b)

def evoluir(avaliar, numel, imp, orcamento, snr_meta, pop=10, geracoes=20, semente=0, log=print, iniciais=(), piso=0):
    """avaliar(g) -> erro relativo da saida (menor = melhor). Quem passa do orcamento de tamanho MORRE;
    entre os vivos vence o menor erro. Para quando algum vivo atinge a meta de SNR (dB)."""
    rng = np.random.default_rng(semente)
    N = len(numel)
    cache = {}
    def pontuar(g):
        k = g.tobytes()
        if k not in cache:
            t = tamanho_frac(g, numel)
            cache[k] = (t, avaliar(g) if t <= orcamento else float("inf"))
        return cache[k]
    populacao = [np.asarray(g, np.int64) for g in iniciais][:pop]
    while len(populacao) < pop:
        base = populacao[rng.integers(0, len(populacao))] if populacao else np.full(N, max(piso, 1), np.int64)
        populacao.append(mutar(base, imp, 0.1, rng, piso))
    pm, sem_melhora, melhor_global = PM0, 0, (float("inf"), None, None)
    historico = []
    for ger in range(geracoes):
        ponts = [pontuar(g) for g in populacao]
        ordem = sorted(range(len(populacao)), key=lambda i: (ponts[i][1], ponts[i][0]))
        vivos = [i for i in ordem if np.isfinite(ponts[i][1])]
        mortos = len(populacao) - len(vivos)
        if vivos:
            b = vivos[0]; e, t = ponts[b][1], ponts[b][0]
            snr = -10 * np.log10(max(e, 1e-30))
            if e < melhor_global[0] * 0.999:
                melhor_global, sem_melhora = (e, populacao[b].copy(), t), 0
            else:
                sem_melhora += 1
            log("geracao %2d | melhor: %.1f dB com %.1f%% do tamanho | mortos por tamanho: %d | mutacao %.2f"
                % (ger, snr, 100 * t, mortos, pm))
            historico.append((ger, snr, t))
            if snr >= snr_meta:
                log("META ATINGIDA (>= %.0f dB dentro do orcamento)." % snr_meta)
                break
        else:
            log("geracao %2d | todos estouraram o orcamento" % ger)
        pm = min(0.35, pm * 1.5) if sem_melhora >= 3 else PM0
        sobreviventes = [populacao[i] for i in (vivos or ordem)[:max(2, pop // 2)]]
        elite = sobreviventes[:3]
        filhos = list(elite)
        while len(filhos) < pop:
            if rng.random() < 0.3 and len(sobreviventes) > 1:
                i, j = rng.choice(len(sobreviventes), 2, replace=False)
                f = mutar(cruzar(sobreviventes[i], sobreviventes[j], rng), imp, pm / 2, rng, piso)
            else:
                f = mutar(sobreviventes[rng.integers(0, len(sobreviventes))], imp, pm, rng, piso)
            filhos.append(f)
        populacao = filhos
    return melhor_global, historico


def guloso_inicial(efeito4, numel, orcamento, piso=0):
    """Ponto de partida informado: usa o efeito MEDIDO de cada tensor em 4x4 e sobe, um degrau por vez, o tensor
    com maior ganho por bit extra, enquanto couber no orcamento. A evolucao refina o resto (efeitos nao aditivos)."""
    N = len(numel); g = np.full(N, piso, np.int64)
    ganho = lambda i: efeito4[i] * (ERRO_REL_4x4[g[i]] - ERRO_REL_4x4[g[i] + 1]) / (numel[i] * (OPCOES_BITS[g[i] + 1] - OPCOES_BITS[g[i]]))
    ativo = set(range(N))
    while ativo:
        i = max(ativo, key=ganho)
        g[i] += 1
        if tamanho_frac(g, numel) > orcamento:
            g[i] -= 1; ativo.discard(i); continue
        if g[i] >= len(OPCOES_BITS) - 1: ativo.discard(i)
    return g
