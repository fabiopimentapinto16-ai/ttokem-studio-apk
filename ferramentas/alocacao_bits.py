"""Alocacao de bits por tensor (numpy puro, testavel sem torch).

Cada tensor tem varias OPCOES (bits por peso, erro previsto). Escolhe uma opcao por tensor para MINIMIZAR o erro total
dentro de um orcamento de tamanho: casco convexo por tensor + guloso global pela inclinacao (erro poupado por bit gasto).
Modelo de erro: erro_i(opcao) = efeito_i * ruido_i(opcao) / ruido_i(ref)  (resposta linear; erros de tensores quase somam).
"""
import numpy as np


def casco(bpp, err):
    """Indices (ordenados por bpp) do casco convexo inferior de (bpp, err): so as opcoes que valem a pena."""
    ordem = sorted(range(len(bpp)), key=lambda i: (bpp[i], err[i]))
    util = []
    for i in ordem:                                   # tira dominadas (mais caras e nao melhores)
        if not util or err[i] < err[util[-1]] - 1e-30:
            util.append(i)
    h = []
    for i in util:                                    # casco convexo (inclinacao cada vez menos negativa)
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (err[b] - err[a]) * (bpp[i] - bpp[b]) >= (err[i] - err[b]) * (bpp[b] - bpp[a]):
                h.pop()
            else:
                break
        h.append(i)
    return h


def alocar(bpp, err, numel, orcamento_bits):
    """bpp[o] (bits/peso por opcao, comum a todos), err[i][o] (erro previsto do tensor i na opcao o), numel[i].
    orcamento_bits: total de bits permitido. Devolve (escolha[i] = indice da opcao, bits usados, erro previsto)."""
    N = len(numel)
    bpp = np.asarray(bpp, float)
    H = [casco(bpp, err[i]) for i in range(N)]
    pos = np.zeros(N, int)
    escolha = np.array([H[i][0] for i in range(N)])
    usado = float(sum(numel[i] * bpp[escolha[i]] for i in range(N)))
    if usado > orcamento_bits:
        return escolha, usado, float(sum(err[i][escolha[i]] for i in range(N)))      # nem o minimo cabe
    def inclinacao(i):
        if pos[i] + 1 >= len(H[i]):
            return -1.0
        a, b = H[i][pos[i]], H[i][pos[i] + 1]
        return (err[i][a] - err[i][b]) / (numel[i] * (bpp[b] - bpp[a]))
    ativo = set(range(N))
    while ativo:
        i = max(ativo, key=inclinacao)
        if inclinacao(i) < 0:
            break
        a, b = H[i][pos[i]], H[i][pos[i] + 1]
        custo = numel[i] * (bpp[b] - bpp[a])
        if usado + custo > orcamento_bits:
            ativo.discard(i)
            continue
        pos[i] += 1; escolha[i] = b; usado += custo
        if pos[i] + 1 >= len(H[i]):
            ativo.discard(i)
    return escolha, usado, float(sum(err[i][escolha[i]] for i in range(N)))
