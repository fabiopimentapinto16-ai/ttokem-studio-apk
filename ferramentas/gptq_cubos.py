"""GPTQ adaptado aos CUBOS do .ttq (numpy puro, testavel sem torch).

GPTQ: quantiza o peso coluna a coluna e COMPENSA o erro de cada arredondamento nas colunas que ainda faltam,
usando a Hessiana H = X^T X das entradas reais da camada (calibracao). Resultado: o mesmo numero de bits erra
bem menos na SAIDA da camada do que o arredondamento simples (RTN).
Escala: uma por cubo (bh linhas x bw colunas), absmax/nivel, arredondada para fp16 (igual ao formato .ttq).
Aproximacao: Hessiana em BLOCOS DIAGONAIS de `chunk` colunas (camadas grandes nao cabem inteiras).
"""
import numpy as np


def _escalas(g, bh, niv):
    """g: (R, w) pesos de um grupo de colunas. Devolve (escala por bloco de bh linhas, escala por LINHA (R,))."""
    R = g.shape[0]
    nb = -(-R // bh)
    p = np.zeros((nb * bh, g.shape[1]), np.float32); p[:R] = g
    mx = np.abs(p).reshape(nb, bh * g.shape[1]).max(1)
    sc = np.where(mx == 0, 1.0, mx / niv).astype(np.float16).astype(np.float32)
    return sc, np.repeat(sc, bh)[:R]


def rtn(W, bloco=(4, 16), bits=4):
    """Arredondamento simples por cubo (referencia). Identico a quantizar/desquantizar do cubo_quant."""
    bh, bw = bloco; niv = 2 ** (bits - 1) - 1
    W = np.asarray(W, np.float32); R, C = W.shape
    Q = np.empty_like(W)
    for c0 in range(0, C, bw):
        sc = _escalas(W[:, c0:c0 + bw], bh, niv)[1][:, None]
        Q[:, c0:c0 + bw] = np.clip(np.rint(W[:, c0:c0 + bw] / sc), -niv - 1, niv) * sc
    return Q


def preparar(H, amortecimento=0.01):
    """Parte CARA do GPTQ (inversa da Hessiana, fator de Cholesky). Pode ser feita UMA vez e reutilizada em varias opcoes de bits."""
    H = np.array(H, np.float64); C = H.shape[0]
    d = np.diag(H).copy(); morto = np.where(d <= 0)[0]
    if len(morto):
        H[morto, morto] = 1.0
    H[np.diag_indices(C)] += amortecimento * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)
    Hinv = (Hinv + Hinv.T) / 2
    return np.linalg.cholesky(Hinv).T.astype(np.float32), morto               # triangular superior


def gptq(W, H, bloco=(4, 16), bits=4, amortecimento=0.01, lazy=128, devolver_escalas=False, pre=None):
    W = np.array(W, np.float32); R, C = W.shape
    bh, bw = bloco; niv = 2 ** (bits - 1) - 1
    assert lazy % bw == 0
    U, morto = pre if pre is not None else preparar(H, amortecimento)
    if len(morto):
        W[:, morto] = 0.0
    Q = np.zeros_like(W); S = []
    for i1 in range(0, C, lazy):
        i2 = min(C, i1 + lazy)
        W1 = W[:, i1:i2].copy(); Err = np.zeros_like(W1); U1 = U[i1:i2, i1:i2]
        scr = None
        for i in range(i2 - i1):
            if (i1 + i) % bw == 0:
                sb, scr = _escalas(W1[:, i:i + bw], bh, niv); S.append(sb.astype(np.float16))
            w = W1[:, i]
            q = np.clip(np.rint(w / scr), -niv - 1, niv) * scr
            Q[:, i1 + i] = q
            err = (w - q) / U1[i, i]
            W1[:, i:] -= np.outer(err, U1[i, i:])
            Err[:, i] = err
        if i2 < C:
            W[:, i2:] -= Err @ U[i1:i2, i2:]
    if devolver_escalas:
        return Q, np.stack(S, axis=1)                  # escalas fp16: (blocos de linhas, grupos de colunas)
    return Q


def erro_rel(W, Q, H):
    """Erro relativo na SAIDA da camada: tr(dW H dW^T) / tr(W H W^T)."""
    W = np.asarray(W, np.float64); dW = W - np.asarray(Q, np.float64); H = np.asarray(H, np.float64)
    return float(((dW @ H) * dW).sum() / max(((W @ H) * W).sum(), 1e-30))


def gptq_blocos(W, Hblocos, chunk, pre=None, **kw):
    """Camada grande: Hessiana so em blocos diagonais de `chunk` colunas (chunk multiplo de bw).
    Devolve (Q, erro relativo agregado, escalas fp16 (blocos de linhas, grupos de colunas))."""
    W = np.asarray(W, np.float32); R, C = W.shape
    Q = np.empty_like(W); num = den = 0.0; Ss = []
    for k, c0 in enumerate(range(0, C, chunk)):
        c1 = min(C, c0 + chunk); H = Hblocos[k]
        Q[:, c0:c1], Sk = gptq(W[:, c0:c1], H, devolver_escalas=True, pre=None if pre is None else pre[k], **kw); Ss.append(Sk)
        Wc = W[:, c0:c1].astype(np.float64); dW = Wc - Q[:, c0:c1]
        num += float(((dW @ H) * dW).sum()); den += float(((Wc @ H) * Wc).sum())
    return Q, num / max(den, 1e-30), np.concatenate(Ss, axis=1)
