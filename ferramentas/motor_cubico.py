"""MOTOR CUBICO: roda camadas de uma rede de difusao em numpy (Python puro, sem torch),
lendo os pesos de um arquivo .ttq (cubos de 4 bits) UMA CAMADA POR VEZ.

Ideia (do Fabio): o modelo e maior que a RAM do celular, entao
  1) os pesos ficam comprimidos em cubos de 4 bits no armazenamento (cubo_quant.py);
  2) a cada camada o motor "abre" so aquele pedaco, calcula e joga fora;
  3) as contas grandes (convolucao, atencao) tambem andam por partes (linhas / blocos).
O pico de RAM fica perto de  (ativacoes + a MAIOR camada)  e nao do tamanho do modelo.

Nomes dos pesos seguem o diffusers (resnets.0.norm1, attn1.to_q, ff.net.0.proj ...), entao
portar o UNet real do SDXS/SD e compor estes blocos. Os blocos sao validados contra
implementacoes ingenuas nos testes; o UNet COMPLETO real ainda NAO foi validado (precisa dos
pesos reais e de uma referencia).
"""
import time
from collections import OrderedDict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import cubo_quant as cq
except ImportError:                                  # quando importado como pacote
    from . import cubo_quant as cq


# ------------------------------------------------------------------ pesos por partes
class Pesos:
    """Entrega pesos float32 de um .ttq. cache_mb=0: nada fica na RAM depois do uso."""

    def __init__(self, caminho, cache_mb=0):
        self.lt = cq.Leitor(caminho)
        self.cache_mb, self._cache, self._bytes = cache_mb, OrderedDict(), 0
        self.leituras = 0
        self.t_leitura = 0.0

    def tem(self, nome):
        return nome in self.lt.meta

    def __call__(self, nome):
        if nome in self._cache:
            self._cache.move_to_end(nome)
            return self._cache[nome]
        t = time.time()
        a = self.lt.ler(nome)
        self.t_leitura += time.time() - t
        self.leituras += 1
        if self.cache_mb and a.nbytes <= self.cache_mb * 2**20:
            self._cache[nome] = a
            self._bytes += a.nbytes
            while self._bytes > self.cache_mb * 2**20:
                _, v = self._cache.popitem(last=False)
                self._bytes -= v.nbytes
        return a

    def fechar(self):
        self.lt.fechar()


class PesosMemoria:
    """Mesma interface, pesos ja em RAM (referencia para comparar)."""

    def __init__(self, estado):
        self.e = {k: np.asarray(v, np.float32) for k, v in estado.items()}

    def tem(self, nome):
        return nome in self.e

    def __call__(self, nome):
        return self.e[nome]


# ------------------------------------------------------------------ multiplicacao em varios nucleos
# Onde o numpy NAO tem BLAS otimizado (ex.: dentro do APK) cada matmul usa 1 nucleo so. Aqui divide em fatias e roda
# em varias threads (o numpy solta o GIL). configurar_hilos() MEDE e so liga se ganhar (no Pydroid, que ja tem BLAS
# multi-nucleo, fica desligado). Resultado identico a a @ b.
_HILOS = 1
_POOL = None


def _fatias(n, k):
    base, resto = divmod(n, k)
    i = 0
    for j in range(k):
        t = base + (1 if j < resto else 0)
        if t:
            yield i, i + t
        i += t


try:                                   # matmul propria multi-thread (so existe no APK; no resto cai no numpy)
    import ttgemm as _tg
except Exception:
    _tg = None
_NT_TG = [0]


def _tg_ok(a, b):
    return (_tg is not None and a.dtype == np.float32 and b.dtype == np.float32
            and a.flags.c_contiguous and b.flags.c_contiguous)


def _tg_mm(a, b):
    import os
    if not _NT_TG[0]:
        _NT_TG[0] = max(1, min(8, os.cpu_count() or 1))
    m, k = a.shape
    n = b.shape[1]
    out = np.empty((m, n), np.float32)
    _tg.sgemm(a, b, out, m, k, n, _NT_TG[0])
    return out


def mm(a, b):
    """a @ b, dividido em threads quando configurar_hilos() ligou. a,b: 2D, ou 3D com o mesmo 1o eixo (cabecas)."""
    if a.ndim == 2 and b.ndim == 2 and _tg_ok(a, b) and a.shape[1] == b.shape[0]:
        return _tg_mm(a, b)
    if a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0] and _tg_ok(a, b):
        return np.stack([_tg_mm(a[i], b[i]) for i in range(a.shape[0])])
    k = _HILOS
    if k <= 1:
        return a @ b
    if a.ndim == 3 and b.ndim == 3 and a.shape[0] == b.shape[0] and a.shape[0] >= 2:
        out = np.empty((a.shape[0], a.shape[1], b.shape[2]), np.result_type(a, b))
        fut = [_POOL.submit(np.matmul, a[i:j], b[i:j], out[i:j]) for i, j in _fatias(a.shape[0], min(k, a.shape[0]))]
        for f in fut:
            f.result()
        return out
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] * b.shape[1] * a.shape[1] < 3_000_000:
        return a @ b
    if a.shape[0] < b.shape[1]:                      # poucas linhas e muitas colunas: divide as colunas de b
        out = np.empty((a.shape[0], b.shape[1]), np.result_type(a, b))
        fut = [_POOL.submit(lambda i, j: out.__setitem__((slice(None), slice(i, j)), a @ b[:, i:j]), i, j)
               for i, j in _fatias(b.shape[1], k)]
        for f in fut:
            f.result()
        return out
    out = np.empty((a.shape[0], b.shape[1]), np.result_type(a, b))
    fut = [_POOL.submit(np.matmul, a[i:j], b, out[i:j]) for i, j in _fatias(a.shape[0], k)]
    for f in fut:
        f.result()
    return out


def configurar_hilos(n=None, log=None, forcar=None):
    """Mede 1 thread x N threads numa matmul grande e liga so se ganhar >= 1,3x. forcar=True/False ignora a medida.
    Devolve o numero de threads em uso. Idempotente (mede so na 1a vez)."""
    global _HILOS, _POOL
    import os
    if getattr(configurar_hilos, "feito", False) and forcar is None:
        return _HILOS
    configurar_hilos.feito = True
    nuc = n or os.cpu_count() or 1
    if forcar is False or nuc < 2:
        _HILOS = 1
        return 1
    from concurrent.futures import ThreadPoolExecutor
    _POOL = ThreadPoolExecutor(max_workers=nuc)
    if forcar:
        _HILOS = nuc
        return nuc
    a = np.random.rand(768, 768).astype(np.float32)
    b = np.random.rand(768, 768).astype(np.float32)

    def tempo(k):
        global _HILOS
        _HILOS = k
        mm(a, b)
        t = time.time()
        for _ in range(3):
            mm(a, b)
        return (time.time() - t) / 3
    t1, tn = tempo(1), tempo(nuc)
    _HILOS = nuc if tn * 1.3 <= t1 else 1
    if log:
        log("matmul: 1 thread %.0f ms, %d threads %.0f ms -> %s" % (t1 * 1e3, nuc, tn * 1e3,
                                                                  "usando %d threads" % _HILOS if _HILOS > 1 else "1 thread (BLAS ja e rapido)"))
    return _HILOS


# ------------------------------------------------------------------ operacoes
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu(x):                                         # aproximacao tanh (erro ~1e-3)
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def linear(x, w, b=None):
    y = mm(x, w.T)
    return y + b if b is not None else y


def conv2d(x, w, b=None, stride=1, pad=0, linhas=16):
    """x (C,H,W), w (O,C,kh,kw). Anda `linhas` linhas de saida por vez (pouca RAM)."""
    C, H, W = x.shape
    O, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad))) if pad else x
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (W + 2 * pad - kw) // stride + 1
    out = np.empty((O, Ho, Wo), np.float32)
    wm = w.reshape(O, -1)
    jan = sliding_window_view(xp, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]
    for r0 in range(0, Ho, linhas):
        r1 = min(Ho, r0 + linhas)
        cols = jan[:, r0:r1].transpose(1, 2, 0, 3, 4).reshape((r1 - r0) * Wo, C * kh * kw)
        out[:, r0:r1, :] = mm(cols, wm.T).reshape(r1 - r0, Wo, O).transpose(2, 0, 1)
    if b is not None:
        out += b[:, None, None]
    return out


def group_norm(x, g, b, grupos=32, eps=1e-5):
    C, H, W = x.shape
    xg = x.reshape(grupos, -1)
    m = xg.mean(axis=1, keepdims=True)
    v = xg.var(axis=1, keepdims=True)
    y = ((xg - m) / np.sqrt(v + eps)).reshape(C, H, W)
    return y * g[:, None, None] + b[:, None, None]


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def atencao(q, k, v, cabecas, bloco=256):
    """q (N,D), k/v (M,D). Softmax por blocos de consultas (nao monta a matriz NxM inteira)."""
    N, D = q.shape
    d = D // cabecas
    qh = q.reshape(N, cabecas, d).transpose(1, 0, 2)
    kh = k.reshape(-1, cabecas, d).transpose(1, 2, 0)
    vh = v.reshape(-1, cabecas, d).transpose(1, 0, 2)
    out = np.empty((cabecas, N, d), np.float32)
    esc = 1.0 / np.sqrt(d)
    for i in range(0, N, bloco):
        s = mm(qh[:, i:i + bloco], kh) * esc
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, i:i + bloco] = mm(p, vh)
    return out.transpose(1, 0, 2).reshape(N, D)


def conv2d_desl(x, w, b=None, bloco=4096):
    """Convolucao 3x3, passo 1, borda 1, SEM im2col: 9 multiplicacoes de matriz sobre fatias do plano achatado
    (sem copiar). 2,2x a 2,8x mais rapida que conv2d quando ha POUCOS canais (ex.: TAESD, 64 canais) e igual
    nos canais grandes. Resultado identico (diferenca ~4e-6)."""
    C, H, W = x.shape
    O = w.shape[0]
    Wp = W + 2
    xp = np.zeros((C, H + 3, Wp), np.float32)            # +1 linha extra: as fatias deslocadas nunca saem do vetor
    xp[:, 1:H + 1, 1:W + 1] = x
    xf = xp.reshape(C, -1)
    L = H * Wp
    wk = np.ascontiguousarray(w.transpose(2, 3, 0, 1).reshape(9, O, C), dtype=np.float32)
    out = np.empty((O, L), np.float32)
    for c0 in range(0, L, bloco):
        c1 = min(L, c0 + bloco)
        acc = mm(wk[0], xf[:, c0:c1])
        for k in range(1, 9):
            off = (k // 3) * Wp + (k % 3)
            acc += mm(wk[k], xf[:, off + c0:off + c1])
        out[:, c0:c1] = acc
    out = out.reshape(O, H, Wp)[:, :, :W]
    if b is not None:
        out = out + b[:, None, None]
    return np.ascontiguousarray(out, np.float32)


def sobe2x(x):
    return x.repeat(2, axis=1).repeat(2, axis=2)


def _b(P, nome):
    return P(nome) if P.tem(nome) else None


# ------------------------------------------------------------------ blocos (nomes do diffusers)
def bloco_resnet(x, temb, P, p, grupos=32):
    h = silu(group_norm(x, P(p + ".norm1.weight"), P(p + ".norm1.bias"), grupos))
    h = conv2d(h, P(p + ".conv1.weight"), P(p + ".conv1.bias"), pad=1)
    if temb is not None and P.tem(p + ".time_emb_proj.weight"):
        t = linear(silu(temb), P(p + ".time_emb_proj.weight"), P(p + ".time_emb_proj.bias"))
        h = h + t[:, None, None]
    h = silu(group_norm(h, P(p + ".norm2.weight"), P(p + ".norm2.bias"), grupos))
    h = conv2d(h, P(p + ".conv2.weight"), P(p + ".conv2.bias"), pad=1)
    if P.tem(p + ".conv_shortcut.weight"):
        x = conv2d(x, P(p + ".conv_shortcut.weight"), P(p + ".conv_shortcut.bias"))
    return x + h


def _proj(h, P, nome):                               # conv 1x1 ou linear, conforme o peso
    w = P(nome + ".weight")
    b = _b(P, nome + ".bias")
    return linear(h, w.reshape(w.shape[0], -1), b)


def _atn(h, ctx, P, p, cabecas):
    ctx = h if ctx is None else ctx
    q = linear(h, P(p + ".to_q.weight"))
    k = linear(ctx, P(p + ".to_k.weight"))
    v = linear(ctx, P(p + ".to_v.weight"))
    o = atencao(q, k, v, cabecas)
    return linear(o, P(p + ".to_out.0.weight"), P(p + ".to_out.0.bias"))


def bloco_transformer(x, ctx, P, p, cabecas=8, grupos=32):
    C, H, W = x.shape
    h = group_norm(x, P(p + ".norm.weight"), P(p + ".norm.bias"), grupos, eps=1e-6)
    h = h.reshape(C, H * W).T
    h = _proj(h, P, p + ".proj_in")
    t = p + ".transformer_blocks.0"
    h = h + _atn(layer_norm(h, P(t + ".norm1.weight"), P(t + ".norm1.bias")), None, P, t + ".attn1", cabecas)
    h = h + _atn(layer_norm(h, P(t + ".norm2.weight"), P(t + ".norm2.bias")), ctx, P, t + ".attn2", cabecas)
    f = layer_norm(h, P(t + ".norm3.weight"), P(t + ".norm3.bias"))
    f = linear(f, P(t + ".ff.net.0.proj.weight"), P(t + ".ff.net.0.proj.bias"))
    a, gate = np.split(f, 2, axis=-1)
    f = linear(a * gelu(gate), P(t + ".ff.net.2.weight"), P(t + ".ff.net.2.bias"))
    h = _proj(h + f, P, p + ".proj_out")
    return x + h.T.reshape(C, H, W)


def embedding_tempo(t, dim, P, p):
    metade = dim // 2
    freq = np.exp(-np.log(10000.0) * np.arange(metade) / metade)
    a = t * freq
    e = np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)
    e = silu(linear(e, P(p + ".linear_1.weight"), P(p + ".linear_1.bias")))
    return linear(e, P(p + ".linear_2.weight"), P(p + ".linear_2.bias"))


def mini_unet(x, t, ctx, P, canais=64, cabecas=4):
    """UNet pequeno (1 nivel) com os MESMOS blocos do SD, para provar o motor por partes."""
    temb = embedding_tempo(t, canais, P, "time_embedding")
    h = conv2d(x, P("conv_in.weight"), P("conv_in.bias"), pad=1)
    h = bloco_resnet(h, temb, P, "down.resnets.0", grupos=8)
    h = bloco_transformer(h, ctx, P, "down.attn.0", cabecas, grupos=8)
    pula = h
    h = conv2d(h, P("down.downsample.weight"), P("down.downsample.bias"), stride=2, pad=1)
    h = bloco_resnet(h, temb, P, "mid.resnets.0", grupos=8)
    h = sobe2x(h)
    h = conv2d(h, P("up.upsample.weight"), P("up.upsample.bias"), pad=1)
    h = bloco_resnet(np.concatenate([h, pula]), temb, P, "up.resnets.0", grupos=8)
    h = silu(group_norm(h, P("norm_out.weight"), P("norm_out.bias"), 8))
    return conv2d(h, P("conv_out.weight"), P("conv_out.bias"), pad=1)


def pesos_aleatorios_mini(canais=64, ctx_dim=32, entrada=4, seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s, e=0.05: (r.standard_normal(s) * e).astype(np.float32)
    uns = lambda c: np.ones(c, np.float32)
    z = lambda c: np.zeros(c, np.float32)
    C, T = canais, canais * 4
    s = {}

    def res(p, cin, cout):
        s[p + ".norm1.weight"], s[p + ".norm1.bias"] = uns(cin), z(cin)
        s[p + ".conv1.weight"], s[p + ".conv1.bias"] = n(cout, cin, 3, 3), z(cout)
        s[p + ".time_emb_proj.weight"], s[p + ".time_emb_proj.bias"] = n(cout, T), z(cout)
        s[p + ".norm2.weight"], s[p + ".norm2.bias"] = uns(cout), z(cout)
        s[p + ".conv2.weight"], s[p + ".conv2.bias"] = n(cout, cout, 3, 3), z(cout)
        if cin != cout:
            s[p + ".conv_shortcut.weight"], s[p + ".conv_shortcut.bias"] = n(cout, cin, 1, 1), z(cout)

    s["time_embedding.linear_1.weight"], s["time_embedding.linear_1.bias"] = n(T, C), z(T)
    s["time_embedding.linear_2.weight"], s["time_embedding.linear_2.bias"] = n(T, T), z(T)
    s["conv_in.weight"], s["conv_in.bias"] = n(C, entrada, 3, 3), z(C)
    res("down.resnets.0", C, C)
    p = "down.attn.0"
    s[p + ".norm.weight"], s[p + ".norm.bias"] = uns(C), z(C)
    s[p + ".proj_in.weight"], s[p + ".proj_in.bias"] = n(C, C, 1, 1), z(C)
    t = p + ".transformer_blocks.0"
    for k in (1, 2, 3):
        s["%s.norm%d.weight" % (t, k)], s["%s.norm%d.bias" % (t, k)] = uns(C), z(C)
    for a, dc in (("attn1", C), ("attn2", ctx_dim)):
        s["%s.%s.to_q.weight" % (t, a)] = n(C, C)
        s["%s.%s.to_k.weight" % (t, a)] = n(C, dc)
        s["%s.%s.to_v.weight" % (t, a)] = n(C, dc)
        s["%s.%s.to_out.0.weight" % (t, a)], s["%s.%s.to_out.0.bias" % (t, a)] = n(C, C), z(C)
    s[t + ".ff.net.0.proj.weight"], s[t + ".ff.net.0.proj.bias"] = n(C * 8, C), z(C * 8)
    s[t + ".ff.net.2.weight"], s[t + ".ff.net.2.bias"] = n(C, C * 4), z(C)
    s[p + ".proj_out.weight"], s[p + ".proj_out.bias"] = n(C, C, 1, 1), z(C)
    s["down.downsample.weight"], s["down.downsample.bias"] = n(C, C, 3, 3), z(C)
    res("mid.resnets.0", C, C)
    s["up.upsample.weight"], s["up.upsample.bias"] = n(C, C, 3, 3), z(C)
    res("up.resnets.0", 2 * C, C)
    s["norm_out.weight"], s["norm_out.bias"] = uns(C), z(C)
    s["conv_out.weight"], s["conv_out.bias"] = n(entrada, C, 3, 3), z(entrada)
    return s


def medir_gflops(n=512, vezes=5):
    """Mede quantos GFLOPS ESTE aparelho faz com numpy (rode no celular e me mande o numero)."""
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b
    t = time.time()
    for _ in range(vezes):
        a @ b
    seg = (time.time() - t) / vezes
    return 2.0 * n ** 3 / seg / 1e9


def relatorio_aparelho():
    import os, platform
    g = medir_gflops()
    print("Python", platform.python_version(), "| numpy", np.__version__, "| nucleos", os.cpu_count())
    print("Velocidade de calculo: %.1f GFLOPS" % g)
    print("Tempo estimado = (GFLOP do modelo por passo) / %.1f  segundos" % g)
    return g


if __name__ == "__main__":
    relatorio_aparelho()
