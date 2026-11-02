import os
aqui=os.path.dirname(os.path.abspath(__file__))
C=open(os.path.join(aqui,'cubo_quant.py'),encoding='utf-8').read()
M=open(os.path.join(aqui,'motor_cubico.py'),encoding='utf-8').read()
cab='''# TTOKEN - AFINADOR NO CELULAR (Pydroid 3). Testa variantes de desquantizacao e de convolucao NO SEU APARELHO e diz qual e a mais rapida.
# Arquivo UNICO, so numpy, sem baixar peso. ~2-3 min. Me mande TODO o texto impresso.
import sys, types, time, os
import numpy as np
'''
corpo="_C=%r\n_M=%r\n"%(C,M)+'''
def _mod(n, c):
    m = types.ModuleType(n); sys.modules[n] = m; exec(c, m.__dict__); return m
cq = _mod("cubo_quant", _C); mc = _mod("motor_cubico", _M)
rng = np.random.RandomState(0)
def tempo(f, rep=3):
    f(); t = time.time()
    for _ in range(rep): r = f()
    return (time.time() - t) / rep, r

print("nucleos:", os.cpu_count(), "| numpy", np.__version__)
print("\\n=== DESQUANTIZAR 8 bits (milhoes de pesos/s; maior = melhor) ===")
w = (rng.randn(1280, 1280) * 0.02).astype("f")
melhor_dq = {}
for bl in ((4, 4), (8, 8)):
    pack, esc, R, C = cq.quantizar(w, bl, 8)
    bh, bw = bl; nbh, nbw = R // bh, C // bw
    ref = cq.desquantizar(pack, esc, R, C, (R, C), bl, np.float32, 8)
    def v_rep():
        q = pack.reshape(nbh, nbw, bh, bw).transpose(0, 2, 1, 3).reshape(R, C)
        e = np.repeat(np.repeat(esc.astype(np.float32).reshape(nbh, nbw), bh, 0), bw, 1)
        return (q.astype(np.float32) - 128.0) * e
    def v_out():
        e = esc.astype(np.float32).reshape(nbh, 1, nbw, 1)
        out = np.empty((nbh, bh, nbw, bw), np.float32)
        np.subtract(pack.reshape(nbh, nbw, bh, bw).transpose(0, 2, 1, 3), 128, out=out, dtype=np.float32); out *= e
        return out.reshape(R, C)
    cand = [("atual (partes de 65k)", lambda: cq.desquantizar(pack, esc, R, C, (R, C), bl, np.float32, 8))]
    for me in (1 << 14, 1 << 18, 1 << 20, 1 << 22):
        cand.append(("atual, partes de %dk" % (me >> 10), lambda me=me: cq.desquantizar(pack, esc, R, C, (R, C), bl, np.float32, 8, max_elementos=me)))
    cand += [("repeat (reordena em uint8)", v_rep), ("saida pre-alocada", v_out)]
    print("  cubo %dx%d:" % bl)
    for nome, f in cand:
        dt, r = tempo(f)
        ok = "ok" if np.array_equal(r, ref) else "DIFERENTE!"
        print("    %-30s %7.1f M/s  %s" % (nome, w.size / dt / 1e6, ok))
        melhor_dq[(bl, nome)] = w.size / dt / 1e6

print("\\n=== CONVOLUCAO 3x3: quantas linhas de saida por vez (GFLOPS; maior = melhor) ===")
for C_, H in ((320, 32), (640, 16)):
    x = rng.randn(C_, H, H).astype("f"); ww = (rng.randn(C_, C_, 3, 3) * 0.02).astype("f")
    print("  %d canais %dx%d:" % (C_, H, H))
    for ln in (2, 4, 8, 16, 32, 64):
        dt, _ = tempo(lambda: mc.conv2d(x, ww, None, 1, 1, linhas=ln))
        print("    linhas=%-3d %6.2f GFLOPS" % (ln, 2 * C_ * C_ * 9 * H * H / dt / 1e9))

print("\\n=== multiplicacao de matriz (referencia) ===")
a = rng.randn(1024, 640).astype("f"); b = rng.randn(640, 2560).astype("f")
dt, _ = tempo(lambda: a @ b)
print("  %.2f GFLOPS" % (2 * 1024 * 640 * 2560 / dt / 1e9))
print("\\nMe mande este texto inteiro.")
'''
open(os.path.join(aqui,'afinar_celular.py'),'w',encoding='utf-8').write(cab+corpo)
