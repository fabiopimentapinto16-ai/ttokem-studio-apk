import os
aqui=os.path.dirname(os.path.abspath(__file__))
C=open(os.path.join(aqui,'cubo_quant.py'),encoding='utf-8').read()
M=open(os.path.join(aqui,'motor_cubico.py'),encoding='utf-8').read()
cab='''# TTOKEN - MEDE O TEMPO REAL DO MOTOR NO CELULAR (Pydroid 3). Sem baixar peso nenhum. Arquivo UNICO, so numpy.
# Roda as operacoes do tiny-sd com os tamanhos REAIS (pesos aleatorios) e projeta o tempo de uma imagem 256x256, 4 passos.
# Leva ~1-2 minutos. Me mande TODO o texto que ele imprimir.
import sys, types, time, os
import numpy as np
'''
corpo="_C=%r\n_M=%r\n"%(C,M)+'''
def _mod(n, c):
    m = types.ModuleType(n); sys.modules[n] = m; exec(c, m.__dict__); return m
cq = _mod("cubo_quant", _C); mc = _mod("motor_cubico", _M)
rng = np.random.RandomState(0)
def tempo(f, rep=2):
    f(); t = time.time()
    for _ in range(rep): f()
    return (time.time() - t) / rep
print("nucleos:", os.cpu_count(), "| numpy", np.__version__)

print("\\n--- convolucao 3x3 (a operacao que mais pesa no UNet) ---")
rates = []
for C_, H in ((320, 32), (640, 16), (1280, 8)):
    x = rng.randn(C_, H, H).astype("f"); w = (rng.randn(C_, C_, 3, 3) * 0.02).astype("f")
    dt = tempo(lambda: mc.conv2d(x, w, None, 1, 1))
    gf = 2 * C_ * C_ * 9 * H * H / dt / 1e9; rates.append(gf)
    print("  %4d canais %2dx%-2d : %6.2f s   %5.2f GFLOPS" % (C_, H, H, dt, gf))
conv = float(np.mean(rates))

print("\\n--- multiplicacao de matriz (atencao e camadas lineares) ---")
a = rng.randn(1024, 640).astype("f"); b = rng.randn(640, 2560).astype("f")
dt = tempo(lambda: a @ b); mm = 2 * 1024 * 640 * 2560 / dt / 1e9
print("  1024x640 @ 640x2560 : %.2f s   %.2f GFLOPS" % (dt, mm))

print("\\n--- desquantizar pesos do .ttq (8 bits, cubo 4x4) ---")
w = (rng.randn(1280, 1280) * 0.02).astype("f")
pack, esc, R, Cc = cq.quantizar(w, (4, 4), 8)
dt = tempo(lambda: cq.desquantizar(pack, esc, R, Cc, (R, Cc), (4, 4), np.float32, 8))
dq = w.size / dt / 1e6
print("  %.1f milhoes de pesos por segundo" % dq)

print("\\n--- PROJECAO (UNet tiny-sd ~240 GFLOP por passo com guia; VAE ~300 GFLOP; UNet tem 308 M pesos) ---")
mix = 1.0 / (0.75 / conv + 0.25 / mm)          # ~75% convolucao, ~25% matriz (estimativa)
t_calc = 240 / mix; t_dq = 308 / dq
passo = t_calc + t_dq
vae = 300 / conv
print("  (otimista = so convolucao/matriz; realista = x2.5, pois normalizacao/softmax/GELU/copias tambem custam)")
for passos in (1, 2, 4):
    otim = passos * passo + vae + 8
    print("  %d passo(s): otimista %.1f min | realista ~%.1f min" % (passos, otim / 60, (passos * (t_calc * 2.5 + t_dq) + vae * 2.5 + 8) / 60))
print("\\nNOTA: projecao com 1 processo (75 por cento conv / 25 por cento matriz). Depois eu meco a imagem inteira com os .ttq.")
print("Me mande este texto inteiro.")
'''
open(os.path.join(aqui,'medir_tempo_celular.py'),'w',encoding='utf-8').write(cab+corpo)
