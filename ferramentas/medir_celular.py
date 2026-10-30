"""Mede a velocidade do celular (GFLOPS) so com numpy. Cole inteiro no Pydroid 3 e rode."""
import time, os, platform
import numpy as np

def medir(n=256, repeticoes=8):
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b                                   # aquecimento
    t = time.time()
    for _ in range(repeticoes):
        a @ b
    dt = time.time() - t
    return 2.0 * n ** 3 * repeticoes / dt / 1e9

print("Python", platform.python_version(), "| numpy", np.__version__, "| nucleos", os.cpu_count())
v = [medir() for _ in range(3)]
g = max(v)
print("GFLOPS (3 medidas):", ["%.1f" % x for x in v])
print("Resultado: %.1f GFLOPS" % g)
# estimativa grosseira: ~150 GFLOP por passo (tiny-sd 256px, com guia = 2x) -> 4 passos
print("Estimativa de tempo para 1 imagem (4 passos): ~%.0f minutos" % (4 * 150 / max(g, 0.01) / 60))
