"""Perfil do aparelho (autoevolutivo): mede o celular, escolhe o melhor motor/resolucao e aprende com cada imagem gerada.
Instalou num celular melhor? Na 1a geracao ele mede de novo e passa a usar 512 px, mais RAM e o motor mais rapido sozinho.
Guarda tudo em perfil_aparelho.json (so neste celular)."""
import json
import os
import statistics
import time

TRABALHO_GFLOP = {256: 1100.0, 512: 2400.0}     # trabalho do modelo por imagem (medido no A5: ~44 s a 25 GFLOPS e ~96 s)
VALIDADE_S = 14 * 86400                          # mede de novo a cada 14 dias
LIMITE_APP_256_S = 90                            # se o motor do app faz 256 px em ate 90 s, nao precisa do Pydroid
LIMITE_512_S = 240                               # 512 px so se levar ate 4 min por imagem


def ram_mb():
    """(total, disponivel) em MB, lendo /proc/meminfo (Android/Linux). (0, 0) se nao der."""
    try:
        d = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, v = ln.split(":")
                d[k] = int(v.split()[0]) / 1024.0
        return d.get("MemTotal", 0), d.get("MemAvailable", d.get("MemFree", 0))
    except Exception:
        return 0.0, 0.0


def medir_gflops():
    """GFLOPS reais do motor do app (usa a mesma multiplicacao do motor, com varios nucleos se ajudar)."""
    import numpy as np
    import sys
    mot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motor_offline")
    if os.path.isdir(mot) and mot not in sys.path:
        sys.path.insert(0, mot)                              # o motor mora em motor_offline/ (antes a medida nao o achava)
    try:
        import motor_cubico as mc
        mc.configurar_hilos()
        mm = mc.mm
    except Exception:
        mm = np.matmul
    a = np.random.rand(512, 512).astype(np.float32)
    b = np.random.rand(512, 512).astype(np.float32)
    mm(a, b)
    t = time.time()
    for _ in range(4):
        mm(a, b)
    return 2.0 * 512 ** 3 / ((time.time() - t) / 4) / 1e9


MOTOR_GERACAO = "k2"        # troque ao mudar o motor de contas: a medida e os tempos antigos deixam de valer


def versao_motor():
    try:
        import ttgemm  # noqa: F401
        return "tg-" + MOTOR_GERACAO
    except Exception:
        return "np-" + MOTOR_GERACAO


def diagnostico(pasta_modelo=None):
    """Mede onde o tempo vai: matmul (GFLOPS), conta simples (milhoes de numeros/s) e leitura de arquivo (MB/s)."""
    import numpy as np
    d = {"gflops": round(medir_gflops(), 1)}
    x = np.random.rand(4_000_000).astype(np.float32)
    y = x.copy()
    t = time.time()
    for _ in range(5):
        y = y * 1.0001 + x
    d["conta_simples_mps"] = round(5 * 2 * 4.0 / (time.time() - t + 1e-9), 0)      # M numeros/s (mul+add)
    q = (np.random.rand(4_000_000) * 255).astype(np.uint8)
    t = time.time()
    for _ in range(3):
        w = (q.astype(np.float32) - 128.0) * 0.01
    d["dequant_mps"] = round(3 * 4.0 / (time.time() - t + 1e-9), 0)                # M pesos/s (int8 -> float)
    try:
        if pasta_modelo:
            arq = os.path.join(pasta_modelo, "sdxs_texto_q8.ttq")
            t = time.time()
            n = 0
            with open(arq, "rb") as f:
                while n < 64 * 1024 * 1024:
                    b = f.read(4 * 1024 * 1024)
                    if not b:
                        break
                    n += len(b)
            d["leitura_mbs"] = round(n / 1048576.0 / (time.time() - t + 1e-9), 0)
    except Exception:
        pass
    return d


class Aparelho:
    def __init__(self, pasta, medidor=medir_gflops, relogio=time.time):
        self.arq = os.path.join(pasta, "perfil_aparelho.json")
        self.medidor, self.relogio = medidor, relogio
        self.d = {}
        try:
            with open(self.arq, encoding="utf-8") as f:
                self.d = json.load(f)
        except Exception:
            self.d = {}

    def _salvar(self):
        try:
            with open(self.arq, "w", encoding="utf-8") as f:
                json.dump(self.d, f)
        except Exception:
            pass

    def medir_se_preciso(self, forcar=False):
        """Mede na 1a vez, a cada 14 dias, ou quando o aparelho mudou (RAM/nucleos diferentes). Devolve True se mediu."""
        tot, liv = ram_mb()
        nuc = os.cpu_count() or 1
        mudou = abs(self.d.get("ram_total", 0) - tot) > 256 or self.d.get("nuc") != nuc
        motor_novo = self.d.get("motor_v") != versao_motor()          # APK novo com motor diferente: mede de novo
        velho = self.relogio() - self.d.get("quando", 0) > VALIDADE_S
        if not forcar and self.d.get("gflops") and not mudou and not velho and not motor_novo:
            self.d["ram_livre"] = liv
            return False
        hist = {} if (mudou or motor_novo) else self.d.get("hist", {})   # aparelho/motor novo: esquece o que aprendeu antes
        self.d = {"gflops": float(self.medidor()), "ram_total": tot, "ram_livre": liv, "nuc": nuc,
                  "quando": self.relogio(), "hist": hist, "motor_v": versao_motor(), "sem_512": False if mudou else self.d.get("sem_512", False)}
        self._salvar()
        return True

    def estimar(self, lado, motor="app"):
        """Segundos esperados para 1 imagem: usa o que ja foi medido de verdade (mediana das ultimas 3); senao a conta."""
        h = self.d.get("hist", {}).get("%s:%d" % (motor, lado), [])
        if h:
            return statistics.median(h[-3:])
        if motor == "pydroid":
            return 45.0 if lado == 256 else 95.0                   # medido no A5; melhora/piora conforme o historico
        return TRABALHO_GFLOP[lado] / max(self.d.get("gflops", 1.0), 0.1)

    def registrar(self, motor, lado, segundos, falta_memoria=False):
        """Aprende: guarda o tempo real; se faltou memoria em 512, nao tenta 512 de novo."""
        if falta_memoria:
            if lado == 512:
                self.d["sem_512"] = True
        else:
            h = self.d.setdefault("hist", {}).setdefault("%s:%d" % (motor, lado), [])
            h.append(round(float(segundos), 1))
            del h[:-6]
        self._salvar()

    def decidir(self, qualidade="max", servidor_ativo=False):
        """Escolhe motor, resolucao e cache. Devolve dict: motor, lado, cache_mb, est_s, texto."""
        liv = self.d.get("ram_livre", 0)
        app_256 = self.estimar(256, "app")
        motor = "app" if (app_256 <= LIMITE_APP_256_S or not servidor_ativo) else "pydroid"
        lado = 256
        if (qualidade == "max" and not self.d.get("sem_512") and self.estimar(512, motor) <= LIMITE_512_S
                and (liv >= 900 or liv == 0)):
            lado = 512
        cache = 2200 if liv >= 4500 else 0                         # RAM sobrando: mantem os pesos prontos (videos mais rapidos)
        est = self.estimar(lado, motor)
        txt = "🧠 Aparelho: %.1f GFLOPS, %.1f GB livres → motor %s, %d px (~%s)" % (
            self.d.get("gflops", 0), liv / 1024.0, "do app" if motor == "app" else "rapido do Pydroid", lado,
            ("%d s" % est) if est < 120 else ("%.0f min" % (est / 60.0)))
        return {"motor": motor, "lado": lado, "cache_mb": cache, "est_s": est, "texto": txt}
