# TTOKEN - SDXS OFFLINE NO CELULAR (Pydroid 3): 1 passo, sem guia. Gera imagem e mede tempo e RAM por etapa.
# Coloque numa PASTA do celular: sdxs_unet_q8.ttq, sdxs_texto_q8.ttq, sdxs_vae_fp16.ttq e tokenizer.json (do sdxs_cubo.zip;
# o tokenizer.json vem em tok/ e pode ficar la). Ajuste PASTA abaixo se precisar. Tela ligada, carregador, sem outros apps.
PASTA = ""            # vazio = procura na pasta deste script e na atual
PROMPT = "a red apple on a wooden table, studio photo"
LADOS = (256, 512)    # tenta 256 primeiro; o 512 so roda se o 256 terminar bem
import os, sys, types, json, time, zlib, base64
_PAC = (
    "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"
)

_SRC = json.loads(zlib.decompress(base64.b64decode(_PAC)).decode("utf-8"))
for _n in ("cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"):
    m = types.ModuleType(_n); sys.modules[_n] = m; exec(_SRC[_n], m.__dict__)
import numpy as np
mc, pt, ps = sys.modules["motor_cubico"], sys.modules["pipeline_tinysd"], sys.modules["pipeline_sdxs"]
try:
    import resource
    ram = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except Exception:
    ram = lambda: float("nan")

def completo(d):                                      # so vale a pasta com TUDO (um .ttq solto em Download nao serve)
    return (os.path.exists(os.path.join(d, "sdxs_unet_q8.ttq")) and os.path.exists(os.path.join(d, "sdxs_texto_q8.ttq"))
            and os.path.exists(os.path.join(d, "sdxs_vae_fp16.ttq"))
            and (os.path.exists(os.path.join(d, "tokenizer.json")) or os.path.exists(os.path.join(d, "tok", "tokenizer.json"))))
def achar():
    cands = [PASTA] if PASTA else []
    try: cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError: pass
    cands.append(os.getcwd())
    for c in cands:
        if c and completo(c): return c
    raizes = [c for c in cands if c] + ["/storage/emulated/0", "/sdcard", os.path.expanduser("~")]
    visto = set()
    for r in raizes:                                   # procura nas subpastas (o zip extraido cria uma pasta propria)
        if not os.path.isdir(r) or r in visto: continue
        visto.add(r); base = r.rstrip("/").count("/")
        for d, subs, arqs in os.walk(r):
            if d.count("/") - base >= 4: subs[:] = []
            if "sdxs_unet_q8.ttq" in arqs and completo(d): return d
    print("Procurei em:", ", ".join(sorted(visto)))
    raise SystemExit("Nao achei a pasta COMPLETA (3 arquivos .ttq + tokenizer.json). Ajuste PASTA no topo, ex.: PASTA = '/storage/emulated/0/Download/sdxs_cubo'")
pasta = achar(); print("pasta:", pasta)
try: mc.relatorio_aparelho()
except Exception as ex: print("(sem relatorio do aparelho: %s)" % ex)
res = {}
for lado in LADOS:
    marca = []
    def log(s, marca=marca): marca.append((time.time(), s)); print("   [%s]" % s, "RAM pico %.0f MB" % ram(), flush=True)
    print("\n=== %dx%d, 1 passo ===" % (lado, lado), flush=True)
    t0 = time.time()
    try:
        img = ps.gerar(pasta, PROMPT, semente=1, lado=lado, log=log)
    except MemoryError:
        print("  sem memoria em %d px; parando aqui." % lado); break
    tot = time.time() - t0
    out = os.path.join(pasta, "sdxs_celular_%d.png" % lado); pt.salvar_png(img, out)
    res[lado] = {"segundos": tot, "ram_pico_mb": ram()}
    print("  TOTAL %.1f s | RAM pico %.0f MB | salvo %s" % (tot, ram(), out), flush=True)
json.dump(res, open(os.path.join(pasta, "sdxs_celular.json"), "w"), indent=1)
print("\nMe mande este texto e as imagens sdxs_celular_*.png.")
