"""Gera sdxs_celular.py: arquivo unico para o Pydroid 3 (modulos embutidos). Roda o SDXS offline e mede tempo/RAM por etapa."""
import os, json, base64, zlib
aqui = os.path.dirname(os.path.abspath(__file__))
mods = ["cubo_quant", "motor_cubico", "unet_tinysd", "clip_vae_tinysd", "pipeline_tinysd", "unet_sdxs", "taesd_numpy", "pipeline_sdxs"]
src = {m: open(os.path.join(aqui, m + ".py"), encoding="utf-8").read() for m in mods}
b64 = base64.b64encode(zlib.compress(json.dumps(src).encode("utf-8"), 9)).decode()
emb = "_PAC = (\n" + "\n".join('    "%s"' % b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n)\n"
corpo = r'''# TTOKEN - SDXS OFFLINE NO CELULAR (Pydroid 3): 1 passo, sem guia. Gera imagem e mede tempo e RAM por etapa.
# Coloque numa PASTA do celular: sdxs_unet_q8.ttq, sdxs_texto_q8.ttq, sdxs_vae_fp16.ttq e tokenizer.json (do sdxs_cubo.zip;
# o tokenizer.json vem em tok/ e pode ficar la). Ajuste PASTA abaixo se precisar. Tela ligada, carregador, sem outros apps.
PASTA = ""            # vazio = procura na pasta deste script e na atual
PROMPT = "a red apple on a wooden table, studio photo"
LADOS = (256, 512)    # tenta 256 primeiro; o 512 so roda se o 256 terminar bem
import os, sys, types, json, time, zlib, base64
''' + emb + r'''
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
'''
open(os.path.join(aqui, "sdxs_celular.py"), "w", encoding="utf-8").write(corpo)
print("ok", len(corpo))
