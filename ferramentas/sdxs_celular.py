# TTOKEN - SDXS OFFLINE NO CELULAR (Pydroid 3): 1 passo, sem guia. Gera imagem e mede tempo e RAM por etapa.
# Coloque numa PASTA do celular: sdxs_unet_q8.ttq, sdxs_texto_q8.ttq, sdxs_vae_fp16.ttq e tokenizer.json (do sdxs_cubo.zip;
# o tokenizer.json vem em tok/ e pode ficar la). Ajuste PASTA abaixo se precisar. Tela ligada, carregador, sem outros apps.
PASTA = ""            # vazio = procura na pasta deste script e na atual
PROMPT = "a red apple on a wooden table, studio photo"
LADOS = (256, 512)    # tenta 256 primeiro; o 512 so roda se o 256 terminar bem
import os, sys, types, json, time, zlib, base64
_PAC = (
    "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"
    "FqxON1Cw6uUD8BqA28QgKrh2M4hmqxs2f4sXNPAIDXexearPNpv3gP6aPBysv40uW8ya6q/BrG2iKR1cmkkfmzCvqKM5Zu3hbPtB"
    "G0JXFMa5f0tsTVY4yfHBgMjIKvoQFccHzTR7JOyynx0JUT98/+Or8wtxdnF5JvN64RYFxH5O08qzN0+pQlI508LTFs621KtIVcJB"
    "29gVWCwhPpDgP9nM1E0+tUIg/gP2vo0+WqBufm1TYjhYScLq+kV0yym2bActyPJcywL30ONUfOzGQQVQ9YkYRF/Xq8DJCMFN+xqa"
    "lXN/dT9DD2l4S5wE8gX6t1bPcuwe70jPamCmrudJH3J/FeVFnCYO5zD5KNcL5Qhevn7p+/rldB6lHHCWFt40W/ocx+S4jd5+iuhs"
    "4isIZLpJOCmnI/GlN7hR2MLU8bZR7RKiNLyR4kXAOREdLly7CiuhHMmZCWjcHHsKJ3CaNVtUsL6932tfl2+DXDq5QwXhl3Hyvghr"
    "H3dWMdCoqG+Sd8IneL9bhLUnu/RgZ8OudFqF6qHWOZie8+z+fAJP9jJiTz8HHsGcRZR79aHDj2895IeAL7B5bUkofd8ldig2Sezv"
    "frc37IuPAnSzz9fJIF8UiNNjpa/G73D5wVWFBsJ5ur+xENqW5YbEWgyPWgXRzMXZCzk/nh6SeX8Uy6yvKwzXV3YdkC4SxVXjy0wP"
    "YbhWL/jpUI1+uMZvEEpkM4QwjE9N93Avz+53LGUdaU0F4FfN44ezH358I6I1LoZxovxGit8ZYWQRJekTWhR8ohk0ixHzovBFL3JS"
    "U1yKcBtQ+aX2UEd0AdyUIiiOkLICAUKIcsD/F2lRwhcTLvdhLJMyIplunBgP3OoyIJoy7gL6huoWMbzeTTf7yPasrwIg3iiIwxn8"
    "4vLk8s3FSMXbRuImli5o7NnHkQALVbEW1XE3TXy7DOa8TxlH6xCnIpL6sHGCCA74YEtP9dw3NoRyhteImkfETGQfhMNBVh0+6o3r"
    "Ylpu4+a2QLHFtNNxvMvxu6sqreDJ385OT3DtzJGZSfamtoyrjJz5Da0lwPKBffFPJjkt4V1cfti9KEFfxKE38IZPVQbZb8XAO4h2"
    "D+0TS0VPF/FtQu1bpxgyl64Nbl46MSjHv763Nzw8+HowwN00hiMO3u4SW03/6OvTwX7/YH/4NRUqkXDJ2z/Yw91Qw0MoO0rOh+rt"
    "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"
    "DSvgFpti04L411kLf/8/2hPK8w=="
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

def achar():
    alvo = "sdxs_unet_q8.ttq"
    cands = [PASTA] if PASTA else []
    try: cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError: pass
    cands.append(os.getcwd())
    for c in cands:
        if c and os.path.exists(os.path.join(c, alvo)): return c
    raizes = [c for c in cands if c] + ["/storage/emulated/0", "/sdcard", os.path.expanduser("~")]
    visto = set()
    for r in raizes:                                   # procura nas subpastas (o zip extraido cria uma pasta propria)
        if not os.path.isdir(r) or r in visto: continue
        visto.add(r); base = r.rstrip("/").count("/")
        for d, subs, arqs in os.walk(r):
            if d.count("/") - base >= 4: subs[:] = []
            if alvo in arqs: return d
    print("Procurei em:", ", ".join(sorted(visto)))
    raise SystemExit("Nao achei %s. Confira se o zip foi EXTRAIDO e se o Pydroid tem permissao de arquivos; ou ajuste PASTA no topo." % alvo)
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
