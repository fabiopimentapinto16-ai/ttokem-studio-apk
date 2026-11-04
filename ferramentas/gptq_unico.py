# TTOKEN - GPTQ NOS CUBOS: 4 e 3 bits por peso SEM destruir a imagem? (meta: o mais perto possivel de 5x)
# GPTQ = quantiza coluna a coluna e COMPENSA o erro nas colunas seguintes usando as ativacoes REAIS do UNet (calibracao).
# Compara, em imagens com sementes que a calibracao nao viu: RTN 4b, GPTQ 4b, GPTQ 3b, GPTQ 4b cubo 4x8,
# e GPTQ com as camadas mais frageis protegidas em 8 bits. Tamanho = formula (bits + escala fp16 por cubo).
# RAPIDO = False: rodada completa (~60-90 min). RAPIDO = True: teste de fumaca (~10 min; so 11 camadas com GPTQ, NAO compara tamanhos).
# conv_in/conv_out ficam SEMPRE em 8 bits 4x4 (minusculas e as mais sensiveis; GPTQ quase nao ajuda na conv_in).
# Tela ligada, aba aberta (no celular: 'Site para computador').
RAPIDO = False
import os, sys, time, subprocess, json, types, zlib, base64
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "scikit-image"])
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_GQ = (
    "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"
    "aysQ64sAvE3SO5zjPIsBRMZgTO226fT+AzF9ym8="
)

gq = types.ModuleType("gptq_cubos"); exec(zlib.decompress(base64.b64decode(_GQ)).decode("utf-8"), gq.__dict__)
from diffusers import AutoPipelineForText2Image
PROMPT, STEPS, SIZE = "Carro futurista correndo na chuva cyberpunk", 4, 256
FIT_SEEDS = (1,) if RAPIDO else (5762, 1, 11, 12, 13)     # calibracao (trajetorias reais do UNet)
SEEDS = (3,) if RAPIDO else (3, 4, 5)                       # validacao: sementes que a calibracao NAO viu
CHUNK = 4096                                                # Hessiana em blocos diagonais de ate 4096 colunas
OPTS = {"g4": ((4, 16), 4), "g3": ((4, 16), 3)} if RAPIDO else {"g4": ((4, 16), 4), "g3": ((4, 16), 3), "g4b": ((4, 8), 4)}
BPP = lambda bl, b: b + 16.0 / (bl[0] * bl[1])
pipe = AutoPipelineForText2Image.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"): pipe.safety_checker = None
gerar = lambda s: pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=7.5,
                       generator=torch.Generator("cpu").manual_seed(s), height=SIZE, width=SIZE).images[0]
def psnr(a, b):
    x, y = np.asarray(a.convert("RGB"), np.float64), np.asarray(b.convert("RGB"), np.float64)
    m = np.mean((x - y) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)
def ssim(a, b):
    try:
        from skimage.metrics import structural_similarity as S
        return float(S(np.asarray(a.convert("L")), np.asarray(b.convert("L")), data_range=255))
    except Exception:
        x, y = [np.asarray(i.convert("L"), np.float64) for i in (a, b)]
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2; mx, my = x.mean(), y.mean()
        return float(((2*mx*my+c1)*(2*((x-mx)*(y-my)).mean()+c2))/((mx**2+my**2+c1)*(x.var()+y.var()+c2)))

unet = pipe.unet
mods = dict(unet.named_modules())
alvo = {}
for n, p in unet.named_parameters():
    if p.ndim >= 2 and p.numel() >= 256 and n.endswith(".weight"):
        m = mods.get(n[:-7])
        if isinstance(m, (nn.Linear, nn.Conv2d)) and (not isinstance(m, nn.Conv2d) or m.groups == 1):
            alvo[n] = (p, m)
names = list(alvo); N = len(names)
numel = {n: alvo[n][0].numel() for n in names}
TOTAL = float(sum(numel.values()))
orig = {n: alvo[n][0].data.clone() for n in names}
mat = lambda n: orig[n].numpy().reshape(orig[n].shape[0], -1)
print("UNet: %d camadas quantizaveis, %.1f M pesos (RAPIDO=%s)" % (N, TOTAL / 1e6, RAPIDO), flush=True)

print("0) imagens de referencia ...", flush=True)
base = {}
for s_ in SEEDS:
    base[s_] = gerar(s_); print("   semente %d pronta" % s_, flush=True)
print("0b) gravando chamadas reais do UNet para calibrar ...", flush=True)
trilha = []
def _gancho(mod, args, kwargs, out):
    trilha.append((args[0].detach().clone(), args[1].detach().clone() if torch.is_tensor(args[1]) else torch.tensor(args[1]),
                   kwargs["encoder_hidden_states"].detach().clone()))
for sd in FIT_SEEDS:
    h_ = unet.register_forward_hook(_gancho, with_kwargs=True); gerar(sd); h_.remove()
    print("   semente %d gravada (%d chamadas)" % (sd, len(trilha)), flush=True)

GQ_NOMES = names[::15] if RAPIDO else names               # no modo rapido so algumas camadas usam GPTQ
def custo_H(n): C = int(np.prod(alvo[n][0].shape[1:])); return C * min(C, CHUNK) * 4
ordem = sorted(GQ_NOMES, key=lambda n: -custo_H(n))
passes, atual, usado = [], [], 0
for n in ordem:
    if atual and usado + custo_H(n) > 3e9: passes.append(atual); atual, usado = [], 0
    atual.append(n); usado += custo_H(n)
if atual: passes.append(atual)
print("1) Hessianas em %d passada(s) pelo UNet; GPTQ em %d camadas x %d opcoes ..." % (len(passes), len(GQ_NOMES), len(OPTS)), flush=True)

def coletar(grupo):
    H, cnt, hooks = {}, {}, []
    for n in grupo:
        p, m = alvo[n]; C = int(np.prod(p.shape[1:]))
        H[n] = [torch.zeros(min(CHUNK, C - c0), min(CHUNK, C - c0)) for c0 in range(0, C, CHUNK)]; cnt[n] = 0
        def pre(mod, args, n=n, m=m, C=C):
            x = args[0].detach().float()
            if isinstance(m, nn.Conv2d):
                X = F.unfold(x, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride)
                X = X.permute(0, 2, 1).reshape(-1, X.shape[1])
            else:
                X = x.reshape(-1, x.shape[-1])
            for k, c0 in enumerate(range(0, C, CHUNK)):
                Xk = X[:, c0:c0 + CHUNK]; H[n][k] += Xk.T @ Xk
            cnt[n] += X.shape[0]
        hooks.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        for x, t, enc in trilha: unet(x, t, encoder_hidden_states=enc, return_dict=False)
    for h in hooks: h.remove()
    return H, cnt

def erro_rtn_blocos(W, Q, Hb):
    num = den = 0.0
    for k, c0 in enumerate(range(0, W.shape[1], CHUNK)):
        Wc = W[:, c0:c0 + CHUNK].astype(np.float64); dW = Wc - Q[:, c0:c0 + CHUNK]; H = Hb[k]
        num += float(((dW @ H) * dW).sum()); den += float(((Wc @ H) * Wc).sum())
    return num / max(den, 1e-30)

QS = {o: {} for o in OPTS}; ES = {o: {} for o in OPTS}; ER = {o: {} for o in OPTS}; RASO = set()
feitas, t0 = 0, time.time()
for gi, grupo in enumerate(passes):
    H, cnt = coletar(grupo)
    print("   passada %d/%d: Hessianas prontas (%.0f s)" % (gi + 1, len(passes), time.time() - t0), flush=True)
    for n in grupo:
        W = mat(n); C = W.shape[1]
        if cnt[n] < min(C, CHUNK) / 2:                      # amostras demais poucas para GPTQ: cai para 8 bits
            RASO.add(n); continue
        Hb = [(h / cnt[n]).numpy().astype(np.float64) for h in H[n]]
        pre = [gq.preparar(h) for h in Hb]                    # inversa da Hessiana: UMA vez por camada, serve para todas as opcoes
        for o, (bl, b) in OPTS.items():
            Q, e, S = gq.gptq_blocos(W, Hb, CHUNK, pre=pre, bloco=bl, bits=b)
            QS[o][n] = Q.astype(np.float16); ES[o][n] = e
            ER[o][n] = erro_rtn_blocos(W, gq.rtn(W, bl, b), Hb)
        feitas += 1
        if feitas % 10 == 0 or feitas == len(GQ_NOMES): print("   GPTQ %d/%d camadas (%.0f s)" % (feitas, len(GQ_NOMES), time.time() - t0), flush=True)
    del H
print("   camadas com poucas amostras (usam 8 bits 4x4): %d" % len(RASO), flush=True)

print("\n2) ERRO NA SAIDA DAS CAMADAS (media ponderada por pesos; dB maior = melhor)")
for o, (bl, b) in OPTS.items():
    ks = [n for n in QS[o]]
    w = np.array([numel[n] for n in ks], float)
    f = lambda d: -10 * np.log10(max((w * np.array([d[n] for n in ks])).sum() / w.sum(), 1e-30))
    print("   %-4s %db cubo %dx%d (%.2f bits/peso): RTN %5.1f dB  ->  GPTQ %5.1f dB" % (o, b, bl[0], bl[1], BPP(bl, b), f(ER[o]), f(ES[o])))

def peso_da_cfg(cfg, n):
    c = cfg[n]
    if c[0] == "gptq": return QS[c[1]][n].astype(np.float32)
    if c[0] == "fp16": return mat(n).astype(np.float16).astype(np.float32)
    return gq.rtn(mat(n), c[1], c[2])
def bpp_cfg(c): return 16.0 if c[0] == "fp16" else (BPP(*OPTS[c[1]]) if c[0] == "gptq" else BPP(c[1], c[2]))
def tamanho(cfg): return sum(numel[n] * bpp_cfg(cfg[n]) for n in names) / (16.0 * TOTAL)
SEMPRE8 = {n for n in names if n.startswith("conv_in") or n.startswith("conv_out")}
def cfg_base(tipo, op=None, bl=None, b=None):
    cfg = {}
    for n in names:
        if n in RASO or n in SEMPRE8 or (tipo == "gptq" and n not in QS[op]): cfg[n] = ("rtn", (4, 4), 8)
        else: cfg[n] = ("gptq", op) if tipo == "gptq" else ("rtn", bl, b)
    return cfg
def proteger(cfg, op, alvo_tam):
    cfg = dict(cfg)
    for n in sorted(QS[op], key=lambda n: -ES[op][n]):
        if cfg[n][0] != "gptq": continue
        ant = cfg[n]; cfg[n] = ("rtn", (4, 4), 8)
        if tamanho(cfg) > alvo_tam: cfg[n] = ant; break
    return cfg
def n_prot(cfg): return sum(1 for n in names if cfg[n][0] == "rtn" and cfg[n][1] == (4, 4) and cfg[n][2] == 8)

CONFIGS = [("RTN 4b 4x16 (sem GPTQ)", cfg_base("rtn", bl=(4, 16), b=4)), ("GPTQ 4b 4x16", cfg_base("gptq", "g4")),
           ("GPTQ 3b 4x16", cfg_base("gptq", "g3"))]
if not RAPIDO:
    CONFIGS += [("GPTQ 4b 4x8", cfg_base("gptq", "g4b")),
                ("GPTQ 4b + protege(8b) <=30%", proteger(cfg_base("gptq", "g4"), "g4", 0.30)),
                ("GPTQ 3b + protege(8b) <=25%", proteger(cfg_base("gptq", "g3"), "g3", 0.25)),
                ("GPTQ 3b + protege(8b) <=30%", proteger(cfg_base("gptq", "g3"), "g3", 0.30))]
print("\n3) IMAGENS (%d sementes novas)" % len(SEEDS))
print("%-30s %7s %6s | %6s %7s %6s | camadas em 8b" % ("configuracao", "% fp16", "x fp16", "PSNR", "SSIM", "pior"))
res, imagens = {}, {}
for rot, cfg in CONFIGS:
    for n in names: alvo[n][0].data = torch.from_numpy(peso_da_cfg(cfg, n).reshape(orig[n].shape)).float()
    ims = [gerar(s) for s in SEEDS]
    for n in names: alvo[n][0].data = orig[n].clone()
    ps = [psnr(base[s], i) for s, i in zip(SEEDS, ims)]; ss = [ssim(base[s], i) for s, i in zip(SEEDS, ims)]
    fr = tamanho(cfg)
    print("%-30s %6.1f%% %5.2fx | %5.1f %.4f %.4f | %d" % (rot, 100 * fr, 1 / fr, np.mean(ps), np.mean(ss), np.min(ss), n_prot(cfg)), flush=True)
    res[rot] = {"tamanho_frac": fr, "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "ssim_pior": float(np.min(ss))}
    imagens[rot] = ims[0]
try:
    from PIL import Image
    W_ = SIZE; tela = Image.new("RGB", (W_ * (len(CONFIGS) + 1), W_)); tela.paste(base[SEEDS[0]], (0, 0))
    for k, (rot, _) in enumerate(CONFIGS): tela.paste(imagens[rot], (W_ * (k + 1), 0))
    tela.save("tinysd_gptq_comparacao.png"); print("\nSalvo tinysd_gptq_comparacao.png (original | " + " | ".join(r for r, _ in CONFIGS) + ")")
except Exception as ex:
    print("(sem imagem: %s)" % ex)
json.dump({"res": res, "erro_gptq_db": {o: {n: float(-10 * np.log10(max(v, 1e-30))) for n, v in ES[o].items()} for o in OPTS},
           "erro_rtn_db": {o: {n: float(-10 * np.log10(max(v, 1e-30))) for n, v in ER[o].items()} for o in OPTS}},
          open("tinysd_gptq.json", "w"), indent=1)
try:
    from google.colab import files
    files.download("tinysd_gptq.json"); files.download("tinysd_gptq_comparacao.png")
except Exception:
    pass
print("Me mande TODO este texto, o tinysd_gptq.json e a imagem.")
