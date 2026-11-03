import os, base64, zlib
aqui = os.path.dirname(os.path.abspath(__file__))
alo = open(os.path.join(aqui, "alocacao_bits.py"), encoding="utf-8").read()
b64 = base64.b64encode(zlib.compress(alo.encode("utf-8"), 9)).decode()
emb = "_ALOC = (\n" + "\n".join('    "%s"' % b64[i:i + 100] for i in range(0, len(b64), 100)) + "\n)\n"
corpo = r'''# TTOKEN - CURVA v2: ONDE A QUALIDADE QUEBRA quando se desce de 8 para 3 bits por tensor (meta: 5x menor que o fp16).
# Opcoes por tensor: fp16, 8 bits 2x2, 8/7/6 bits 4x4, 5 e 4 bits em cubos 4x8/4x4, 4 e 3 bits em cubos 4x16.
# 1) mede a sensibilidade REAL de cada tensor (replay das chamadas do UNet);  2) aloca bits por tensor (casco convexo + guloso)
# em varios orcamentos (20% = 5x ... 56%);  3) valida em imagens com sementes que a medicao nao viu.
# Tamanho = formula (bits + escala fp16 por cubo); se a fronteira for boa eu implemento o empacotamento de 3/5/6/7 bits no .ttq.
# ~40-50 min em CPU. MANTENHA A TELA LIGADA e a aba aberta (no celular, use 'Site para computador' e deixe a tela acesa). Rode numa celula. Me mande TODO o texto e tinysd_v2_comparacao.png.
import os, sys, time, subprocess, json, types, zlib, base64
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "scikit-image"])
import numpy as np, torch
''' + emb + r'''
al = types.ModuleType("alocacao_bits"); exec(zlib.decompress(base64.b64decode(_ALOC)).decode("utf-8"), al.__dict__)
from diffusers import AutoPipelineForText2Image
PROMPT, STEPS, SIZE = "Carro futurista correndo na chuva cyberpunk", 4, 256
FIT_SEEDS = (5762, 1)
SEEDS = (3, 4, 5)                       # sementes so para validar imagens
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

# ---- quantizacao simulada com QUALQUER numero de bits (simetrica, escala fp16 por cubo, absmax) ----
def sim(m, bloco, bits):
    bh, bw = bloco; R, C = m.shape
    Rp, Cp = -(-R // bh) * bh, -(-C // bw) * bw
    p = np.zeros((Rp, Cp), np.float32); p[:R, :C] = m
    b = p.reshape(Rp // bh, bh, Cp // bw, bw).transpose(0, 2, 1, 3)
    mx = np.abs(b).max(axis=(2, 3)); niv = 2 ** (bits - 1) - 1
    esc = np.where(mx == 0, 1.0, mx / niv).astype(np.float16).astype(np.float32)[:, :, None, None]
    q = np.clip(np.rint(b / esc), -niv - 1, niv)
    return (q * esc).transpose(0, 2, 1, 3).reshape(Rp, Cp)[:R, :C]

OPC = [("s", (4, 16), 3), ("s", (4, 16), 4), ("s", (4, 8), 4), ("s", (4, 4), 4), ("s", (4, 8), 5), ("s", (4, 4), 6),
       ("s", (4, 4), 7), ("s", (4, 4), 8), ("s", (2, 2), 8), ("h",)]
NOME = lambda c: "fp16" if c[0] == "h" else "%dx%d %db" % (c[1][0], c[1][1], c[2])
BPP = np.array([16.0 if c[0] == "h" else c[2] + 16.0 / (c[1][0] * c[1][1]) for c in OPC])
REF = 5                                   # opcao de referencia para MEDIR a sensibilidade: 4x4 6 bits (ruido bem acima do piso)

unet = pipe.unet
alvo = {n: p for n, p in unet.named_parameters() if p.ndim >= 2 and p.numel() >= 256}
orig = {n: p.data.clone() for n, p in alvo.items()}
names = list(alvo); N = len(names)
numel = np.array([alvo[n].numel() for n in names], np.int64)
print("0) gerando %d imagens de referencia (~1 min cada em CPU; sem mais nada na tela ate acabar) ..." % len(SEEDS), flush=True)
base = {}
for s_ in SEEDS:
    base[s_] = gerar(s_); print("   referencia semente %d pronta" % s_, flush=True)

def versao(n, op):
    m = orig[n].numpy().reshape(orig[n].shape[0], -1)
    return (m.astype(np.float16).astype(np.float32) if op[0] == "h" else sim(m, op[1], op[2])), m
def aplicar(n, op):
    w, _ = versao(n, op); alvo[n].data = torch.from_numpy(w.reshape(orig[n].shape)).float()
def restaurar():
    for n, p in alvo.items(): p.data = orig[n].clone()

# ---- chamadas reais do UNet (4 passos) para medir erro de SAIDA ----
trilha, dono = [], []
def _gancho(mod, args, kwargs, out):
    trilha.append((args[0].detach().clone(), args[1].detach().clone() if torch.is_tensor(args[1]) else torch.tensor(args[1]),
                   kwargs["encoder_hidden_states"].detach().clone(), out[0].detach().clone().numpy()))
print("0b) gravando as chamadas reais do UNet (%d imagens) ..." % len(FIT_SEEDS), flush=True)
for k_sd, sd in enumerate(FIT_SEEDS):
    h_ = unet.register_forward_hook(_gancho, with_kwargs=True)
    antes = len(trilha); gerar(sd); h_.remove(); dono += [k_sd] * (len(trilha) - antes)
    print("   semente %d gravada" % sd, flush=True)
POR_SEMENTE = len(trilha) // len(FIT_SEEDS)
en = [float((o ** 2).sum()) for *_, o in trilha]
def reproduzir(passos=None, sementes=None):
    erro = tot = 0.0
    with torch.no_grad():
        for k, (x, t, enc, o) in enumerate(trilha):
            if passos is not None and (k % POR_SEMENTE) not in passos: continue
            if sementes is not None and dono[k] not in sementes: continue
            y = unet(x, t, encoder_hidden_states=enc, return_dict=False)[0].numpy()
            erro += float(((y - o) ** 2).sum()); tot += en[k]
    return erro / tot
snr_db = lambda e: -10 * np.log10(max(e, 1e-30))
print("Gravadas %d chamadas reais do UNet. %d tensores, %.1f M pesos." % (len(trilha), N, numel.sum() / 1e6))

print("\n1) Ruido de cada opcao em cada tensor (so pesos, rapido) ...")
t0 = time.time()
ruido = np.zeros((N, len(OPC)))
for i, n in enumerate(names):
    for o, op in enumerate(OPC):
        w, m = versao(n, op); ruido[i, o] = float(((m - w) ** 2).sum())
    if (i + 1) % 25 == 0: print("   %d/%d" % (i + 1, N), flush=True)
print("   %.0f s" % (time.time() - t0))

print("2) Sensibilidade de cada tensor (%s, um por vez, replay real) ... ~15 min" % NOME(OPC[REF]))
CK = "v2_ckpt.json"                       # checkpoint: se o Colab cair e o ambiente continuar, ele RETOMA daqui
efeito = np.zeros(N); feito = 0
try:
    c = json.load(open(CK))
    if c["names"] == names and c["ref"] == REF: efeito[:] = c["efeito"]; feito = c["feito"]; print("   RETOMANDO do tensor %d/%d" % (feito, N))
except Exception:
    pass
for i in range(feito, N):
    n = names[i]
    aplicar(n, OPC[REF]); efeito[i] = reproduzir(passos=(1,), sementes=(0,)); alvo[n].data = orig[n].clone()   # 1 chamada (t medio)
    if (i + 1) % 10 == 0:
        json.dump({"names": names, "ref": REF, "efeito": efeito.tolist(), "feito": i + 1}, open(CK, "w"))
    if (i + 1) % 25 == 0: print("   %d/%d" % (i + 1, N), flush=True)
print("   mais sensiveis: " + ", ".join(names[i][-32:] for i in np.argsort(-efeito)[:5]))
err = (efeito[:, None] * ruido / np.maximum(ruido[:, [REF]], 1e-30))          # erro previsto de saida, tensor x opcao

estado = np.full(N, -1)
def carregar(g):
    for i in np.where(estado != g)[0]:
        aplicar(names[i], OPC[g[i]]); estado[i] = g[i]
def grade(g, ncols=None):
    carregar(g); return [gerar(s) for s in SEEDS]

print("\n3) CURVA: alocacao por tensor em varios tamanhos; validada em %d imagens novas" % len(SEEDS))
print("%-16s %8s %6s %9s %9s | %6s %7s %6s | tensores por opcao" % ("orcamento", "% fp16", "x fp16", "dB prev.", "dB real", "PSNR", "SSIM", "pior"))
ORCS = (0.20, 0.25, 0.30, 0.35, 0.45, 0.55)
saidas, res = {}, {}
for orc in ORCS:
    esc, usado, e_prev = al.alocar(BPP, err.tolist(), numel.tolist(), orc * 16.0 * numel.sum())
    g = np.asarray(esc, np.int64)
    ims = grade(g)
    ps = [psnr(base[s], im) for s, im in zip(SEEDS, ims)]; ss = [ssim(base[s], im) for s, im in zip(SEEDS, ims)]
    frac = usado / (16.0 * numel.sum())
    real = snr_db(reproduzir())
    cont = " ".join("%s:%d" % (NOME(OPC[o]), int((g == o).sum())) for o in range(len(OPC)) if (g == o).any())
    print("%-16s %7.1f%% %5.2fx %8.1f %9.1f | %5.1f %.4f %.4f | %s" % ("orcamento %d%%" % round(100 * orc), 100 * frac, 1 / frac,
          snr_db(e_prev), real, np.mean(ps), np.mean(ss), np.min(ss), cont), flush=True)
    saidas[orc] = ims
    res["%d%%" % round(100 * orc)] = {"tamanho_frac": frac, "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "unet_db": real,
                                      "receita": {names[i]: NOME(OPC[int(g[i])]) for i in range(N)}}
restaurar()
try:
    from PIL import Image
    W = SIZE; tela = Image.new("RGB", (W * (len(ORCS) + 1), W))
    tela.paste(base[SEEDS[0]], (0, 0))
    for k, orc in enumerate(ORCS): tela.paste(saidas[orc][0], (W * (k + 1), 0))
    tela.save("tinysd_v2_comparacao.png"); print("\nSalvo tinysd_v2_comparacao.png (original | " + " | ".join("%d%%" % round(100 * o) for o in ORCS) + ")")
except Exception as ex:
    print("(sem imagem: %s)" % ex)
json.dump({"opcoes": [NOME(c) for c in OPC], "bpp": BPP.tolist(), "efeito": {n: float(efeito[i]) for i, n in enumerate(names)}, "curva": res},
          open("tinysd_v2.json", "w"), indent=1)
try:
    from google.colab import files
    files.download("tinysd_v2.json"); files.download("tinysd_v2_comparacao.png")
except Exception:
    pass
print("Me mande TODO este texto, o tinysd_v2.json e a imagem.")
'''
open(os.path.join(aqui, "curva_v2_unico.py"), "w", encoding="utf-8").write(corpo)
