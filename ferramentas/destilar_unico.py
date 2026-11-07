# TTOKEN - 4 BITS NOS CUBOS COM "DESTILACAO" (QAT): o UNet de 4 bits aprende a imitar o UNet original.
# Por que: o GPTQ acertou cada camada sozinha mas o erro se acumulou (SSIM 0,66). Aqui TODAS as camadas sao
# ajustadas juntas, olhando a saida final do UNet (professor = UNet fp32). Treina: pesos latentes, escala de cada
# cubo (4x16, fp16) e, na 2a configuracao, um ramo de baixo posto em fp16 (estilo SVDQuant) que absorve os atipicos.
# Formato final continua compativel com o .ttq: 4 bits por peso + 1 escala fp16 por cubo (+ matrizes pequenas A,B).
# Precisa de GPU: Ambiente de execucao > Alterar tipo > T4 GPU. Tela ligada, aba aberta (celular: 'Site para computador').
# Roda um teste de fumaca de ~1 min antes; se algo quebrar, ele para cedo e mostra o erro.
# Mede em PROMPTS que o treino NAO viu (nada de "colar" no prompt de calibracao).
TEMPO_MIN = 18          # minutos de treino por configuracao (o passo-a-passo se adapta ao tempo)
CONFIGS = [("QAT 4b 4x16", 0), ("QAT 4b + baixo posto 16", 16)]   # (rotulo, posto do ramo fp16; 0 = sem)
AMP = True              # fp16 automatico no treino (mais rapido na T4); desliga sozinho se der NaN
import os, sys, time, json, copy, math, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate", "scikit-image"])
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
if not torch.cuda.is_available():
    raise SystemExit("Sem GPU. Ambiente de execucao > Alterar tipo de ambiente > T4 GPU, e rode de novo.")
dev = "cuda"
from diffusers import AutoPipelineForText2Image

STEPS_INF, SIZE, LOTE = 4, 256, 2
N_PROMPTS = 300          # v1 usou so 24 prompts: a perda de treino caiu 14x mas o teste quase nao mexeu (decorou). Agora ~300.
_SUJ = ["a red apple", "a cat", "a dog", "an old fisherman", "a robot", "a dragon", "a lighthouse", "a castle", "a spaceship",
        "a bicycle", "a bowl of ramen", "a cup of coffee", "a bouquet of flowers", "a sailing ship", "a horse", "an astronaut",
        "a small house", "a mountain", "a city street", "a car", "a bird", "a violin", "a lantern", "a bridge", "a windmill",
        "a fox", "a tree", "a train", "a market stall", "a girl with a hat", "a wolf", "a tractor", "a teapot", "a statue"]
_LUG = ["on a wooden table", "in a snowy forest", "at sunrise", "during a storm", "in a dark alley", "by the sea",
        "in a desert", "on mars", "in a library", "under the stars", "in a green valley", "at golden hour", "in the rain",
        "on a cliff", "in a field of flowers", "inside a cozy cabin", "above the clouds", "in an old town", "on a lake",
        "in a neon city"]
_EST = ["watercolor", "oil painting", "macro photo", "cyberpunk", "studio lighting", "pencil sketch", "cinematic",
        "pixel art", "long exposure", "soft light", "dramatic light", "top view", "fantasy art", "vintage photo"]
def _montar_prompts(n, semente):
    r = random.Random(semente); fora = set(); res = []
    while len(res) < n:
        p = "%s %s, %s" % (r.choice(_SUJ), r.choice(_LUG), r.choice(_EST))
        if p not in fora: fora.add(p); res.append(p)
    return res
PROMPTS_TREINO = _montar_prompts(N_PROMPTS, 123)
PROMPTS_VAL = [p for p in _montar_prompts(40, 999) if p not in PROMPTS_TREINO][:8]   # perda de validacao durante o treino
PROMPTS_TESTE = ["a futuristic city street with flying cars at dusk", "a wooden boat on a calm lake in the morning",
                 "a bowl of fresh strawberries on a white table", "a lion resting under a tree, savannah"]
SEM_TREINO, SEM_TESTE = (1, 2), (7, 8)

pipe = AutoPipelineForText2Image.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32).to(dev)
pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"): pipe.safety_checker = None
teacher = pipe.unet.eval()
teacher.requires_grad_(False)
gerar = lambda p, s: pipe(prompt=p, num_inference_steps=STEPS_INF, guidance_scale=7.5,
                          generator=torch.Generator("cpu").manual_seed(s), height=SIZE, width=SIZE).images[0]


def psnr(a, b):
    x, y = np.asarray(a.convert("RGB"), np.float64), np.asarray(b.convert("RGB"), np.float64)
    m = np.mean((x - y) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)


def ssim(a, b):
    from skimage.metrics import structural_similarity as S
    return float(S(np.asarray(a.convert("L")), np.asarray(b.convert("L")), data_range=255))


# ------------------------------------------------------------------ dados: chamadas reais do UNet (professor)
print("0) referencias e chamadas reais do UNet ...", flush=True)
DADOS, VAL = [], []
_ALVO = [DADOS]
def _gancho(mod, args, kwargs, out):
    x = args[0].detach().cpu()
    t = torch.as_tensor(args[1]).detach().cpu().reshape(-1).expand(x.shape[0]).clone()
    _ALVO[0].append((x, t, kwargs["encoder_hidden_states"].detach().cpu().half()))   # [incondicional, condicional]
t0_ = time.time()
with torch.no_grad():
    h_ = teacher.register_forward_hook(_gancho, with_kwargs=True)
    for i_, p in enumerate(PROMPTS_TREINO):
        for s in SEM_TREINO: gerar(p, s)
        if (i_ + 1) % 50 == 0: print("   %d/%d prompts (%.0f s)" % (i_ + 1, len(PROMPTS_TREINO), time.time() - t0_), flush=True)
    _ALVO[0] = VAL
    for p in PROMPTS_VAL: gerar(p, 5)
    h_.remove()
    BASE = {(p, s): gerar(p, s) for p in PROMPTS_TESTE for s in SEM_TESTE}
print("   %d chamadas de treino, %d de validacao, %d imagens de referencia (prompts de teste/validacao NUNCA vistos no treino)"
      % (len(DADOS), len(VAL), len(BASE)), flush=True)


# ------------------------------------------------------------------ camada em cubos (fake-quant com STE)
class Cubo(nn.Module):
    """Peso W (R x C) em cubos bh x bw. W_efetivo = quant(W - A@B) + A@B. Escala por cubo = exp(logs), arredondada p/ fp16."""
    def __init__(self, W4d, bits, bloco, rank):
        super().__init__()
        self.forma = tuple(W4d.shape); R = self.forma[0]; W = W4d.detach().reshape(R, -1).float()
        self.R, self.C = W.shape; self.bits = bits; self.bh, self.bw = bloco; self.niv = 2 ** (bits - 1) - 1
        self.nbh, self.nbw = -(-self.R // self.bh), -(-self.C // self.bw)
        self.duro = False
        self.W = nn.Parameter(W.clone())
        r = min(rank, self.R, self.C)
        if r > 0:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.A = nn.Parameter(U[:, :r] * S[:r].sqrt()[None, :]); self.B = nn.Parameter(S[:r].sqrt()[:, None] * Vh[:r])
        else:
            self.A = self.B = None
        with torch.no_grad():
            res = W - (self.A @ self.B if r > 0 else 0.0)
            mx = self._cubos(res).abs().amax(dim=(1, 3))
            s = torch.where(mx == 0, torch.ones_like(mx), mx / self.niv)
        self.logs = nn.Parameter(s.half().float().log())

    def _cubos(self, x):
        x = F.pad(x, (0, self.nbw * self.bw - self.C, 0, self.nbh * self.bh - self.R))
        return x.view(self.nbh, self.bh, self.nbw, self.bw)

    def _lr(self):
        if self.A is None: return None
        if self.duro: return self.A.half().float() @ self.B.half().float()
        return self.A @ self.B

    def _q(self, res, s):
        u = self._cubos(res) / s[:, None, :, None]
        if self.duro: q = u.round()
        else: q = u + (u.round() - u).detach()                 # STE: gradiente passa pelo arredondamento
        return q.clamp(-self.niv - 1, self.niv)

    def _escala(self):
        s = self.logs.exp()
        return s.half().float() if self.duro else s + (s.half().float() - s).detach()

    def _efetivo(self):
        lr = self._lr(); res = self.W if lr is None else self.W - lr
        s = self._escala()
        y = (self._q(res, s) * s[:, None, :, None]).reshape(self.nbh * self.bh, self.nbw * self.bw)[:self.R, :self.C]
        if lr is not None: y = y + lr
        return y.reshape(self.forma)

    def forward(self):
        if self.duro or not torch.is_grad_enabled(): return self._efetivo()
        return checkpoint(self._efetivo, use_reentrant=False)  # nao guarda os intermediarios (poupa ~6 GB)

    def exportar(self):
        with torch.no_grad():
            self.duro = True
            lr = self._lr(); res = self.W if lr is None else self.W - lr
            s = self._escala()
            q = self._q(res, s).to(torch.int8).reshape(self.nbh * self.bh, self.nbw * self.bw)[:self.R, :self.C]
            self.duro = False
        d = {"q": q.cpu().numpy(), "esc": s.half().cpu().numpy()}
        if self.A is not None: d["A"] = self.A.detach().half().cpu().numpy(); d["B"] = self.B.detach().half().cpu().numpy()
        return d

    def bits_total(self):
        b = self.R * self.C * (self.bits + 16.0 / (self.bh * self.bw))
        if self.A is not None: b += (self.A.numel() + self.B.numel()) * 16.0
        return b


def instalar(unet, rank):
    """Troca o peso de cada Linear/Conv por um Cubo. conv_in/conv_out ficam em 8 bits 4x4 (minusculas e sensiveis)."""
    cubos = nn.ModuleDict(); alvo = []
    for n, m in unet.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and (not isinstance(m, nn.Conv2d) or m.groups == 1) \
                and m.weight.ndim >= 2 and m.weight.numel() >= 256:
            alvo.append((n, m))
    for n, m in alvo:
        sempre8 = n.startswith("conv_in") or n.startswith("conv_out")
        c = Cubo(m.weight.data, 8 if sempre8 else 4, (4, 4) if sempre8 else (4, 16), 0 if sempre8 else rank).to(dev)
        cubos[n.replace(".", "__")] = c
        del m._parameters["weight"]
        m.register_forward_pre_hook(lambda mod, args, c=c: setattr(mod, "weight", c()))
    for p in unet.parameters(): p.requires_grad_(True)             # biases, normas, embeddings: tambem se ajustam
    return cubos


def dados_lote(fonte=None, idx=None):
    fonte = DADOS if fonte is None else fonte
    idx = random.sample(range(len(fonte)), LOTE) if idx is None else idx
    x = torch.cat([fonte[i][0] for i in idx]).to(dev); t = torch.cat([fonte[i][1] for i in idx]).to(dev)
    e = torch.cat([fonte[i][2] for i in idx]).to(dev).float()
    return x, t, e


GUIA = 7.5
def combinado(eps):
    """Saida que de fato vira imagem: eps_u + guia*(eps_c - eps_u). Erros na diferenca sao amplificados 7,5x."""
    r = eps.view(-1, 2, *eps.shape[1:]); return r[:, 0] + GUIA * (r[:, 1] - r[:, 0])


def perda(sai, alvo_):
    return F.mse_loss(sai, alvo_) + F.mse_loss(combinado(sai), combinado(alvo_))


def perda_val(student, amp):
    student.eval(); tot = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for i in range(0, len(VAL) - LOTE + 1, LOTE):
            x, t, e = dados_lote(VAL, list(range(i, i + LOTE)))
            a = teacher(x, t, encoder_hidden_states=e, return_dict=False)[0].float()
            b = student(x, t, encoder_hidden_states=e, return_dict=False)[0].float()
            tot.append(perda(b, a).item())
    student.train(); return float(np.mean(tot))


def treinar(student, cubos, segundos, amp):
    gW = [p for n, p in cubos.named_parameters() if n.endswith(".W")]
    gS = [p for n, p in cubos.named_parameters() if n.endswith(".logs")]
    gL = [p for n, p in cubos.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    gO = [p for p in student.parameters()]
    grupos = [{"params": gW, "lr": 3e-5}, {"params": gS, "lr": 2e-3}, {"params": gO, "lr": 1e-5}]
    if gL: grupos.append({"params": gL, "lr": 3e-5})
    base = [g["lr"] for g in grupos]
    opt = torch.optim.Adam(grupos)
    try: scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except Exception: scaler = torch.cuda.amp.GradScaler(enabled=amp)
    student.train(); t0 = time.time(); k = 0; ema = None
    melhor, estado = None, None
    v0 = perda_val(student, amp); print("      validacao inicial %.5f" % v0, flush=True)
    while True:
        frac = (time.time() - t0) / segundos
        if frac >= 1.0: break
        f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac))
        for g, b in zip(opt.param_groups, base): g["lr"] = b * f
        x, t, e = dados_lote()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            alvo_ = teacher(x, t, encoder_hidden_states=e, return_dict=False)[0].float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            sai = student(x, t, encoder_hidden_states=e, return_dict=False)[0].float()
        loss = perda(sai, alvo_)
        if not torch.isfinite(loss): return None
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        k += 1; ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if k % 150 == 0:
            v = perda_val(student, amp)
            marca = ""
            if melhor is None or v < melhor:
                melhor = v; estado = None; torch.cuda.empty_cache()
                estado = ({n: p.detach().clone() for n, p in cubos.state_dict().items()},
                          {n: p.detach().clone() for n, p in student.state_dict().items()}); marca = "  <- melhor"
            print("      passo %4d  treino %.5f  VALIDACAO %.5f%s  (%.1f s/passo, %d%% do tempo)"
                  % (k, ema, v, marca, (time.time() - t0) / k, 100 * frac), flush=True)
    if estado is not None:
        cubos.load_state_dict(estado[0]); student.load_state_dict(estado[1])
        print("      restaurado o melhor ponto (validacao %.5f; inicial %.5f)" % (melhor, v0), flush=True)
    return ema if ema is not None else 0.0


def avaliar(student, cubos, rotulo):
    for c in cubos.values(): c.duro = True
    student.eval(); pipe.unet = student
    ps, ss, ims = [], [], {}
    with torch.no_grad():
        for (p, s), ref in BASE.items():
            im = gerar(p, s); ims[(p, s)] = im
            ps.append(psnr(ref, im)); ss.append(ssim(ref, im))
    pipe.unet = teacher
    for c in cubos.values(): c.duro = False
    bits = sum(c.bits_total() for c in cubos.values()); total = sum(c.R * c.C for c in cubos.values())
    fr = bits / (16.0 * total)
    print("   %-26s %5.1f%% %5.2fx | PSNR %5.1f  SSIM %.4f  pior %.4f" % (rotulo, 100 * fr, 1 / fr, np.mean(ps), np.mean(ss), np.min(ss)), flush=True)
    return {"tamanho_frac": fr, "x_fp16": 1 / fr, "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "ssim_pior": float(np.min(ss))}, ims


def nova(rank):
    s = copy.deepcopy(teacher); c = instalar(s, rank); return s, c


# ------------------------------------------------------------------ 1) fumaca: o laco de treino funciona?
print("\n1) teste de fumaca (~1 min) ...", flush=True)
for tentativa in (AMP, False):
    st, cb = nova(8)
    r = treinar(st, cb, 20, tentativa)
    del st, cb; torch.cuda.empty_cache()
    if r is not None:
        AMP = tentativa; print("   ok (AMP=%s, perda %.5f)" % (AMP, r), flush=True); break
    print("   perda nao finita com AMP=%s; tentando sem AMP" % tentativa, flush=True)
else:
    raise SystemExit("o treino deu NaN mesmo sem AMP; me mande este texto.")

# ------------------------------------------------------------------ 2) referencia: 4 bits SEM treino (confere com o RTN ja medido)
print("\n2) MEDIDAS (SSIM/PSNR contra o UNet original, em prompts nunca vistos no treino)", flush=True)
res, grade = {}, {}
st, cb = nova(0)
res["RTN 4b 4x16 (sem treino)"], grade["RTN 4b 4x16 (sem treino)"] = avaliar(st, cb, "RTN 4b 4x16 (sem treino)")
del st, cb; torch.cuda.empty_cache()

# ------------------------------------------------------------------ 3) treino por configuracao
for rot, rank in CONFIGS:
    print("\n3) treinando '%s' por %d min ..." % (rot, TEMPO_MIN), flush=True)
    st, cb = nova(rank)
    r = treinar(st, cb, TEMPO_MIN * 60, AMP)
    if r is None:
        print("   perda NaN; pulando esta configuracao", flush=True); del st, cb; torch.cuda.empty_cache(); continue
    res[rot], grade[rot] = avaliar(st, cb, rot)
    estado = {}
    for n, c in cb.items():
        for k_, v in c.exportar().items(): estado["%s::%s" % (n, k_)] = v
    for n, p in st.named_parameters(): estado["resto::" + n] = p.detach().half().cpu().numpy()
    arq = "tinysd_qat_%s.npz" % ("lr%d" % rank if rank else "q4")
    np.savez_compressed(arq, **estado); print("   pesos salvos em %s (%.0f MB)" % (arq, os.path.getsize(arq) / 1e6), flush=True)
    json.dump(res, open("tinysd_qat.json", "w"), indent=1)
    del st, cb, estado; torch.cuda.empty_cache()

# ------------------------------------------------------------------ 4) quadro final
print("\n4) RESUMO (comparacao: GPTQ 4b da rodada anterior = 27,5% / SSIM 0,66)")
print("%-28s %7s %7s | %6s %7s %7s" % ("configuracao", "% fp16", "x fp16", "PSNR", "SSIM", "pior"))
for r_, v in res.items():
    print("%-28s %6.1f%% %6.2fx | %6.1f %7.4f %7.4f" % (r_, 100 * v["tamanho_frac"], v["x_fp16"], v["psnr"], v["ssim"], v["ssim_pior"]))
try:
    from PIL import Image
    ch = (PROMPTS_TESTE[0], SEM_TESTE[0]); rots = list(grade)
    tela = Image.new("RGB", (SIZE * (len(rots) + 1), SIZE)); tela.paste(BASE[ch], (0, 0))
    for k_, r_ in enumerate(rots): tela.paste(grade[r_][ch], (SIZE * (k_ + 1), 0))
    tela.save("tinysd_qat_comparacao.png"); print("Imagem: original | " + " | ".join(rots))
except Exception as ex:
    print("(sem imagem: %s)" % ex)
json.dump(res, open("tinysd_qat.json", "w"), indent=1)
try:
    from google.colab import files
    files.download("tinysd_qat.json"); files.download("tinysd_qat_comparacao.png")
except Exception:
    pass
print("Me mande TODO este texto, o tinysd_qat.json e a imagem. Os pesos ficaram em tinysd_qat_*.npz (para o exportador .ttq).")
