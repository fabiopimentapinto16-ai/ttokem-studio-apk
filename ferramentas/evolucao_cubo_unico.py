# TTOKEN - EVOLUCAO: uma populacao de 'receitas' de quantizacao; quem estoura o tamanho ou fica longe da meta MORRE,
# e os melhores geram filhos com mutacoes (genes = precisao de cada tensor: 8x8, 4x4, 2x2 ou fp16). tiny-sd REAL.
# Fitness = erro na SAIDA do UNet reproduzindo as 4 chamadas REAIS da amostragem (nao nos pesos). Valida em imagens (4 sementes).
# Gera tinysd_unet_final.ttq (receita campea ja gravada) + tinysd_genoma.json. ~35-40 min (CPU).
# NAO FOI EXECUTADO AINDA (o sandbox onde escrevi nao tem torch): rode (~20-30 min) e me mande as tabelas.
# Roda os MESMOS dados do seu teste (seed 5762, 256x256, 4 passos) e compara configuracoes que o
# codec de cores nao testou: cubos pequenos, 4 bits com cubos finos, reordenar, e TODOS os
# pesos (convolucoes tambem), nao so as Linear. Arquivo UNICO: ja traz o cubo_quant dentro. Cole tudo numa celula.
# NAO FOI EXECUTADO AINDA (o sandbox onde escrevi nao tem torch): rode e me mande a tabela.
import os, sys, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers", "transformers", "accelerate"])
import numpy as np, torch, types
# --- cubo_quant.py embutido (nao precisa subir mais nenhum arquivo) ---
_CODIGO = r'''"""CUBO-QUANT: quantizacao de pesos em CUBOS (blocos) de 4 bits + leitor em streaming.

Sem torch: so numpy. Serve para o Colab (comprimir) e para o Pydroid 3 (ler).

O que era errado no codigo antigo (erro 'cannot reshape array of size 12800 into shape
(320,4,3,3)'): para quantizar, a matriz era COMPLETADA com zeros ate ser multiplo de 8
(36 colunas -> 40; 320*40 = 12800), mas na hora de ler ninguem cortava esse enchimento antes de
voltar a forma original (320*4*3*3 = 11520). Aqui a forma original e o tamanho real ficam
gravados no arquivo e o enchimento e sempre cortado.

Outra correcao honesta: o "Q4" antigo guardava cada valor de 4 bits num int8 (1 byte), entao so
chegava a 1/4 do FP32. Aqui dois valores de 4 bits sao empacotados em 1 byte (1/8 do FP32).

Formato .ttq:  b"TTQ1" | uint64 tamanho do cabecalho | cabecalho JSON | blobs (alinhados em 8).
Cada tensor 2D+ vira matriz (linhas x resto); a matriz e cortada em cubos bh x bw; cada cubo tem
UMA escala (float16) = max|cubo| / 7; os valores vao de -8 a 7 (guardados como 0..15).
"""
import json
import mmap
import os
import struct
import sys

import numpy as np

MAGICO = b"TTQ1"
MIN_ELEMENTOS = 256          # tensores menores (bias, norm) ficam em float16, sem perda


def _alinha(n, a=8):
    return (n + a - 1) // a * a


def quantizar(arr, bloco=(8, 8), bits=4):
    """Devolve (empacotado uint8, escalas float16, linhas, colunas) de um tensor >= 2D.
    bits=4: 2 valores por byte (-8..7). bits=8: 1 valor por byte (-128..127), mais qualidade."""
    bh, bw = bloco
    arr = np.asarray(arr, dtype=np.float32)
    m = arr.reshape(arr.shape[0], -1)
    R, C = m.shape
    Rp, Cp = _alinha(R, bh), _alinha(C, bw)
    p = np.zeros((Rp, Cp), dtype=np.float32)
    p[:R, :C] = m
    b = p.reshape(Rp // bh, bh, Cp // bw, bw).transpose(0, 2, 1, 3)       # (nbh, nbw, bh, bw)
    maximo = np.abs(b).max(axis=(2, 3))
    nivel = 7.0 if bits == 4 else 127.0
    esc = np.where(maximo == 0, 1.0, maximo / nivel).astype(np.float16)
    esc32 = esc.astype(np.float32)                    # usa a escala JA arredondada (consistente)
    q = np.rint(b / esc32[:, :, None, None])
    if bits == 4:
        plano = (np.clip(q, -8, 7).astype(np.int8) + 8).astype(np.uint8).reshape(-1)
        if plano.size % 2:
            plano = np.concatenate([plano, np.zeros(1, np.uint8)])
        return ((plano[0::2] << 4) | plano[1::2]).astype(np.uint8), esc, R, C
    return (np.clip(q, -128, 127).astype(np.int16) + 128).astype(np.uint8).reshape(-1), esc, R, C


def desquantizar(pack, esc, R, C, forma, bloco=(8, 8), dtype=np.float32, bits=4,
                 max_elementos=1 << 16):
    """Reconstroi POR PARTES (algumas linhas de cubos por vez): os temporarios ficam pequenos,
    so o resultado final ocupa a RAM do tensor. Corta o enchimento e volta a forma original."""
    bh, bw = bloco
    nbh, nbw = _alinha(R, bh) // bh, _alinha(C, bw) // bw
    esc = esc.astype(np.float32).reshape(nbh, nbw)
    por_linha = nbw * bh * bw                           # elementos de uma linha de cubos
    passo = max(1, max_elementos // por_linha)
    if bits == 4 and por_linha % 2:                     # cubo impar: nao da para partir no meio do byte
        passo = nbh
    out = np.empty((R, C), dtype=np.float32)
    for a in range(0, nbh, passo):
        z = min(nbh, a + passo)
        i0, i1 = a * por_linha, z * por_linha
        if bits == 4:
            bytes_ = pack[i0 // 2:(i1 + 1) // 2]
            plano = np.empty(bytes_.size * 2, dtype=np.uint8)
            plano[0::2] = bytes_ >> 4
            plano[1::2] = bytes_ & 15
            q = plano[:i1 - i0].astype(np.float32) - 8.0
        else:
            q = pack[i0:i1].astype(np.float32) - 128.0
        blocos = q.reshape(z - a, nbw, bh, bw) * esc[a:z, :, None, None]
        faixa = blocos.transpose(0, 2, 1, 3).reshape((z - a) * bh, nbw * bw)
        r0, r1 = a * bh, min(R, z * bh)
        out[r0:r1, :] = faixa[:r1 - r0, :C]
    return out.reshape(forma).astype(dtype, copy=False)


def ordem_cubo(m):
    """Permutacao de linhas e colunas que junta valores de tamanho parecido (como arrumar as
    pecas do cubo magico por cor): ordena por energia. Cubos vizinhos ficam com faixa de valores
    parecida, a escala de cada cubo fica justa e o MESMO numero de bits erra menos."""
    pr = np.argsort(np.abs(m).max(axis=1))
    pc = np.argsort(np.abs(m).max(axis=0))
    return pr, pc


def erro(original, recuperado):
    """Metricas honestas: SNR em dB, erro maximo e erro medio relativo."""
    o = np.asarray(original, np.float64).ravel()
    r = np.asarray(recuperado, np.float64).ravel()
    ruido = np.sum((o - r) ** 2)
    sinal = np.sum(o ** 2)
    snr = float("inf") if ruido == 0 else 10 * np.log10(max(sinal, 1e-30) / ruido)
    return {"snr_db": round(float(snr), 2), "erro_max": float(np.abs(o - r).max()),
            "erro_medio_rel": float(np.abs(o - r).mean() / (np.abs(o).mean() + 1e-12))}


# ------------------------------------------------------------------ gravar
def salvar(estado, caminho, bloco=(8, 8), bits=4, reordenar=False, receita=None):
    """estado: dict nome -> ndarray. Grava o .ttq (atomico). Devolve estatisticas.
    receita (opcional): dict nome -> "fp16" | ((bh, bw), bits). Permite um cubo/precisao DIFERENTE por tensor
    (ex.: camadas sensiveis em fp16 ou 2x2, o resto em 4x4). Quem nao aparece usa (bloco, bits)."""
    receita = receita or {}
    meta, blobs, pos = {}, [], 0
    bruto = 0
    for nome, arr in estado.items():
        arr = np.asarray(arr)
        bruto += arr.size * 4
        rec = receita.get(nome)
        if rec == "fp16":
            rec_bloco, rec_bits = None, None
        else:
            rec_bloco, rec_bits = (tuple(rec[0]), int(rec[1])) if rec else (tuple(bloco), bits)
        if arr.ndim >= 2 and arr.size >= MIN_ELEMENTOS and rec != "fp16":
            m = arr.astype(np.float32).reshape(arr.shape[0], -1)
            pr = pc = None
            if reordenar:                                  # "resolve o cubo": agrupa parecidos
                pr, pc = ordem_cubo(m)
                m = m[pr][:, pc]
            pack, esc, R, C = quantizar(m, rec_bloco, rec_bits)
            dados = [pack.tobytes(), esc.tobytes()]
            meta[nome] = {"modo": "q%d" % rec_bits, "forma": list(arr.shape), "R": R, "C": C,
                          "off": pos, "n": len(dados[0]),
                          "off_esc": pos + _alinha(len(dados[0])), "n_esc": len(dados[1])}
            if rec:                                          # so grava por tensor quando a receita pediu
                meta[nome]["bloco"], meta[nome]["bits"] = list(rec_bloco), rec_bits
            tamanho = _alinha(len(dados[0])) + _alinha(len(dados[1]))
            blobs.append((pos, dados[0])); blobs.append((meta[nome]["off_esc"], dados[1]))
            if pr is not None:
                dr, dc = pr.astype(np.uint32).tobytes(), pc.astype(np.uint32).tobytes()
                meta[nome]["off_pr"], meta[nome]["off_pc"] = pos + tamanho, pos + tamanho + _alinha(len(dr))
                blobs.append((meta[nome]["off_pr"], dr)); blobs.append((meta[nome]["off_pc"], dc))
                tamanho += _alinha(len(dr)) + _alinha(len(dc))
        else:
            d = np.ascontiguousarray(arr.astype(np.float16)).tobytes()
            meta[nome] = {"modo": "f16", "forma": list(arr.shape), "off": pos, "n": len(d)}
            blobs.append((pos, d))
            tamanho = _alinha(len(d))
        pos += tamanho
    cab = json.dumps({"bloco": list(bloco), "bits": bits, "tensores": meta}).encode("utf-8")
    ini = _alinha(4 + 8 + len(cab))
    tmp = caminho + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGICO + struct.pack("<Q", len(cab)) + cab)
        f.write(b"\0" * (ini - 12 - len(cab)))
        for off, d in blobs:
            f.seek(ini + off)
            f.write(d)
        f.truncate(ini + pos)
    os.replace(tmp, caminho)
    final = os.path.getsize(caminho)
    return {"bruto_fp32_mb": bruto / 2**20, "arquivo_mb": final / 2**20,
            "economia_pct": round(100 * (1 - final / max(bruto, 1)), 1)}


# ------------------------------------------------------------------ ler (streaming)
class Leitor:
    """Le UM tensor por vez direto do arquivo (mmap): a RAM so guarda o tensor em uso."""

    def __init__(self, caminho):
        self._f = open(caminho, "rb")
        if self._f.read(4) != MAGICO:
            self._f.close()
            raise ValueError("arquivo nao e um .ttq valido")
        (n,) = struct.unpack("<Q", self._f.read(8))
        cab = json.loads(self._f.read(n).decode("utf-8"))
        self.bloco = tuple(cab["bloco"])
        self.bits = int(cab.get("bits", 4))
        self.meta = cab["tensores"]
        self._ini = _alinha(12 + n)
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def nomes(self):
        return list(self.meta)

    def ler(self, nome, dtype=np.float32):
        t = self.meta[nome]
        forma = tuple(t["forma"])
        base = self._ini
        if t["modo"] == "f16":
            d = np.frombuffer(self._mm, np.float16, count=int(np.prod(forma, dtype=np.int64)),
                              offset=base + t["off"])
            return d.reshape(forma).astype(dtype)
        pack = np.frombuffer(self._mm, np.uint8, count=t["n"], offset=base + t["off"])
        esc = np.frombuffer(self._mm, np.float16, count=t["n_esc"] // 2,
                            offset=base + t["off_esc"])
        bloco_t = tuple(t.get("bloco", self.bloco))
        bits_t = int(t.get("bits", self.bits))
        bh, bw = bloco_t
        esc = esc.reshape(_alinha(t["R"], bh) // bh, _alinha(t["C"], bw) // bw)
        if "off_pr" in t:                                    # desfaz a reordenacao do cubo
            m = desquantizar(pack, esc, t["R"], t["C"], (t["R"], t["C"]), bloco_t, np.float32, bits_t)
            pr = np.frombuffer(self._mm, np.uint32, count=t["R"], offset=base + t["off_pr"])
            pc = np.frombuffer(self._mm, np.uint32, count=t["C"], offset=base + t["off_pc"])
            ir, ic = np.argsort(pr), np.argsort(pc)
            return m[ir][:, ic].reshape(forma).astype(dtype, copy=False)
        return desquantizar(pack, esc, t["R"], t["C"], forma, bloco_t, dtype, bits_t)

    def itens(self):
        for nome in self.meta:
            yield nome, self.ler(nome)

    def fechar(self):
        try:
            self._mm.close()
        finally:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.fechar()


def para_torch(caminho, dispositivo="cpu", dtype_nome="float16"):
    """Colab: monta um state_dict do PyTorch lendo o .ttq (importa torch so aqui)."""
    import torch
    dt = getattr(torch, dtype_nome)
    sd = {}
    with Leitor(caminho) as lt:
        for nome, arr in lt.itens():
            sd[nome] = torch.from_numpy(arr).to(dispositivo).to(dt)
    return sd


def de_safetensors(entrada, saida, bloco=(8, 8), bits=4):
    """Converte um .safetensors (so float32/float16) em .ttq. Precisa de `safetensors`."""
    from safetensors.numpy import load_file
    return salvar(load_file(entrada), saida, bloco, bits)


if __name__ == "__main__":      # python cubo_quant.py modelo.safetensors saida.ttq
    if len(sys.argv) != 3:
        print("uso: python cubo_quant.py entrada.safetensors saida.ttq")
        sys.exit(1)
    print(de_safetensors(sys.argv[1], sys.argv[2]))
'''
cq = types.ModuleType("cubo_quant"); exec(_CODIGO, cq.__dict__); sys.modules["cubo_quant"] = cq

_CODIGO_K = '"""CUBO-QUANT K: quantizacao em blocos ASSIMETRICA (minimo + escala) com busca da melhor faixa.\n\nPor que existe: o cubo_quant.py usa so uma escala por cubo (simetrico, max/7). No tiny-sd real isso\nquebrou o 4 bits (SSIM <= 0.42). Aqui cada bloco guarda `minimo` e `escala` (float16) e a faixa e\nENCOLHIDA quando isso reduz o erro quadratico do bloco (ignora um ou outro valor extremo), como o\nQ4_K/AWQ fazem. Blocos finos (1 x 32, ao longo da entrada) seguem o estilo do GGUF.\n\nSo numpy. Custo em bits/peso (4 bits): bloco de 32 valores = 5.0 ; de 64 = 4.5 (escala+minimo em float16).\nSTATUS: testado em dados sinteticos; o ganho no modelo real so se prova com teste_k_unico (Colab).\n"""\nimport numpy as np\n\nFATORES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65)     # quanto da faixa original manter\n\n\ndef _alinha(n, a):\n    return (n + a - 1) // a * a\n\n\ndef _em_blocos(m, bloco):\n    bh, bw = bloco\n    R, C = m.shape\n    Rp, Cp = _alinha(R, bh), _alinha(C, bw)\n    p = np.pad(m, ((0, Rp - R), (0, Cp - C)), mode="edge") if (Rp, Cp) != (R, C) else m\n    b = p.reshape(Rp // bh, bh, Cp // bw, bw).transpose(0, 2, 1, 3).reshape(-1, bh * bw)\n    return b, Rp // bh, Cp // bw\n\n\ndef quantizar_k(arr, bloco=(1, 32), bits=4, busca=True, chunk=1 << 20):\n    """Devolve (pack uint8, escala f16, minimo f16, R, C). arr: qualquer forma (linhas x resto)."""\n    m = np.asarray(arr, np.float32)\n    m = m.reshape(m.shape[0], -1)\n    R, C = m.shape\n    b, nbh, nbw = _em_blocos(m, bloco)\n    nb, n = b.shape\n    L = float(2 ** bits - 1)\n    fat = FATORES if busca else (1.0,)\n    q_all = np.empty((nb, n), np.uint8)\n    esc_all = np.empty(nb, np.float16)\n    mn_all = np.empty(nb, np.float16)\n    por = max(1, chunk // n)\n    for a in range(0, nb, por):\n        x = b[a:a + por]\n        lo, hi = x.min(1), x.max(1)\n        mid, meia = (hi + lo) / 2, (hi - lo) / 2\n        melhor = np.full(len(x), np.inf, np.float32)\n        for f in fat:\n            l = (mid - meia * f).astype(np.float16)\n            sc = np.where(meia * f > 0, 2 * meia * f / L, 1.0).astype(np.float16)\n            l32, s32 = l.astype(np.float32), sc.astype(np.float32)\n            s32 = np.where(s32 == 0, 1.0, s32)\n            q = np.clip(np.rint((x - l32[:, None]) / s32[:, None]), 0, L)\n            e = ((q * s32[:, None] + l32[:, None] - x) ** 2).sum(1)\n            k = e < melhor\n            if k.any():\n                melhor[k] = e[k]\n                q_all[a:a + por][k] = q[k].astype(np.uint8)\n                esc_all[a:a + por][k] = sc[k]\n                mn_all[a:a + por][k] = l[k]\n    plano = q_all.reshape(-1)\n    if bits == 4:\n        if plano.size % 2:\n            plano = np.concatenate([plano, np.zeros(1, np.uint8)])\n        pack = ((plano[0::2] << 4) | plano[1::2]).astype(np.uint8)\n    else:\n        pack = plano\n    return pack, esc_all, mn_all, R, C\n\n\ndef desquantizar_k(pack, esc, mn, R, C, forma, bloco=(1, 32), bits=4, dtype=np.float32):\n    bh, bw = bloco\n    nbh, nbw = _alinha(R, bh) // bh, _alinha(C, bw) // bw\n    nb, n = nbh * nbw, bh * bw\n    if bits == 4:\n        plano = np.empty(pack.size * 2, np.uint8)\n        plano[0::2], plano[1::2] = pack >> 4, pack & 15\n        q = plano[:nb * n]\n    else:\n        q = pack[:nb * n]\n    s = esc.astype(np.float32)\n    s = np.where(s == 0, 1.0, s)\n    x = q.reshape(nb, n).astype(np.float32) * s[:, None] + mn.astype(np.float32)[:, None]\n    x = x.reshape(nbh, nbw, bh, bw).transpose(0, 2, 1, 3).reshape(nbh * bh, nbw * bw)\n    return x[:R, :C].reshape(forma).astype(dtype, copy=False)\n\n\ndef bytes_k(pack, esc, mn):\n    return pack.nbytes + esc.nbytes + mn.nbytes\n\n\ndef bits_por_peso(bloco, bits):\n    return bits + 32.0 / (bloco[0] * bloco[1])\n'
ck = types.ModuleType("cubo_quant_k"); exec(_CODIGO_K, ck.__dict__); sys.modules["cubo_quant_k"] = ck

from diffusers import AutoPipelineForText2Image
PROMPT, STEPS, SIZE = "Carro futurista correndo na chuva cyberpunk", 4, 256
SEEDS = (5762, 1, 2, 3)
SSIM_ALTO, SSIM_BAIXO = 0.93, 0.80
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

def grupo(n):
    if "time_embedding" in n: return "tempo (embedding)"
    if n.startswith("conv_in") or n.startswith("conv_out"): return "conv entrada/saida"
    if ".attn1." in n or ".attn2." in n: return "atencao (q,k,v,saida)"
    if ".ff." in n: return "feed-forward"
    if "proj_in" in n or "proj_out" in n: return "proj_in/proj_out"
    if "time_emb_proj" in n: return "resnet: tempo"
    if "resnets" in n: return "resnet: convolucoes"
    if "downsamplers" in n or "upsamplers" in n: return "amostragem"
    return "outros"

unet = pipe.unet
alvo = {n: p for n, p in unet.named_parameters() if p.ndim >= 2 and p.numel() >= cq.MIN_ELEMENTOS}
orig = {n: p.data.clone() for n, p in alvo.items()}
grupos = {}
for n in alvo: grupos.setdefault(grupo(n), []).append(n)
fp16_total = sum(p.numel() * 2 for p in alvo.values())
base = {s: gerar(s) for s in SEEDS}

def aplicar(n, cfg):
    m = orig[n].numpy().reshape(orig[n].shape[0], -1)
    if cfg[0] == "h":
        w = m.astype(np.float16).astype(np.float32); nb = m.size * 2
    elif cfg[0] == "s":
        pk, es, R, C = cq.quantizar(m, cfg[1], cfg[2])
        w = cq.desquantizar(pk, es, R, C, (R, C), cfg[1], bits=cfg[2]); nb = pk.nbytes + es.nbytes
    else:
        pk, es, mn, R, C = ck.quantizar_k(m, cfg[1], cfg[2], cfg[3])
        w = ck.desquantizar_k(pk, es, mn, R, C, (R, C), cfg[1], cfg[2]); nb = ck.bytes_k(pk, es, mn)
    alvo[n].data = torch.from_numpy(w.reshape(orig[n].shape)).float()
    return nb

def restaurar():
    for n, p in alvo.items(): p.data = orig[n].clone()

def rodar(cfg_de, sementes):
    """cfg_de(nome) -> cfg. Quantiza tudo, gera nas sementes, restaura. Devolve (% do fp16, psnr medio, ssim medio, ssim minimo)."""
    t0 = time.time(); tot = sum(aplicar(n, cfg_de(n)) for n in alvo); tq = time.time() - t0
    ps, ss = [], []
    for s in sementes:
        im = gerar(s); ps.append(psnr(base[s], im)); ss.append(ssim(base[s], im))
    restaurar()
    return 100 * tot / fp16_total, float(np.mean(ps)), float(np.mean(ss)), float(np.min(ss)), tq

nome = lambda c: ("sim" if c[0] == "s" else "assim%s" % ("+busca" if c[3] else "")) + " %dx%d %d bits" % (c[1][0], c[1][1], c[2])
S8 = ("s", (8, 8), 8)

# ------------------------------------------------------------ NUCLEO EVOLUTIVO (numpy puro, testavel sem torch)
OPCOES_BITS = np.array([8.25, 9.0, 12.0, 16.0])      # 8x8, 4x4, 2x2, fp16 (bits por peso, com a escala)
ERRO_REL_4x4 = np.array([2.9, 1.0, 0.23, 0.0])        # estimativa do erro de cada opcao relativo ao 4x4 (refinado pela evolucao)
PM0 = 0.12

def tamanho_frac(g, numel):
    return float((numel * OPCOES_BITS[g]).sum() / (numel.sum() * 16.0))

def mutar(g, imp, pm, rng):
    """Cada gene muda com prob. pm. Tensores IMPORTANTES tendem a subir de precisao, os outros a descer."""
    f = g.copy()
    mud = rng.random(len(g)) < pm
    sobe = rng.random(len(g)) < np.clip(0.15 + 0.7 * imp, 0.05, 0.95)
    passo = np.where(sobe, 1, -1)
    f[mud] = np.clip(f[mud] + passo[mud], 0, len(OPCOES_BITS) - 1)
    return f

def cruzar(a, b, rng):
    return np.where(rng.random(len(a)) < 0.5, a, b)

def evoluir(avaliar, numel, imp, orcamento, snr_meta, pop=10, geracoes=20, semente=0, log=print, iniciais=()):
    """avaliar(g) -> erro relativo da saida (menor = melhor). Quem passa do orcamento de tamanho MORRE;
    entre os vivos vence o menor erro. Para quando algum vivo atinge a meta de SNR (dB)."""
    rng = np.random.default_rng(semente)
    N = len(numel)
    cache = {}
    def pontuar(g):
        k = g.tobytes()
        if k not in cache:
            t = tamanho_frac(g, numel)
            cache[k] = (t, avaliar(g) if t <= orcamento else float("inf"))
        return cache[k]
    populacao = [np.asarray(g, np.int64) for g in iniciais][:pop]
    while len(populacao) < pop:
        base = populacao[rng.integers(0, len(populacao))] if populacao else np.ones(N, np.int64)
        populacao.append(mutar(base, imp, 0.1, rng))
    pm, sem_melhora, melhor_global = PM0, 0, (float("inf"), None, None)
    historico = []
    for ger in range(geracoes):
        ponts = [pontuar(g) for g in populacao]
        ordem = sorted(range(len(populacao)), key=lambda i: (ponts[i][1], ponts[i][0]))
        vivos = [i for i in ordem if np.isfinite(ponts[i][1])]
        mortos = len(populacao) - len(vivos)
        if vivos:
            b = vivos[0]; e, t = ponts[b][1], ponts[b][0]
            snr = -10 * np.log10(max(e, 1e-30))
            if e < melhor_global[0] * 0.999:
                melhor_global, sem_melhora = (e, populacao[b].copy(), t), 0
            else:
                sem_melhora += 1
            log("geracao %2d | melhor: %.1f dB com %.1f%% do tamanho | mortos por tamanho: %d | mutacao %.2f"
                % (ger, snr, 100 * t, mortos, pm))
            historico.append((ger, snr, t))
            if snr >= snr_meta:
                log("META ATINGIDA (>= %.0f dB dentro do orcamento)." % snr_meta)
                break
        else:
            log("geracao %2d | todos estouraram o orcamento" % ger)
        pm = min(0.35, pm * 1.5) if sem_melhora >= 3 else PM0
        sobreviventes = [populacao[i] for i in (vivos or ordem)[:max(2, pop // 2)]]
        elite = sobreviventes[:3]
        filhos = list(elite)
        while len(filhos) < pop:
            if rng.random() < 0.3 and len(sobreviventes) > 1:
                i, j = rng.choice(len(sobreviventes), 2, replace=False)
                f = mutar(cruzar(sobreviventes[i], sobreviventes[j], rng), imp, pm / 2, rng)
            else:
                f = mutar(sobreviventes[rng.integers(0, len(sobreviventes))], imp, pm, rng)
            filhos.append(f)
        populacao = filhos
    return melhor_global, historico


def guloso_inicial(efeito4, numel, orcamento):
    """Ponto de partida informado: usa o efeito MEDIDO de cada tensor em 4x4 e sobe, um degrau por vez, o tensor
    com maior ganho por bit extra, enquanto couber no orcamento. A evolucao refina o resto (efeitos nao aditivos)."""
    N = len(numel); g = np.zeros(N, np.int64)
    ganho = lambda i: efeito4[i] * (ERRO_REL_4x4[g[i]] - ERRO_REL_4x4[g[i] + 1]) / (numel[i] * (OPCOES_BITS[g[i] + 1] - OPCOES_BITS[g[i]]))
    ativo = set(range(N))
    while ativo:
        i = max(ativo, key=ganho)
        g[i] += 1
        if tamanho_frac(g, numel) > orcamento:
            g[i] -= 1; ativo.discard(i); continue
        if g[i] >= len(OPCOES_BITS) - 1: ativo.discard(i)
    return g


import json
# ------------------------------ METAS (mude aqui) ------------------------------
ORC = 0.62          # tamanho maximo (fracao do fp16). Passou disso, o individuo MORRE.
SNR_META = 50.0     # meta: UNet a 50 dB do original (praticamente indistinguivel); 100% so existe em fp16
POP, GER = 12, 20
OPC = [("s", (8, 8), 8), ("s", (4, 4), 8), ("s", (2, 2), 8), ("h",)]
NOMES_OPC = ["8x8", "4x4", "2x2", "fp16"]
names = list(alvo); N = len(names)
numel = np.array([alvo[n].numel() for n in names], np.int64)

# --- grava as chamadas REAIS do UNet numa geracao fp32 (entradas e saidas de cada um dos 4 passos) ---
trilha = []
def _gancho(mod, args, kwargs, out):
    trilha.append((args[0].detach().clone(), args[1].detach().clone() if torch.is_tensor(args[1]) else torch.tensor(args[1]),
                   kwargs["encoder_hidden_states"].detach().clone(), out[0].detach().clone().numpy()))
h_ = unet.register_forward_hook(_gancho, with_kwargs=True)
gerar(SEEDS[0]); h_.remove()
print("Gravadas %d chamadas reais do UNet (batch %d) na semente %d" % (len(trilha), trilha[0][0].shape[0], SEEDS[0]))
en = [float((o ** 2).sum()) for *_, o in trilha]
def reproduzir(passos=None):
    erro = tot = 0.0
    with torch.no_grad():
        for k, (x, t, enc, o) in enumerate(trilha):
            if passos is not None and k not in passos: continue
            y = unet(x, t, encoder_hidden_states=enc, return_dict=False)[0].numpy()
            erro += float(((y - o) ** 2).sum()); tot += en[k]
    return erro / tot
snr_db = lambda e: -10 * np.log10(max(e, 1e-30))

print("1) Sensibilidade de cada tensor (8 bits 4x4, um por vez) ...")
efeito = np.zeros(N)
for i, n in enumerate(names):
    aplicar(n, OPC[1]); efeito[i] = reproduzir(passos=(0, 2)); alvo[n].data = orig[n].clone()
    if (i + 1) % 30 == 0: print("   %d/%d" % (i + 1, N))
imp = np.argsort(np.argsort(efeito / numel)) / N          # importancia por peso (0 a 1)
print("   mais sensiveis: " + ", ".join(names[i][-34:] for i in np.argsort(-efeito)[:5]))

estado = np.full(N, -1)
def carregar(g):
    for i in np.where(estado != g)[0]:
        aplicar(names[i], OPC[g[i]]); estado[i] = g[i]
def avaliar(g):
    carregar(g); return reproduzir()

def regra(prefixos, alto=3):                              # achado do diagnostico: erro concentrado em conv_in/out e up_blocks.2
    g = np.ones(N, np.int64)
    for i, n in enumerate(names):
        if n.startswith(prefixos): g[i] = alto
    return g
RA = regra(("conv_in", "conv_out", "up_blocks.2."))
RB = regra(("conv_in", "conv_out", "up_blocks.2.", "down_blocks.0."))
t4 = np.ones(N, np.int64); t8 = np.zeros(N, np.int64)
gi = guloso_inicial(efeito, numel, ORC)
ini = {"4x4 puro (referencia)": t4, "regra A (conv_in/out + up_blocks.2 em fp16)": RA,
       "regra B (A + down_blocks.0 em fp16)": RB, "guloso informado": gi}
print("\nPontos de partida (UNet dB nas 4 chamadas reais | tamanho):")
for rot, g in ini.items():
    print("   %-46s %.1f dB | %.1f%%" % (rot, snr_db(avaliar(g)), 100 * tamanho_frac(g, numel)))
print("\n2) EVOLUCAO (meta %.0f dB, orcamento %.0f%% do fp16). Quem estoura o orcamento morre; os melhores geram filhos." % (SNR_META, 100 * ORC))
iniciais = [g for g in (RA, RB, gi, t4, t8) if tamanho_frac(g, numel) <= ORC]
(e_b, g_b, t_b), hist = evoluir(avaliar, numel, imp, ORC, SNR_META, pop=POP, geracoes=GER, iniciais=iniciais)
print("\nCAMPEAO: %.1f dB com %.1f%% do tamanho | " % (snr_db(e_b), 100 * t_b)
      + ", ".join("%s: %d tensores" % (NOMES_OPC[k], int((g_b == k).sum())) for k in range(4)))
print("Meta de %.0f dB %s." % (SNR_META, "ATINGIDA" if snr_db(e_b) >= SNR_META else "NAO atingida dentro do orcamento (melhor possivel achado)"))

def imagens(g):
    carregar(g)
    ims = [gerar(s) for s in SEEDS]
    return (float(np.mean([psnr(base[s], i) for s, i in zip(SEEDS, ims)])),
            float(np.mean([ssim(base[s], i) for s, i in zip(SEEDS, ims)])),
            float(np.min([ssim(base[s], i) for s, i in zip(SEEDS, ims)])))
print("\n3) Validacao em IMAGENS (4 sementes; referencia ja medida: 4x4 puro 0.963 / pior 0.940, 'protege 90%' 0.989 / 0.986):")
print("%-46s %8s %9s | %7s %7s %7s" % ("configuracao", "% fp16", "UNet dB", "PSNR", "SSIM", "pior"))
cands = dict(ini); cands["CAMPEAO da evolucao"] = g_b
for rot, g in cands.items():
    p, sm, pior = imagens(g)
    print("%-46s %7.1f%% %8.1f | %5.1f dB %.4f %.4f" % (rot, 100 * tamanho_frac(g, numel), snr_db(avaliar(g)), p, sm, pior))

# --- grava o .ttq FINAL com a receita campea (cubo/precisao por tensor) ---
BLOCO_GENE = {0: (8, 8), 1: (4, 4), 2: (2, 2)}
receita = {names[i]: ("fp16" if g_b[i] == 3 else (BLOCO_GENE[int(g_b[i])], 8)) for i in range(N)}
est = {k: v.detach().float().numpy() for k, v in unet.state_dict().items() if v.is_floating_point()}
info = cq.salvar(est, "tinysd_unet_final.ttq", (4, 4), 8, receita=receita)
print("\nGravado tinysd_unet_final.ttq = %.1f MB (fp32 original %.1f MB)" % (info["arquivo_mb"], info["bruto_fp32_mb"]))
json.dump({"opcoes": NOMES_OPC, "tamanho_pct": 100 * t_b, "snr_unet_db": snr_db(e_b),
           "genoma": {n: NOMES_OPC[int(g_b[i])] for i, n in enumerate(names)}}, open("tinysd_genoma.json", "w"), indent=1)
try:
    from google.colab import files
    files.download("tinysd_genoma.json")
except Exception:
    pass
print("Me mande esta saida e o tinysd_genoma.json. O .ttq final (~360 MB) fica no Colab para o colab_validar_unet.py.")
