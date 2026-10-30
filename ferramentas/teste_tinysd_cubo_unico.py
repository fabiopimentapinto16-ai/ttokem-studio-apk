# TTOKEN - BARREDURA DO CUBO-QUANT NO tiny-sd (modelo real). Colab/PC com torch; CPU basta.
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
def salvar(estado, caminho, bloco=(8, 8), bits=4, reordenar=False):
    """estado: dict nome -> ndarray. Grava o .ttq (atomico). Devolve estatisticas."""
    meta, blobs, pos = {}, [], 0
    bruto = 0
    for nome, arr in estado.items():
        arr = np.asarray(arr)
        bruto += arr.size * 4
        if arr.ndim >= 2 and arr.size >= MIN_ELEMENTOS:
            m = arr.astype(np.float32).reshape(arr.shape[0], -1)
            pr = pc = None
            if reordenar:                                  # "resolve o cubo": agrupa parecidos
                pr, pc = ordem_cubo(m)
                m = m[pr][:, pc]
            pack, esc, R, C = quantizar(m, bloco, bits)
            dados = [pack.tobytes(), esc.tobytes()]
            meta[nome] = {"modo": "q%d" % bits, "forma": list(arr.shape), "R": R, "C": C,
                          "off": pos, "n": len(dados[0]),
                          "off_esc": pos + _alinha(len(dados[0])), "n_esc": len(dados[1])}
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
        bh, bw = self.bloco
        esc = esc.reshape(_alinha(t["R"], bh) // bh, _alinha(t["C"], bw) // bw)
        if "off_pr" in t:                                    # desfaz a reordenacao do cubo
            m = desquantizar(pack, esc, t["R"], t["C"], (t["R"], t["C"]), self.bloco, np.float32, self.bits)
            pr = np.frombuffer(self._mm, np.uint32, count=t["R"], offset=base + t["off_pr"])
            pc = np.frombuffer(self._mm, np.uint32, count=t["C"], offset=base + t["off_pc"])
            ir, ic = np.argsort(pr), np.argsort(pc)
            return m[ir][:, ic].reshape(forma).astype(dtype, copy=False)
        return desquantizar(pack, esc, t["R"], t["C"], forma, self.bloco, dtype, self.bits)

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


'''
cq = types.ModuleType("cubo_quant"); exec(_CODIGO, cq.__dict__); sys.modules["cubo_quant"] = cq
from diffusers import AutoPipelineForText2Image

SEED, PROMPT, STEPS, SIZE = 5762, "Carro futurista correndo na chuva cyberpunk", 4, 256
pipe = AutoPipelineForText2Image.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
pipe.set_progress_bar_config(disable=True)
if hasattr(pipe, "safety_checker"): pipe.safety_checker = None
gerar = lambda: pipe(prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=7.5,
                     generator=torch.Generator("cpu").manual_seed(SEED), height=SIZE, width=SIZE).images[0]

def psnr(a, b):
    x, y = np.asarray(a.convert("RGB"), np.float64), np.asarray(b.convert("RGB"), np.float64)
    m = np.mean((x - y) ** 2); return float("inf") if m == 0 else 10 * np.log10(255 ** 2 / m)
def ssim(a, b):                      # SSIM de janela (padrao); cai no global se nao houver skimage
    try:
        from skimage.metrics import structural_similarity as S
        return float(S(np.asarray(a.convert("L")), np.asarray(b.convert("L")), data_range=255))
    except Exception:
        x, y = [np.asarray(i.convert("L"), np.float64) for i in (a, b)]
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2; mx, my = x.mean(), y.mean()
        return float(((2*mx*my+c1)*(2*((x-mx)*(y-my)).mean()+c2))/((mx**2+my**2+c1)*(x.var()+y.var()+c2)))

unet = pipe.unet
alvo = {n: p for n, p in unet.named_parameters() if p.ndim >= 2 and p.numel() >= cq.MIN_ELEMENTOS}
orig = {n: p.data.clone() for n, p in alvo.items()}
fp16_bytes = sum(p.numel() * 2 for p in alvo.values())
print("tensores 2D+ do UNet: %d | %.1f MB em fp16 (convolucoes incluidas)" % (len(alvo), fp16_bytes / 1e6))
base = gerar(); base.save("base.png")

def rodar(rotulo, bits, bloco, reordenar):
    t = time.time(); bytes_q = 0
    for n, p in alvo.items():
        m = orig[n].numpy().reshape(orig[n].shape[0], -1)
        if reordenar:
            pr, pc = cq.ordem_cubo(m); m = m[pr][:, pc]
        pk, es, R, C = cq.quantizar(m, bloco, bits)
        rec = cq.desquantizar(pk, es, R, C, (R, C), bloco, bits=bits)
        if reordenar:
            rec = rec[np.argsort(pr)][:, np.argsort(pc)]
        p.data = torch.from_numpy(rec.reshape(orig[n].shape)).float()
        bytes_q += pk.nbytes + es.nbytes + (8 * (R + C) // 2 if reordenar else 0)
    img = gerar()
    print("%-34s tamanho %5.1f%% do fp16 | PSNR %5.1f dB | SSIM %.4f | %.0fs" %
          (rotulo, 100 * bytes_q / fp16_bytes, psnr(base, img), ssim(base, img), time.time() - t))
    img.save("cfg_%s.png" % rotulo.replace(" ", "_").replace("/", "-"))
    for n, p in alvo.items(): p.data = orig[n].clone()

print("\nconfiguracao                        resultado")
rodar("8 bits, cubo 8x8", 8, (8, 8), False)
rodar("8 bits, cubo 64x64 (igual ao seu)", 8, (64, 64), False)
rodar("4 bits, cubo 64x64 (igual ao seu)", 4, (64, 64), False)
rodar("4 bits, cubo 8x8", 4, (8, 8), False)
rodar("4 bits, cubo 4x4", 4, (4, 4), False)
rodar("4 bits, cubo 1x32 (estilo Q4_0)", 4, (1, 32), False)
rodar("4 bits, cubo 8x8 reordenado", 4, (8, 8), True)
rodar("4 bits, cubo 1x32 reordenado", 4, (1, 32), True)
