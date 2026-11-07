"""CUBO-QUANT: quantizacao de pesos em CUBOS (blocos) de 4 bits + leitor em streaming.

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


def desquantizar_rapido(pack, esc, R, C, forma, bloco=(8, 8), dtype=np.float32):
    """Versao rapida para 8 bits (medida no Galaxy A5: ~2.8x mais rapida que `desquantizar`).
    Escreve direto numa saida pre-alocada, sem temporarios; resultado IDENTICO ao `desquantizar`."""
    bh, bw = bloco
    nbh, nbw = _alinha(R, bh) // bh, _alinha(C, bw) // bw
    e = esc.astype(np.float32).reshape(nbh, 1, nbw, 1)
    out = np.empty((nbh, bh, nbw, bw), dtype=np.float32)
    np.subtract(pack.reshape(nbh, nbw, bh, bw).transpose(0, 2, 1, 3), 128, out=out, dtype=np.float32)
    out *= e
    m = out.reshape(nbh * bh, nbw * bw)
    if m.shape != (R, C):                                  # corta o enchimento
        m = np.ascontiguousarray(m[:R, :C])
    return m.reshape(forma).astype(dtype, copy=False)


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
        if bits_t == 8:                                      # caminho rapido (identico ao lento)
            return desquantizar_rapido(pack, esc, t["R"], t["C"], forma, bloco_t, dtype)
        return desquantizar(pack, esc, t["R"], t["C"], forma, bloco_t, dtype, bits_t)

    def ler_filas(self, nome, filas, dtype=np.float32):
        """Le SO as linhas pedidas (ex.: 77 tokens de um embedding de 49408 linhas), sem desquantizar o tensor todo."""
        filas = np.asarray(filas, np.int64).reshape(-1)
        t = self.meta[nome]
        forma = tuple(t["forma"])
        base = self._ini
        if t["modo"] == "f16":
            d = np.frombuffer(self._mm, np.float16, count=int(np.prod(forma, dtype=np.int64)), offset=base + t["off"])
            return d.reshape(forma[0], -1)[filas].astype(dtype)
        bloco_t, bits_t = tuple(t.get("bloco", self.bloco)), int(t.get("bits", self.bits))
        bh, bw = bloco_t
        R, C = t["R"], t["C"]
        nbw = _alinha(C, bw) // bw
        por = nbw * bh * bw
        if "off_pr" in t or (bits_t == 4 and por % 2):       # casos raros: le tudo
            return self.ler(nome, dtype).reshape(R, -1)[filas]
        pack = np.frombuffer(self._mm, np.uint8, count=t["n"], offset=base + t["off"])
        esc = np.frombuffer(self._mm, np.float16, count=t["n_esc"] // 2, offset=base + t["off_esc"])
        esc = esc.reshape(_alinha(R, bh) // bh, nbw)
        bandas = {}
        for a in np.unique(filas // bh):
            ps = pack[a * por: (a + 1) * por] if bits_t == 8 else pack[a * por // 2: (a + 1) * por // 2]
            bandas[int(a)] = desquantizar(ps, esc[a:a + 1], bh, C, (bh, C), bloco_t, np.float32, bits_t)
        return np.stack([bandas[int(f // bh)][int(f % bh)] for f in filas]).astype(dtype, copy=False)

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
