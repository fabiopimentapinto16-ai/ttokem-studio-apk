"""MP4 de verdade (H.264 + AAC) feito no proprio celular com o PyAV/ffmpeg do APK. Sem API, sem internet.
Se o PyAV nao existir (ou algo falhar), quem chama cai no GIF."""
import os
import wave

import video_cenas

ULTIMO_ERRO = [""]


def _ler_wav(caminho):
    """(amostras int16 mono como bytes, taxa) ou (None, 0). Aceita WAV PCM 16 bits mono/estereo (mistura em mono)."""
    try:
        with wave.open(caminho, "rb") as w:
            if w.getsampwidth() != 2:
                return None, 0
            n, ch, sr = w.getnframes(), w.getnchannels(), w.getframerate()
            dados = w.readframes(n)
    except Exception:
        return None, 0
    if ch == 1:
        return dados, sr
    import array
    a = array.array("h")
    a.frombytes(dados)
    mono = array.array("h", (int(sum(a[i:i + ch]) / ch) for i in range(0, len(a) - ch + 1, ch)))
    return mono.tobytes(), sr


def montar_mp4(imagens, legendas, saida, duracoes=None, wav="", fps=15, lado=384, transicao=0.5, semente=""):
    """Grava o MP4 em `saida` (com a narracao `wav`, se houver) e devolve (saida, duracao_s). Levanta erro se faltar PyAV."""
    import av
    quadros, fps = video_cenas.quadros_das_cenas(imagens, legendas, duracoes, fps, lado, transicao, semente)
    lado = quadros[0].size[0] - (quadros[0].size[0] % 2)
    pcm, sr = _ler_wav(wav) if wav else (None, 0)
    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    tmp = saida + ".tmp.mp4"
    c = av.open(tmp, "w", format="mp4", options={"movflags": "faststart"})
    try:
        v = c.add_stream("libx264", rate=fps)
        v.width = v.height = lado
        v.pix_fmt = "yuv420p"
        v.options = {"preset": "veryfast", "crf": "24"}
        a = None
        if pcm:
            a = c.add_stream("aac", rate=sr)
            a.layout = "mono"
        LOTE = 1024
        total_amostras = len(pcm) // 2 if pcm else 0
        iv, ia = 0, 0
        while iv < len(quadros) or (a is not None and ia < total_amostras):
            t_v = iv / float(fps) if iv < len(quadros) else 1e18
            t_a = ia / float(sr) if (a is not None and ia < total_amostras) else 1e18
            if t_v <= t_a:                                   # intercala video e audio pelo tempo
                fr = av.VideoFrame.from_image(quadros[iv].resize((lado, lado)).convert("RGB"))
                fr.pts = iv
                for pkt in v.encode(fr):
                    c.mux(pkt)
                iv += 1
            else:
                bloco = pcm[ia * 2:(ia + LOTE) * 2]
                import numpy as np
                arr = np.frombuffer(bloco, dtype=np.int16).reshape(1, -1)
                fa = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")
                fa.sample_rate = sr
                fa.pts = ia
                for pkt in a.encode(fa):
                    c.mux(pkt)
                ia += LOTE
        for pkt in v.encode():
            c.mux(pkt)
        if a is not None:
            for pkt in a.encode():
                c.mux(pkt)
    finally:
        c.close()
    os.replace(tmp, saida)
    try:                                               # capa (1o quadro) para mostrar no app
        quadros[0].save(os.path.splitext(saida)[0] + "_capa.png")
    except Exception:
        pass
    return saida, len(quadros) / float(fps)
