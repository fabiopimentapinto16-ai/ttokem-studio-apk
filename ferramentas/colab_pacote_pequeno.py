# Roda NO MESMO Colab, depois do colab_exportar_tinysd.py. Gera um zip PEQUENO (poucos MB) para eu
# conferir a arquitetura e as referencias sem precisar dos 590 MB de pesos.
import json, struct, zipfile, os

def cabecalho_ttq(caminho):
    with open(caminho, "rb") as f:
        assert f.read(4) == b"TTQ1"
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))

with zipfile.ZipFile("tinysd_pequeno.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for nome in ("unet", "texto", "vae"):
        h = cabecalho_ttq("tinysd_%s_q8.ttq" % nome)
        z.writestr("cab_%s.json" % nome, json.dumps(h))
    for a in ("tinysd_config.json", "tinysd_ref.npz", "tinysd_ref.png"):
        z.write(a)
print("tinysd_pequeno.zip = %.1f MB" % (os.path.getsize("tinysd_pequeno.zip") / 1e6))
try:
    from google.colab import files; files.download("tinysd_pequeno.zip")
except Exception:
    print("Baixe tinysd_pequeno.zip pelo painel de arquivos do Colab.")
