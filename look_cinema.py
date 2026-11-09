"""Look de cinema (Hollywood) aplicado DEPOIS da imagem pronta, 100% local (numpy + Pillow).
Faz o que um colorista faz: curva de contraste em S, cores separadas (sombras frias / luzes quentes), brilho suave nas
luzes (bloom), vinheta, granulacao de filme e nitidez fina. Nao inventa detalhe: melhora o acabamento.
Honestidade: o modelo continua sendo o SDXS; isto da o 'acabamento' de cinema, nao troca o modelo."""
import numpy as np
from PIL import Image, ImageFilter


def _curva_s(x, forca):
    """Contraste em S suave (preserva pretos e brancos)."""
    s = x * x * (3.0 - 2.0 * x)
    return x + (s - x) * forca


def aplicar(img, forca=1.0, graos=0.022, semente=7):
    """img: PIL.Image ou array uint8 (H,W,3). Devolve PIL.Image RGB. forca 0 = original, 1 = look completo."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    f = max(0.0, min(float(forca), 1.5))
    if f == 0:
        return img
    base = np.asarray(img, dtype=np.float32) / 255.0
    H, W = base.shape[:2]
    lum = (0.2126 * base[..., 0] + 0.7152 * base[..., 1] + 0.0722 * base[..., 2])[..., None]

    # 1) contraste em S + pretos profundos
    x = _curva_s(np.clip(base, 0, 1), 0.75 * f)
    x = x ** (1.0 - 0.06 * f)                                    # levanta um pouco os meios-tons
    x = np.clip((x - 0.015 * f) / (1.0 - 0.015 * f), 0, 1)

    # 2) cor de cinema: sombras para azul-esverdeado, luzes para quente (teal & orange suave)
    l2 = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    sombra = (1.0 - l2) ** 2
    luz = l2 ** 2
    x = x + f * (sombra * np.array([-0.035, 0.0, 0.045], np.float32) + luz * np.array([0.045, 0.015, -0.035], np.float32))
    sat = 1.0 + 0.10 * f                                         # cor viva, mas natural
    cinza = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = np.clip(cinza + (x - cinza) * sat, 0, 1)

    # 3) bloom: as luzes fortes "vazam" um brilho suave
    brilho = np.clip((l2 - 0.62) / 0.38, 0, 1) * x
    raio = max(2.0, min(H, W) * 0.018)
    halo = np.asarray(Image.fromarray((brilho * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(raio)),
                      dtype=np.float32) / 255.0
    x = 1.0 - (1.0 - x) * (1.0 - 0.55 * f * halo)               # mistura "screen"

    # 4) vinheta (cantos mais escuros, olho vai para o centro)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.414
    vin = 1.0 - 0.27 * f * np.clip(d, 0, 1) ** 2.2
    x = x * vin[..., None]

    # 5) granulacao de filme (mais nos tons medios; estavel pela semente)
    if graos > 0:
        rng = np.random.RandomState(semente)
        ruido = rng.normal(0.0, 1.0, (H, W, 1)).astype(np.float32)
        peso = 1.0 - np.abs(l2 - 0.5) * 1.2
        x = x + ruido * graos * f * np.clip(peso, 0.3, 1.0)

    out = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    # 6) nitidez fina (detalhe de lente)
    return out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(70 * f), threshold=2))
