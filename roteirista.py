"""O 'cerebro' das cenas: transforma UMA ideia em varias cenas profissionais (prompt de imagem em ingles,
narracao e legenda). Com internet e chave de IA usa o Maestro de IAs do app; sem isso usa o roteirista local
(modelos de plano de cinema), que funciona 100% offline."""
import json
import re

# estilo que o SDXS/DreamShaper entende bem: foto realista, luz de cinema, sem texto na imagem
ESTILO = "photorealistic, cinematic lighting, 35mm photo, shallow depth of field, highly detailed, sharp focus"

SYSTEM_CENAS = (
    "Voce e diretor de cena de videos curtos verticais para redes sociais. Responda SOMENTE com JSON valido: "
    '{"titulo": str, "cenas": [{"prompt": str, "narracao": str, "legenda": str}]} com EXATAMENTE {N} cenas. '
    '"prompt": descricao visual da cena em INGLES, 25 a 45 palavras, concreta (sujeito, acao, local, luz, lente), '
    "realista, SEM texto na imagem, SEM pessoas famosas, cenas diferentes entre si e coerentes na historia. "
    '"narracao": 1 frase curta em {IDIOMA} para ser falada (ate 18 palavras). '
    '"legenda": ate 6 palavras em {IDIOMA}. A cena 1 e um gancho forte; a ultima fecha com chamada para acao.')

# planos de cinema: como contar a mesma ideia em 4 planos diferentes (funciona sem internet)
PLANOS = (
    ("wide establishing shot of {x}, golden hour, epic atmosphere", "Tudo comeca aqui."),
    ("medium shot of {x}, natural light, rich colors, subtle motion", "Olha esse detalhe."),
    ("extreme close-up detail of {x}, macro lens, soft bokeh, dramatic light", "De pertinho, e ainda melhor."),
    ("hero shot of {x}, low angle, cinematic backlight, vibrant, polished", "Pronto para voce viver isso."),
)


def estilizar(prompt):
    """Acrescenta o estilo profissional ao prompt (sem repetir se ja tiver)."""
    p = re.sub(r"\s+", " ", str(prompt or "")).strip().rstrip(".,")
    return p if "cinematic" in p.lower() else "%s, %s" % (p, ESTILO)


def cenas_offline(ideia, n=4):
    """Roteirista local: 1 ideia -> n planos de cinema. Funciona melhor com a ideia escrita em ingles."""
    x = re.sub(r"\s+", " ", str(ideia or "")).strip().rstrip(".,") or "the subject"
    n = max(2, min(int(n), len(PLANOS)))
    out = []
    for i in range(n):
        modelo, fala = PLANOS[i if n == len(PLANOS) else round(i * (len(PLANOS) - 1) / (n - 1))]
        out.append({"prompt": estilizar(modelo.format(x=x)), "narracao": fala, "legenda": x[:40] if i == 0 else ""})
    return {"titulo": x[:40], "cenas": out, "origem": "local"}


def _parse(texto):
    t = re.sub(r"^```(?:json)?|```$", "", str(texto).strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else t)
    except (ValueError, AttributeError):
        return None


def cenas_ia(ideia, llm, idioma="Portugues do Brasil", n=4):
    """Pede ao Maestro de IAs. Devolve o mesmo formato de cenas_offline, ou None se falhar (sem rede/chave/JSON ruim)."""
    try:
        raw, api = llm.chat([
            {"role": "system", "content": SYSTEM_CENAS.replace("{N}", str(n)).replace("{IDIOMA}", idioma)},
            {"role": "user", "content": "Ideia do video: %s" % str(ideia).strip()[:400]}], max_tokens=1100)
    except Exception:
        return None
    d = _parse(raw)
    cenas = []
    for c in (d or {}).get("cenas") or []:
        if isinstance(c, dict) and len(str(c.get("prompt", ""))) >= 8:
            cenas.append({"prompt": estilizar(c["prompt"]), "narracao": str(c.get("narracao", "")).strip()[:160],
                          "legenda": str(c.get("legenda", "")).strip()[:60]})
    if len(cenas) < 2:
        return None
    return {"titulo": str((d or {}).get("titulo", ""))[:60], "cenas": cenas[:n], "origem": "IA (%s)" % api}


def montar_cenas(ideia_ou_linhas, llm=None, idioma="Portugues do Brasil", n=4, usar_ia=True):
    """Entrada: texto do usuario. Varias linhas = cada linha e uma cena (voce manda). Uma linha so = o cerebro
    cria as cenas (IA se houver; senao o roteirista local)."""
    linhas = [l.strip() for l in str(ideia_ou_linhas or "").splitlines() if len(l.strip()) >= 3]
    if len(linhas) >= 2:
        return {"titulo": linhas[0][:40], "origem": "suas linhas",
                "cenas": [{"prompt": estilizar(l), "narracao": l, "legenda": l[:50]} for l in linhas[:8]]}
    ideia = linhas[0] if linhas else ""
    if usar_ia and llm is not None:
        r = cenas_ia(ideia, llm, idioma, n)
        if r:
            return r
    return cenas_offline(ideia, n)
