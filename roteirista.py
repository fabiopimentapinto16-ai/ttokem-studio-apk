"""O 'cerebro' das cenas: transforma UMA ideia em varias cenas profissionais (prompt de imagem em ingles,
narracao e legenda). Com internet e chave de IA usa o Maestro de IAs do app; sem isso usa o roteirista local
(modelos de plano de cinema), que funciona 100% offline."""
import json
import re

# estilo que o SDXS/DreamShaper entende bem: foto realista, luz de cinema, sem texto na imagem
ESTILO = ("RAW photo, cinematic film still, soft directional key light, creamy bokeh, rich film color grade, "
          "deep blacks, luminous highlights, anamorphic 35mm, highly detailed, sharp focus")

SYSTEM_CENAS = (
    "Voce e diretor de cena de videos curtos verticais para redes sociais. Responda SOMENTE com JSON valido: "
    '{"titulo": str, "cenas": [{"prompt": str, "narracao": str, "legenda": str}]} com EXATAMENTE {N} cenas. '
    '"prompt": descricao visual da cena em INGLES, 25 a 45 palavras, concreta (sujeito, acao, local, luz, lente), '
    "realista, SEM texto na imagem, SEM pessoas famosas, cenas diferentes entre si e coerentes na historia. "
    '"narracao": 1 frase curta para ser falada (ate 18 palavras). '
    '"legenda": ate 6 palavras. "narracao" e "legenda" ficam NO MESMO IDIOMA em que a ideia foi escrita (qualquer idioma), '
    'mas "prompt" e SEMPRE em INGLES. A cena 1 e um gancho forte; a ultima fecha com chamada para acao.')

# planos de cinema: como contar a mesma ideia em 4 planos diferentes (funciona sem internet)
PLANOS = (
    ("wide establishing shot of {x}, golden hour, epic atmosphere", "Tudo comeca aqui."),
    ("medium shot of {x}, natural light, rich colors, subtle motion", "Olha esse detalhe."),
    ("extreme close-up detail of {x}, macro lens, soft bokeh, dramatic light", "De pertinho, e ainda melhor."),
    ("hero shot of {x}, low angle, cinematic backlight, vibrant, polished", "Pronto para voce viver isso."),
)

PLANOS_LONGOS = (
    "{x}, wide establishing shot, golden hour, epic atmosphere",
    "{x}, medium shot, natural light, rich colors, subtle motion",
    "{x}, extreme close-up detail, macro lens, soft bokeh, dramatic light",
    "{x}, hero low angle shot, cinematic backlight, vibrant, polished",
)
_LIXO_INICIO = re.compile(r"^\s*(the\s+)?(video|clip|scene|image|photo)\s+(shows|of|with|depicts)\s+", re.I)


def _limpar_ideia(x, max_palavras=26):
    """Tira 'The video shows ...' do comeco e limita o tamanho (o CLIP so le ~77 tokens)."""
    x = re.sub(r"\s+", " ", str(x or "")).strip().rstrip(".,")
    x = _LIXO_INICIO.sub("", x).strip()
    return " ".join(x.split(" ")[:max_palavras])


def titulo_curto(x, palavras=6):
    """Primeiras palavras da ideia, cortando em palavra inteira (nunca no meio)."""
    p = re.sub(r"\s+", " ", str(x or "")).strip().rstrip(".,").split(" ")
    return " ".join(p[:palavras]) + ("..." if len(p) > palavras else "")


def estilizar(prompt):
    """Acrescenta o estilo profissional ao prompt (sem repetir se ja tiver)."""
    p = re.sub(r"\s+", " ", str(prompt or "")).strip().rstrip(".,")
    return p if "cinematic" in p.lower() else "%s, %s" % (p, ESTILO)


def cenas_offline(ideia, n=4, idioma="pt", ideia_en=None):
    """Roteirista local: 1 ideia -> n planos de cinema. `ideia_en` = a ideia ja em ingles (para o prompt de imagem);
    a narracao sai no idioma da pessoa."""
    legenda_orig = re.sub(r"\s+", " ", str(ideia or "")).strip()
    x = _limpar_ideia(ideia_en or ideia) or "the subject"
    longa = len(x.split()) > 8
    n = max(2, min(int(n), len(PLANOS)))
    out = []
    for i in range(n):
        k = i if n == len(PLANOS) else round(i * (len(PLANOS) - 1) / (n - 1))
        xi = x
        if longa:                                       # frase longa: cada cena foca numa parte da frase
            partes = [p.strip() for p in re.split(r",| and then | then | while | and ", x) if len(p.strip()) > 3]
            if len(partes) >= 2:
                xi = partes[0] + (", " + partes[min(k, len(partes) - 1)] if k >= 1 else "")
        modelo = PLANOS_LONGOS[k].replace("{x}", "{x}") if longa else PLANOS[k][0]
        fala = FALAS[idioma][k] if idioma in FALAS else (legenda_orig if i == 0 else "")
        out.append({"prompt": estilizar(modelo.format(x=xi)), "narracao": fala, "legenda": titulo_curto(legenda_orig) if i == 0 else ""})
    return {"titulo": titulo_curto(legenda_orig), "cenas": out, "origem": "local", "idioma": idioma}


# ------------------------------------------------------------------ qualquer idioma
_PALAVRAS = {
    "pt": "o a os as de do da dos das em no na um uma para por com que e nao se mais como mas ao uma seu sua voce",
    "en": "the a an of and to in is on at for with that this it as by from or be are was your you",
    "es": "el la los las de del en un una para por con que y no se mas como pero al su tu usted es",
    "fr": "le la les des du de un une pour par avec que et ne se plus comme mais au sa son est vous",
    "de": "der die das den dem des ein eine und zu in ist mit auf fur von nicht sich auch wie aber",
    "it": "il lo la gli le di del un una per con che e non si piu come ma al suo sua tu sei",
}
NOME_IDIOMA = {"pt": "Portugues do Brasil", "en": "English", "es": "Espanol", "fr": "Francais", "de": "Deutsch",
               "it": "Italiano"}
LOCALE_TTS = {"pt": "pt-BR", "en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "it": "it-IT"}
FALAS = {   # narracao do roteirista local (4 planos) nos idiomas mais comuns
    "pt": ("Tudo comeca aqui.", "Olha esse detalhe.", "De pertinho, e ainda melhor.", "Pronto para voce viver isso."),
    "en": ("It all starts here.", "Look at this detail.", "Up close, it's even better.", "Ready for you to live it."),
    "es": ("Todo empieza aqui.", "Mira este detalle.", "De cerca, es aun mejor.", "Listo para que lo vivas."),
    "fr": ("Tout commence ici.", "Regardez ce detail.", "De pres, c'est encore mieux.", "Pret a etre vecu."),
    "de": ("Alles beginnt hier.", "Sieh dir dieses Detail an.", "Aus der Nahe noch besser.", "Bereit zum Erleben."),
    "it": ("Tutto comincia qui.", "Guarda questo dettaglio.", "Da vicino e ancora meglio.", "Pronto per essere vissuto."),
}


def _sem_acento(t):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def detectar_idioma(texto):
    """Codigo do idioma ('pt','en','es','fr','de','it') ou 'xx' (outro: japones, arabe, russo...). Heuristica simples."""
    t = str(texto or "")
    letras = [c for c in t if c.isalpha()]
    if not letras:
        return "en"
    if sum(1 for c in letras if ord(c) > 0x024F) > len(letras) * 0.3:        # escrita nao latina
        return "xx"
    palavras = set(re.findall(r"[a-z]+", _sem_acento(t)))
    pont = {k: len(palavras & set(v.split())) for k, v in _PALAVRAS.items()}
    if "ç" in t.lower() or "ã" in t.lower() or "õ" in t.lower():
        pont["pt"] += 2
    if "ñ" in t.lower() or "¿" in t or "¡" in t:
        pont["es"] += 2
    melhor = max(pont, key=pont.get)
    return melhor if pont[melhor] > 0 else "en"


def locale_tts(cod):
    return LOCALE_TTS.get(cod, cod if len(str(cod)) == 2 and cod != "xx" else "en-US")


def traduzir_en(texto, timeout=8):
    """Traduz para ingles pelo servico gratis do Google Translate (precisa de internet; sem chave, nao oficial: pode
    mudar). Devolve (ingles, idioma_origem) ou None."""
    try:
        import requests
        r = requests.get("https://translate.googleapis.com/translate_a/single",
                         params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": str(texto)[:500]},
                         timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        d = r.json()
        en = "".join(p[0] for p in d[0] if p and p[0]).strip()
        return (en, str(d[2])) if en else None
    except Exception:
        return None


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
            {"role": "system", "content": SYSTEM_CENAS.replace("{N}", str(n))},
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


def montar_cenas(ideia_ou_linhas, llm=None, idioma="Portugues do Brasil", n=4, usar_ia=True, traduzir=True):
    """Entrada: texto do usuario em QUALQUER idioma. Varias linhas = cada linha e uma cena. Uma linha = o cerebro cria
    as cenas. Ordem para o prompt de imagem sair em ingles: 1) IA (traduz e roteiriza), 2) tradutor gratis (internet),
    3) offline: usa o texto como esta (so funciona bem se ja estiver em ingles). O resultado traz "idioma" (codigo) para a voz."""
    linhas = [l.strip() for l in str(ideia_ou_linhas or "").splitlines() if len(l.strip()) >= 3]
    cod = detectar_idioma(" ".join(linhas))
    if len(linhas) >= 2:
        cenas = []
        for l in linhas[:8]:
            en = l
            if cod != "en" and traduzir:
                t = traduzir_en(l)
                en = t[0] if t else l
            cenas.append({"prompt": estilizar(en), "narracao": l, "legenda": l[:50]})
        return {"titulo": linhas[0][:40], "origem": "suas linhas", "cenas": cenas, "idioma": cod}
    ideia = linhas[0] if linhas else ""
    if usar_ia and llm is not None:
        r = cenas_ia(ideia, llm, idioma, n)
        if r:
            r["idioma"] = cod
            return r
    ideia_en, aviso = ideia, ""
    if cod != "en" and traduzir:
        t = traduzir_en(ideia)
        if t:
            ideia_en, cod = t[0], (t[1][:2] if len(t[1]) >= 2 else cod)
        else:
            aviso = "sem tradutor: usei o texto como esta (em ingles a imagem sai melhor)"
    r = cenas_offline(ideia, n, idioma=cod, ideia_en=ideia_en)
    if aviso:
        r["aviso"] = aviso
    return r


def prompt_imagem(texto, traduzir=True):
    """Para o botao GERAR IMAGEM: qualquer idioma -> prompt em ingles com o estilo Hollywood. Devolve (prompt, aviso)."""
    t = re.sub(r"\s+", " ", str(texto or "")).strip()
    cod, aviso = detectar_idioma(t), ""
    if cod != "en" and traduzir:
        tr = traduzir_en(t)
        if tr:
            t = tr[0]
        else:
            aviso = "sem tradutor: usei o texto como esta (em ingles a imagem sai melhor)"
    return estilizar(_limpar_ideia(t, 40)), aviso
