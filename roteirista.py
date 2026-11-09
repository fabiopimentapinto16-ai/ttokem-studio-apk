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
    '"legenda": ate 6 palavras. "narracao" e "legenda" sao escritas em {IDIOMA} (a ideia pode vir em qualquer idioma), '
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
        fala = FALAS.get(idioma, FALAS["en"])[k]          # idioma sem falas prontas: ingles (o app traduz se houver internet)
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

_NOMES = {   # nome (sem acento, minusculo) -> codigo. A pessoa pode escrever o idioma em portugues, ingles ou no proprio idioma
    "pt": "portugues portuguese portugues do brasil brasil pt", "en": "english ingles en", "es": "espanol spanish es",
    "fr": "francais frances french fr", "de": "deutsch alemao german de", "it": "italiano italian it",
    "ja": "japones japanese nihongo ja", "zh": "chines chinese mandarim zhongwen zh", "ko": "coreano korean ko",
    "ru": "russo russian ru", "ar": "arabe arabic ar", "hi": "hindi ou indiano hi", "tr": "turco turkish turkce tr",
    "nl": "holandes dutch nederlands nl", "pl": "polones polish polski pl", "sv": "sueco swedish svenska sv",
    "id": "indonesio indonesian id", "vi": "vietnamita vietnamese vi", "th": "tailandes thai th",
    "uk": "ucraniano ukrainian uk", "el": "grego greek el", "he": "hebraico hebrew he", "ro": "romeno romanian ro",
    "cs": "tcheco czech cs", "sw": "suaili swahili sw", "la": "latim latin la",
}
_NATIVOS = {"日本語": "ja", "中文": "zh", "汉语": "zh", "한국어": "ko", "русский": "ru", "العربية": "ar", "हिन्दी": "hi",
            "türkçe": "tr", "ελληνικά": "el", "עברית": "he", "ไทย": "th", "українська": "uk"}
NOMES_EXIBICAO = {"pt": "Português", "en": "English", "es": "Español", "fr": "Français", "de": "Deutsch", "it": "Italiano",
                  "ja": "日本語", "zh": "中文", "ko": "한국어", "ru": "Русский", "ar": "العربية", "hi": "हिन्दी", "tr": "Türkçe"}


def codigo_idioma(nome, padrao="pt"):
    """'Português do Brasil' -> 'pt'; 'Japones' / '日本語' -> 'ja'. Nome desconhecido -> padrao."""
    t = str(nome or "").strip()
    if not t:
        return padrao
    if t.lower() in _NATIVOS:
        return _NATIVOS[t.lower()]
    if t in _NATIVOS:
        return _NATIVOS[t]
    s = _sem_acento(t)
    for cod, nomes in _NOMES.items():
        if s == cod or s in nomes.split(" ") or any(s.startswith(n) for n in nomes.split(" ") if len(n) >= 4):
            return cod
    return padrao


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


def traduzir(texto, destino="en", timeout=8):
    """Traduz para `destino` (codigo: 'pt','en','ja'...) pelo servico gratis do Google Translate (precisa de internet;
    sem chave, nao oficial: pode mudar). Devolve (texto_traduzido, idioma_origem) ou None."""
    try:
        import requests
        r = requests.get("https://translate.googleapis.com/translate_a/single",
                         params={"client": "gtx", "sl": "auto", "tl": destino, "dt": "t", "q": str(texto)[:500]},
                         timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        d = r.json()
        en = "".join(p[0] for p in d[0] if p and p[0]).strip()
        return (en, str(d[2])) if en else None
    except Exception:
        return None


def traduzir_en(texto, timeout=8):
    return traduzir(texto, "en", timeout)


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


def _no_idioma(texto, dest, origem):
    """Texto no idioma de destino: traduz se for diferente (internet); senao devolve como esta."""
    if not texto or dest == origem:
        return texto
    t = traduzir(texto, dest)
    return t[0] if t else texto


def montar_cenas(ideia_ou_linhas, llm=None, idioma="Portugues do Brasil", n=4, usar_ia=True, traduzir_ok=True):
    """Entrada em QUALQUER idioma. `idioma` = idioma de SAIDA da narracao, da legenda e da voz (o que a pessoa escolheu).
    O prompt de imagem sai sempre em ingles. Varias linhas = cada linha e uma cena; uma linha = o cerebro cria as cenas
    (IA se houver; senao o roteirista local, com falas prontas em pt/en/es/fr/de/it e traducao pela internet nos demais).
    Resultado traz "idioma" (codigo de saida) para a voz."""
    linhas = [l.strip() for l in str(ideia_ou_linhas or "").splitlines() if len(l.strip()) >= 3]
    ori = detectar_idioma(" ".join(linhas))
    dest = codigo_idioma(idioma, padrao=ori if ori != "xx" else "en")
    if len(linhas) >= 2:
        cenas = []
        for l in linhas[:8]:
            en = l
            if ori != "en" and traduzir_ok:
                t = traduzir_en(l)
                en = t[0] if t else l
            fala = _no_idioma(l, dest, ori) if traduzir_ok else l
            cenas.append({"prompt": estilizar(en), "narracao": fala, "legenda": fala[:50]})
        return {"titulo": linhas[0][:40], "origem": "suas linhas", "cenas": cenas, "idioma": dest}
    ideia = linhas[0] if linhas else ""
    if usar_ia and llm is not None:
        r_ = cenas_ia(ideia, llm, NOME_IDIOMA.get(dest, str(idioma)), n)
        if r_:
            r_["idioma"] = dest
            return r_
    ideia_en, aviso = ideia, ""
    if ori != "en" and traduzir_ok:
        t = traduzir_en(ideia)
        if t:
            ideia_en = t[0]
        else:
            aviso = "sem tradutor: usei o texto como esta (em ingles a imagem sai melhor)"
    r_ = cenas_offline(ideia, n, idioma=dest, ideia_en=ideia_en)
    if dest not in FALAS and traduzir_ok:                        # idioma sem falas prontas: traduz as de ingles
        for c in r_["cenas"]:
            c["narracao"] = _no_idioma(c["narracao"], dest, "en")
    if r_["cenas"] and r_["cenas"][0]["legenda"]:
        r_["cenas"][0]["legenda"] = _no_idioma(r_["cenas"][0]["legenda"], dest, ori) if traduzir_ok else r_["cenas"][0]["legenda"]
    if aviso:
        r_["aviso"] = aviso
    return r_


SYSTEM_PROMPT_IMG = (
    "You are a professional prompt engineer for a text-to-image model. The user's idea may be in ANY language and may "
    "contain typos: understand what they mean. Reply with ONLY ONE English prompt, no quotes, no explanations, "
    "max 45 words. Structure: main subject (very concrete, keep EXACTLY what the user asked), setting, composition/"
    "camera angle, lighting, mood, color palette. Do not invent extra subjects. No text, logos or names in the image.")
ULTIMA_FONTE = [""]


def _prompt_do_cerebro(texto, cerebro):
    """cerebro(sistema, usuario) -> texto (IA do app ou do backend). Devolve o prompt em ingles ou '' se nao serviu."""
    try:
        r = cerebro(SYSTEM_PROMPT_IMG, "Idea: %s" % texto[:300])
    except Exception:
        return ""
    r = re.sub(r"\s+", " ", str(r or "").splitlines()[0] if str(r or "").strip() else "").strip().strip("\"'` ")
    r = re.sub(r"^(prompt|english prompt)\s*:\s*", "", r, flags=re.I)
    if len(r) < 12 or len(r) > 600 or re.search(r"\b(sorry|cannot|as an ai|unable to)\b", r, re.I):
        return ""
    if sum(ch.isascii() for ch in r) < 0.9 * len(r) or detectar_idioma(r) in ("pt", "es", "fr", "de", "it"):
        return ""                                                 # veio em outro idioma: nao serve
    return " ".join(r.split(" ")[:60])


def prompt_imagem(texto, traduzir_ok=True, cerebro=None):
    """Para o botao GERAR IMAGEM: qualquer idioma -> prompt PROFISSIONAL em ingles com o estilo Hollywood.
    1) cerebro (IA) escreve o prompt; 2) senao traduz (internet); 3) senao usa o texto. Devolve (prompt, aviso).
    ULTIMA_FONTE[0] diz qual caminho valeu: 'IA', 'tradutor' ou 'texto'."""
    t = re.sub(r"\s+", " ", str(texto or "")).strip()
    if cerebro is not None:
        p = _prompt_do_cerebro(t, cerebro)
        if p:
            ULTIMA_FONTE[0] = "IA"
            return estilizar(p), ""
    cod, aviso = detectar_idioma(t), ""
    ULTIMA_FONTE[0] = "texto"
    if cod != "en" and traduzir_ok:
        tr = traduzir_en(t)
        if tr:
            t = tr[0]
            ULTIMA_FONTE[0] = "tradutor"
        else:
            aviso = "sem tradutor: usei o texto como esta (em ingles a imagem sai melhor)"
    return estilizar(_limpar_ideia(t, 40)), aviso
