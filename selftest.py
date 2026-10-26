"""Autoteste do Ttoken Studio ("TESTAR TUDO").

Roda cada funcao do app de verdade (rede, servidor, IAs, video, celular) e devolve uma
lista de resultados com a causa e o que fazer. Nao depende do Kivy: o app injeta o que
precisa em `ctx`, o que permite testar este modulo fora do Android.

ctx: api, ua, acordar(sessao)->True/False/None, ultimo_erro()->str, llm, videos,
     tem_chaves (bool), chaves (dict), autoclass (ou None), android (bool), user_dir,
     versao_app (str)
log(nome, status, detalhe, acao)   status: OK | ERRO | AVISO | PULOU
"""
import os
import time

import requests

import ttokem

ORDEM = {"ERRO": 0, "AVISO": 1, "PULOU": 2, "OK": 3}


def _json(r):
    try:
        return r.json()
    except Exception:
        return {}


def _detalhe(r):
    d = _json(r)
    return str(d.get("detail") or d.get("erro") or r.text[:120] or r.status_code)[:160]


def rodar(ctx, log):
    res = []

    def reg(nome, status, detalhe="", acao=""):
        res.append({"nome": nome, "status": status, "detalhe": detalhe, "acao": acao})
        try:
            log(nome, status, detalhe, acao)
        except Exception:
            pass

    def seguro(nome, fn):
        try:
            fn()
        except Exception as e:                              # um teste nunca derruba os outros
            reg(nome, "ERRO", "%s: %s" % (type(e).__name__, str(e)[:140]),
                "Mande este texto para o Claude corrigir.")

    api = ctx["api"]
    s = requests.Session()
    s.headers.update({"User-Agent": ctx["ua"]})
    estado = {"servidor": False}

    # 1) compressao local (nao depende de rede)
    def t_ttokem():
        txt = ("Quero um video curto mostrando um indio correndo na floresta ao amanhecer, "
               "com musica epica e cortes rapidos, para postar no TikTok.")
        r = ttokem.compress(txt, "auto")
        ok = isinstance(r, dict) and r.get("text")
        reg("Compressor Ttokem (local)", "OK" if ok else "ERRO",
            "" if ok else "resposta inesperada", "" if ok else "Bug no compressor.")
    seguro("Compressor Ttokem (local)", t_ttokem)

    # 2) internet
    def t_net():
        r = s.get("https://api.github.com", timeout=12)
        reg("Internet do celular", "OK" if r.status_code < 500 else "ERRO",
            "HTTP %d" % r.status_code, "")
    seguro("Internet do celular", t_net)

    # 3) chaves locais (so se o usuario colou)
    def t_chaves():
        if not ctx["tem_chaves"]:
            reg("IAs com as chaves do celular", "PULOU",
                "Nenhuma chave de texto colada.", "Opcional: cole uma chave Groq/Gemini.")
            return
        for r in ctx["llm"].testar():
            reg("IA " + r["provedor"] + " (sua chave)", "OK" if r["ok"] else "ERRO",
                "" if r["ok"] else r["erro"][:160],
                "" if r["ok"] else "Confira a chave/credito no site do provedor.")
    seguro("IAs com as chaves do celular", t_chaves)

    # 4) servidor
    def t_servidor():
        t0 = time.time()
        r = ctx["acordar"](s)
        seg = int(time.time() - t0)
        if r is True:
            estado["servidor"] = True
            h = _json(s.get(api + "/health", timeout=20))
            reg("Servidor (SnapDeploy)", "OK",
                "no ar em %ds, versao %s" % (seg, h.get("versao", "?")), "")
            ias = h.get("ias") or []
            n = len([x for x in ias if x.get("disponivel", True)])
            reg("IAs de texto no servidor", "OK" if ias else "ERRO",
                "%d slot(s)" % len(ias),
                "" if ias else "Crie GROQ_API_KEY ou GEMINI_API_KEY no SnapDeploy.")
            vids = h.get("videos") or []
            tipos = sorted({v.get("tipo") or v.get("id", "?") for v in vids}) if vids else []
            reg("Geradores de video no servidor", "OK" if tipos else "AVISO",
                ", ".join(tipos) or "nenhum",
                "" if tipos else "Sem video de IA: crie PEXELS_API_KEY (plano B) ou chave Magic Hour/fal.")
        elif r is None:
            reg("Servidor (SnapDeploy)", "ERRO", "no ar, mas sem chave de IA",
                "Crie GROQ_API_KEY ou GEMINI_API_KEY no SnapDeploy.")
        else:
            reg("Servidor (SnapDeploy)", "ERRO", "sem resposta em %ds (%s)" % (
                seg, ctx["ultimo_erro"]() or "?"),
                "Abra o link do servidor no navegador, espere acordar e teste de novo.")
    seguro("Servidor (SnapDeploy)", t_servidor)

    if estado["servidor"]:
        # 5) cada IA do servidor, de verdade
        def t_ias():
            r = s.get(api + "/api/testar", timeout=120)
            if r.status_code != 200:
                reg("Teste das IAs do servidor", "ERRO", "HTTP %d: %s" % (r.status_code, _detalhe(r)), "")
                return
            for x in _json(r).get("textos", []):
                reg("IA " + x["provedor"] + " (servidor)", "OK" if x["ok"] else "ERRO",
                    "" if x["ok"] else x["erro"][:160],
                    "" if x["ok"] else "Troque a chave/modelo ou espere o limite voltar.")
        seguro("Teste das IAs do servidor", t_ias)

        # 6) chat
        def t_chat():
            r = s.post(api + "/api/chat", json={"mensagem": "Responda somente: ok"}, timeout=90)
            d = _json(r)
            ok = r.status_code == 200 and d.get("resposta")
            reg("Chat de IA", "OK" if ok else "ERRO",
                ("respondeu via %s" % d.get("ia", "?")) if ok else "HTTP %d: %s" % (r.status_code, _detalhe(r)),
                "" if ok else "Nenhuma IA de texto respondeu: veja os testes de IA acima.")
        seguro("Chat de IA", t_chat)

        # 7) legenda de ponta a ponta (job assincrono, sem gastar video)
        def t_legenda():
            r = s.post(api + "/api/gerar", json={"ideia": "teste rapido: gato dormindo",
                                                  "plataforma": "tiktok", "sem_video": True},
                       timeout=60)
            if r.status_code != 200:
                reg("Legenda (gerar de ponta a ponta)", "ERRO",
                    "HTTP %d: %s" % (r.status_code, _detalhe(r)), "")
                return
            jid = _json(r).get("job_id")
            fim = time.time() + 120
            d = {}
            while time.time() < fim:
                time.sleep(3)
                d = _json(s.get(api + "/api/status/" + str(jid), timeout=30))
                if d.get("status") in ("pronto", "erro"):
                    break
            ok = d.get("status") == "pronto" and (d.get("texto") or d.get("titulo"))
            reg("Legenda (gerar de ponta a ponta)", "OK" if ok else "ERRO",
                ("legenda via %s" % d.get("api_llm", "IA")) if ok else
                str(d.get("erro") or d.get("status") or "sem resposta")[:160],
                "" if ok else "Nenhuma IA de texto respondeu a tempo.")
        seguro("Legenda (gerar de ponta a ponta)", t_legenda)

        # 8) tendencias
        def t_trend():
            r = s.get(api + "/api/tendencias", timeout=40)
            if r.status_code == 200:
                reg("Mais assistidos (aba TOP)", "OK", "%d videos" % len(_json(r).get("itens", [])), "")
            elif r.status_code == 503 and not ctx["chaves"].get("youtube"):
                reg("Mais assistidos (aba TOP)", "AVISO", "sem chave do YouTube",
                    "Crie YOUTUBE_API_KEY no SnapDeploy ou cole a chave YouTube no app.")
            else:
                reg("Mais assistidos (aba TOP)", "ERRO", "HTTP %d: %s" % (r.status_code, _detalhe(r)), "")
        seguro("Mais assistidos (aba TOP)", t_trend)

        # 9) compilar pelo Git
        def t_build():
            senha = (ctx["chaves"].get("build") or "").strip()
            r = s.get(api + "/api/build/status/1", headers={"X-Build-Secret": senha}, timeout=30)
            if r.status_code == 503:
                reg("Compilar pelo Git", "AVISO", "servidor sem BUILD_SECRET/GITHUB_TOKEN",
                    "Crie BUILD_SECRET e GITHUB_TOKEN no SnapDeploy.")
            elif r.status_code == 403:
                reg("Compilar pelo Git", "ERRO" if senha else "AVISO",
                    "senha de build %s" % ("incorreta" if senha else "nao preenchida no app"),
                    "Cole no app a mesma senha do BUILD_SECRET.")
            else:   # 200 ou erro do GitHub para o run 1 = autenticacao passou
                reg("Compilar pelo Git", "OK", "senha aceita (HTTP %d)" % r.status_code, "")
        seguro("Compilar pelo Git", t_build)

        # 10) Colab
        def t_colab():
            h = _json(s.get(api + "/health", timeout=20))
            tem = any((v.get("tipo") or v.get("id")) == "colab" for v in (h.get("videos") or []))
            cfg = h.get("config") or {}
            if not tem and cfg.get("COLAB_SECRET") is False:
                reg("Colab (GPU de video)", "AVISO", "COLAB_SECRET nao existe no SnapDeploy",
                    "Crie a variavel COLAB_SECRET (uma senha sua) e use a MESMA no notebook do Colab.")
                return
            reg("Colab (GPU de video)", "OK" if tem else "AVISO",
                "conectado" if tem else "nenhum notebook conectado agora",
                "" if tem else "COLAB_SECRET existe no servidor: rode o notebook do Colab "
                               "com a mesma senha.")
        seguro("Colab (GPU de video)", t_colab)
    else:
        for nome in ("Chat de IA", "Legenda (gerar de ponta a ponta)", "Mais assistidos (aba TOP)",
                     "Compilar pelo Git", "Colab (GPU de video)"):
            reg(nome, "PULOU", "servidor fora do ar", "Resolva o servidor primeiro.")

    # 11) celular
    ac = ctx["autoclass"]
    if not ctx["android"] or ac is None:
        reg("Funcoes do celular (voz, notificacao, Compartilhar)", "PULOU",
            "nao esta no Android", "")
    else:
        def t_perm():
            act = ac("org.kivy.android.PythonActivity").mActivity
            pm = ac("android.content.pm.PackageManager")
            for perm, nome in (("android.permission.RECORD_AUDIO", "Microfone (voz no chat)"),
                               ("android.permission.POST_NOTIFICATIONS", "Notificacoes")):
                ok = act.checkSelfPermission(perm) == pm.PERMISSION_GRANTED
                reg(nome, "OK" if ok else "AVISO", "permitido" if ok else "nao permitido",
                    "" if ok else "Configuracoes do Android > Apps > Ttoken Studio > Permissoes.")
        seguro("Permissoes", t_perm)

        def t_share():
            act = ac("org.kivy.android.PythonActivity").mActivity
            Intent = ac("android.content.Intent")
            i = Intent(Intent.ACTION_SEND)
            i.setType("video/mp4")
            lst = act.getPackageManager().queryIntentActivities(i, 0)
            meu = act.getPackageName()
            ok = any(lst.get(n).activityInfo.packageName == meu for n in range(lst.size()))
            reg("Aparece no menu Compartilhar", "OK" if ok else "ERRO",
                "registrado para video" if ok else "nao registrado",
                "" if ok else "Bug no manifest: mande este resultado ao Claude.")
        seguro("Aparece no menu Compartilhar", t_share)

        def t_arquivos():
            p = os.path.join(ctx["user_dir"], "teste.tmp")
            with open(p, "w") as f:
                f.write("ok")
            os.remove(p)
            reg("Gravar arquivos do app", "OK", "", "")
        seguro("Gravar arquivos do app", t_arquivos)

        def t_notif():
            ctx["notificar_teste"]()
            reg("Notificacao de teste", "OK", "enviada: veja a barra do Android",
                "Se nao apareceu, ative as notificacoes do app.")
        seguro("Notificacao de teste", t_notif)

    res.sort(key=lambda r: ORDEM.get(r["status"], 9))
    return res


def relatorio(res, versao=""):
    n = {k: sum(1 for r in res if r["status"] == k) for k in ORDEM}
    linhas = ["TTOKEN STUDIO %s - AUTOTESTE" % versao,
              "OK %d | ERRO %d | AVISO %d | PULOU %d" % (n["OK"], n["ERRO"], n["AVISO"], n["PULOU"]), ""]
    for r in res:
        linhas.append("[%s] %s%s" % (r["status"], r["nome"], (": " + r["detalhe"]) if r["detalhe"] else ""))
        if r["acao"] and r["status"] in ("ERRO", "AVISO"):
            linhas.append("      -> " + r["acao"])
    return "\n".join(linhas)
