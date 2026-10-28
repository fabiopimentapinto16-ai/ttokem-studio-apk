"""MaestroTtoken: gera video tentando, em cascata, (1) Python local no celular (Pydroid 3),
(2) Colab (GPU na nuvem, via link publico) e (3) APIs gratuitas (Agnes / Pexels / etc.).

Regras:
- nunca trava a interface: tudo roda em thread e fala com a tela por callbacks;
- cada tentativa vira uma mensagem no chat ("Tentando Python local...", "Colab offline...");
- cada usuario usa o SEU servidor: o link e a chave vem juntos (link#chave) do Colab ou do
  Pydroid dele e ficam em config_links.json, so no celular. Nao ha chave embutida no app;
- este modulo nao depende do Kivy: funciona e e testado fora do celular.
"""
import base64
import json
import os
import threading
import time

import requests

CHAVE_PADRAO = ""        # sem chave embutida: cada usuario usa a SUA (o servidor gera uma aleatoria)
ARQ_CONFIG = "config_links.json"
TEMPO_LOCAL = 120          # segundos esperando o Python local
TEMPO_COLAB = 180          # segundos esperando o Colab
TEMPO_PING = 6             # teste rapido "esta no ar?" antes do pedido longo


# ------------------------------------------------------------------ links salvos
def normalizar_link(texto, local=False):
    """'abc.trycloudflare.com/' -> 'https://abc.trycloudflare.com'. Local sem esquema usa http."""
    t = (texto or "").strip().strip('"\'')
    if not t:
        return ""
    if "://" not in t:
        t = ("http://" if local or t.startswith(("127.", "localhost", "192.168.")) else "https://") + t
    return t.rstrip("/")


def separar_link_chave(texto, local=False):
    """'https://x.trycloudflare.com#abc123' -> ('https://x.trycloudflare.com', 'abc123').
    O servidor imprime link e chave juntos para o usuario colar uma vez so."""
    t = (texto or "").strip()
    base, _, chave = t.partition("#")
    return normalizar_link(base, local=local), chave.strip()


def carregar_links(pasta):
    """Le config_links.json; sempre devolve as 3 chaves (colab, local, chave)."""
    dados = {}
    try:
        with open(os.path.join(pasta or ".", ARQ_CONFIG), encoding="utf-8") as f:
            dados = json.load(f)
    except Exception:
        dados = {}
    return {"colab": normalizar_link(dados.get("colab", "")),
            "local": normalizar_link(dados.get("local", ""), local=True),
            "chave": (dados.get("chave") or CHAVE_PADRAO).strip()}


def salvar_links(pasta, colab="", local="", chave=""):
    """Grava config_links.json (escrita atomica). Devolve o dicionario salvo."""
    colab, c1 = separar_link_chave(colab)
    local, c2 = separar_link_chave(local, local=True)
    dados = {"colab": colab, "local": local,
             "chave": (chave or c1 or c2 or CHAVE_PADRAO).strip()}
    os.makedirs(pasta or ".", exist_ok=True)
    destino = os.path.join(pasta or ".", ARQ_CONFIG)
    tmp = destino + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    os.replace(tmp, destino)
    return dados


# ------------------------------------------------------------------ o Maestro
class MaestroTtoken:
    def __init__(self, link_colab="", link_python_local="", chave_api=CHAVE_PADRAO,
                 pasta=None, fallback_apis=None, http=None):
        self.link_colab = normalizar_link(link_colab)
        self.link_python_local = normalizar_link(link_python_local, local=True)
        self.chave_api = chave_api or CHAVE_PADRAO
        self.pasta = pasta or "."
        self.fallback_apis = fallback_apis       # f(prompt, callback_chat) -> dict | None
        self.http = http or requests
        self.ultimo = {}

    # --- utilidades
    def _headers(self):
        return {"x-api-key": self.chave_api} if self.chave_api else {}

    def _no_ar(self, base):
        """Teste rapido: False so quando nem conecta (offline). Sem /saude (404) segue em frente."""
        try:
            r = self.http.get(base + "/saude", headers=self._headers(), timeout=TEMPO_PING)
            return r.status_code < 500
        except Exception:
            return False

    def _guardar(self, dados):
        """Transforma a resposta do servidor em arquivo/URL. Devolve dict ou None."""
        if not isinstance(dados, dict):
            return None
        if dados.get("status") not in (None, "sucesso", "ok", "pronto"):
            return None
        pasta = self.pasta
        os.makedirs(pasta, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        for campo, ext in (("gif", "gif"), ("mp4", "mp4"), ("video", "mp4"), ("png", "png")):
            b64 = dados.get(campo)
            if b64 and not str(b64).startswith(("http://", "https://")):
                caminho = os.path.join(pasta, "ttoken_%s.%s" % (ts, ext))
                with open(caminho, "wb") as f:
                    f.write(base64.b64decode(b64))
                return {"arquivo": caminho, "url": "", "info": dados}
        url = dados.get("video_url") or dados.get("url") or ""
        if url:
            return {"arquivo": "", "url": url, "info": dados}
        return None

    def _tentar(self, base, prompt, tempo, avisar, nome):
        """Uma tentativa em um servidor. Devolve resultado ou None (ja avisou o motivo)."""
        if not self._no_ar(base):
            avisar("⚠️ %s offline." % nome)
            return None
        try:
            r = self.http.post(base + "/gerar", json={"prompt": prompt},
                               headers=self._headers(), timeout=tempo)
        except Exception as e:
            avisar("⚠️ %s nao respondeu (%s)." % (nome, type(e).__name__))
            return None
        if r.status_code == 401:
            avisar("🔑 %s recusou a chave. Confira a chave no cadastro." % nome)
            return None
        if r.status_code != 200:
            avisar("⚠️ %s erro HTTP %d." % (nome, r.status_code))
            return None
        try:
            res = self._guardar(r.json())
        except Exception:
            res = None
        if not res:
            avisar("⚠️ %s respondeu sem video." % nome)
        return res

    # --- cascata
    def gerar_video(self, prompt, callback_chat):
        """Tenta Local -> Colab -> APIs. Bloqueante: chame em thread (use gerar_em_thread)."""
        prompt = (prompt or "").strip()
        if not prompt:
            callback_chat("✍️ Escreva o que voce quer ver no video.")
            return None
        etapas = (("local", self.link_python_local, TEMPO_LOCAL, "⚙️", "Python local"),
                  ("colab", self.link_colab, TEMPO_COLAB, "☁️", "Colab (GPU)"))
        for chave, base, tempo, icone, nome in etapas:
            if not base:
                continue
            callback_chat("%s Tentando gerar no %s..." % (icone, nome))
            res = self._tentar(base, prompt, tempo, callback_chat, nome)
            if res:
                res["origem"] = chave
                callback_chat("✅ Video gerado no %s!" % nome)
                self.ultimo = res
                return res
        callback_chat("🔄 Tentando APIs gratuitas...")
        if self.fallback_apis:
            try:
                res = self.fallback_apis(prompt, callback_chat)
            except Exception as e:
                callback_chat("⚠️ APIs gratuitas falharam (%s)." % str(e)[:60])
                res = None
            if res:
                res = dict(res)
                res.setdefault("arquivo", "")
                res.setdefault("url", "")
                res["origem"] = "apis"
                callback_chat("✅ Video gerado pelas APIs gratuitas!")
                self.ultimo = res
                return res
        callback_chat("❌ Todas as tentativas falharam.")
        return None

    def gerar_em_thread(self, prompt, callback_chat, ao_terminar, ui=None):
        """Roda a cascata em outra thread. `ui(f, *args)` agenda na thread da tela (no app:
        Clock.schedule_once); sem `ui`, chama direto (testes)."""
        agenda = ui or (lambda f, *a: f(*a))

        def msg(texto):
            agenda(callback_chat, texto)

        def trabalho():
            try:
                res = self.gerar_video(prompt, msg)
            except Exception as e:                      # nunca deixa a thread morrer calada
                msg("❌ Erro inesperado: %s" % str(e)[:80])
                res = None
            agenda(ao_terminar, res)

        t = threading.Thread(target=trabalho, daemon=True)
        t.start()
        return t
