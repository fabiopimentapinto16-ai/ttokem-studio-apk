import os
aqui=os.path.dirname(os.path.abspath(__file__))
nomes=["colab_tudo_validar.py","colab_pipeline_validar.py","colab_bits_comparar.py"]
src={n:open(os.path.join(aqui,n),encoding='utf-8').read() for n in nomes}
cab='''# TTOKEN - FECHA TUDO NUMA CELULA SO (Colab, runtime CPU serve). ~20-30 min. Faz, em ordem:
#  1) exporta o tiny-sd em .ttq + valida UNet/CLIP/VAE em numpy contra o torch + salva o tokenizador (tok.zip)
#  2) valida o PIPELINE COMPLETO (tokenizador + agendador + imagem final) contra o diffusers
#  3) mede bits/peso de cada tecnica de compressao nos pesos reais
# Se uma etapa falhar, as outras continuam. Me mande TODO o texto no fim.
import traceback
'''
corpo="_ETAPAS = [\\n" if False else ""
corpo="_ETAPAS = [\n"+"".join("    (%r, %r),\n"%(n,s) for n,s in src.items())+"]\n"
corpo+='''
for _nome, _codigo in _ETAPAS:
    print("\\n" + "#" * 70 + "\\n# ETAPA: " + _nome + "\\n" + "#" * 70, flush=True)
    try:
        exec(compile(_codigo, _nome, "exec"), {"__name__": "__main__"})
    except SystemExit:
        pass
    except Exception:
        print("ETAPA FALHOU:", _nome); traceback.print_exc()
print("\\n" + "#" * 70 + "\\n# FIM. Me mande todo o texto acima.\\n" + "#" * 70)
'''
open(os.path.join(aqui,'colab_fechar_tudo.py'),'w',encoding='utf-8').write(cab+corpo)
