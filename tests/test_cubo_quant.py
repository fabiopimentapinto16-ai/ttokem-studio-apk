import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ferramentas"))
import cubo_quant as cq


def estado(seed=0):
    r = np.random.default_rng(seed)
    return {"conv": (r.standard_normal((320, 4, 3, 3)) * 0.05).astype(np.float32),   # o caso do erro
            "linear": (r.standard_normal((320, 320)) * 0.02).astype(np.float32),
            "impar": (r.standard_normal((37, 53)) * 0.1).astype(np.float32),
            "bias": r.standard_normal(320).astype(np.float32),
            "zeros": np.zeros((64, 64), np.float32)}


class T(unittest.TestCase):
    def test_o_erro_do_reshape_nao_acontece_mais(self):
        st = estado()
        d = tempfile.mkdtemp(); p = os.path.join(d, "m.ttq")
        cq.salvar(st, p)
        with cq.Leitor(p) as lt:
            for nome, original in st.items():
                rec = lt.ler(nome)
                self.assertEqual(rec.shape, original.shape, nome)

    def test_erro_por_valor_limitado_pela_escala(self):
        r = np.random.default_rng(1).standard_normal((40, 40)).astype(np.float32)
        pack, esc, R, C = cq.quantizar(r)
        rec = cq.desquantizar(pack, esc, R, C, r.shape, max_elementos=128)
        # cada valor erra no maximo meia escala do seu cubo (+ folga do float16)
        for i in range(0, 40, 8):
            for j in range(0, 40, 8):
                e = float(esc[i // 8, j // 8])
                self.assertLessEqual(np.abs(r[i:i+8, j:j+8] - rec[i:i+8, j:j+8]).max(),
                                     e * 0.5 * 1.01 + 1e-6)

    def test_qualidade_e_tamanho(self):
        st = estado()
        d = tempfile.mkdtemp(); p = os.path.join(d, "m.ttq")
        info = cq.salvar(st, p)
        self.assertGreater(info["economia_pct"], 80)            # ~1/8 do fp32
        with cq.Leitor(p) as lt:
            m = cq.erro(st["linear"], lt.ler("linear"))
        self.assertGreater(m["snr_db"], 15)                     # 4 bits: ~20 dB e normal
        self.assertTrue(np.all(cq.Leitor(p).ler("zeros") == 0))

    def test_tensor_pequeno_nao_perde(self):
        st = estado(); d = tempfile.mkdtemp(); p = os.path.join(d, "m.ttq"); cq.salvar(st, p)
        with cq.Leitor(p) as lt:
            np.testing.assert_allclose(lt.ler("bias"), st["bias"], atol=1e-2, rtol=1e-3)

    def test_arquivo_invalido(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "x.ttq")
        open(p, "wb").write(b"lixo lixo lixo lixo")
        with self.assertRaises(ValueError):
            cq.Leitor(p)

    def test_8_bits_e_muito_melhor_e_parte_a_parte_igual(self):
        r = np.random.default_rng(5).standard_normal((100, 70)).astype(np.float32)
        out = {}
        for bits in (4, 8):
            pk, es, R, C = cq.quantizar(r, (8, 8), bits)
            inteira = cq.desquantizar(pk, es, R, C, r.shape, bits=bits)
            partes = cq.desquantizar(pk, es, R, C, r.shape, bits=bits, max_elementos=100)
            np.testing.assert_array_equal(inteira, partes)
            out[bits] = cq.erro(r, inteira)["snr_db"]
        self.assertGreater(out[8], out[4] + 15)

    def test_arquivo_guarda_os_bits(self):
        d = tempfile.mkdtemp(); p = os.path.join(d, "m.ttq")
        cq.salvar({"w": np.random.default_rng(1).standard_normal((32, 32)).astype(np.float32)}, p, bits=8)
        with cq.Leitor(p) as lt:
            self.assertEqual(lt.bits, 8)

    def test_reordenar_cubo_volta_exato_e_ajuda_com_canais_desiguais(self):
        r = np.random.default_rng(7)
        lin = np.exp(r.standard_normal((256, 1)) * 0.8); col = np.exp(r.standard_normal((1, 192)) * 0.8)
        w = (r.standard_normal((256, 192)) * 0.02 * lin * col).astype(np.float32)
        d = tempfile.mkdtemp(); snr = {}
        for ro in (False, True):
            p = os.path.join(d, "m%d.ttq" % ro); cq.salvar({"w": w}, p, reordenar=ro)
            with cq.Leitor(p) as lt:
                rec = lt.ler("w")
            self.assertEqual(rec.shape, w.shape)           # a ordem original e restaurada
            snr[ro] = cq.erro(w, rec)["snr_db"]
        self.assertGreater(snr[True], snr[False] + 1.5)
        # 8 bits + reordenar: praticamente sem perda, na ordem certa
        p = os.path.join(d, "m8.ttq"); cq.salvar({"w": w}, p, bits=8, reordenar=True)
        with cq.Leitor(p) as lt:
            self.assertGreater(cq.erro(w, lt.ler("w"))["snr_db"], 35)

    def test_blocos_diferentes(self):
        r = np.random.default_rng(2).standard_normal((64, 64)).astype(np.float32)
        d = tempfile.mkdtemp()
        for bloco in ((8, 8), (4, 16), (1, 32)):
            p = os.path.join(d, "m.ttq"); cq.salvar({"w": r}, p, bloco)
            with cq.Leitor(p) as lt:
                self.assertEqual(lt.ler("w").shape, r.shape)


if __name__ == "__main__":
    unittest.main()
