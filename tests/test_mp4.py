import math, os, struct, sys, tempfile, unittest, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import av
except Exception:
    av = None


@unittest.skipUnless(av, "PyAV nao instalado (roda no CI)")
class TMp4(unittest.TestCase):
    def test_mp4_com_audio(self):
        from PIL import Image
        import video_mp4
        with tempfile.TemporaryDirectory() as d:
            imgs = []
            for i, cor in enumerate(((200, 40, 40), (40, 200, 40), (40, 40, 200))):
                p = os.path.join(d, "c%d.png" % i)
                Image.new("RGB", (256, 256), cor).save(p)
                imgs.append(p)
            w = os.path.join(d, "n.wav")
            with wave.open(w, "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
                f.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 8.0))) for i in range(22050 * 3)))
            out, dur = video_mp4.montar_mp4(imgs, ["um", "dois", "tres"], os.path.join(d, "v.mp4"), [1.0, 1.0, 1.0], wav=w, fps=15, lado=256)
            self.assertTrue(os.path.getsize(out) > 2000)
            self.assertTrue(os.path.exists(os.path.join(d, "v_capa.png")))
            c = av.open(out)
            tipos = {s.type: s.codec_context.name for s in c.streams}
            self.assertEqual(tipos.get("video"), "h264")
            self.assertEqual(tipos.get("audio"), "aac")
            self.assertAlmostEqual(c.duration / 1e6, 3.0, delta=0.6)
            n = sum(1 for _ in c.decode(video=0))
            self.assertGreaterEqual(n, 40)
            c.close()

    def test_mp4_sem_audio(self):
        from PIL import Image
        import video_mp4
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.png"); Image.new("RGB", (256, 256), (9, 9, 9)).save(p)
            out, _ = video_mp4.montar_mp4([p, p], ["a", "b"], os.path.join(d, "v.mp4"), [1.0, 1.0], fps=15, lado=256)
            c = av.open(out)
            self.assertEqual([s.type for s in c.streams], ["video"])
            c.close()


if __name__ == "__main__":
    unittest.main()
