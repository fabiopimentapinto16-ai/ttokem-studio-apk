from setuptools import setup, Extension
setup(name="ttgemm", version="1.0", ext_modules=[Extension("ttgemm", ["ttgemm.c"], extra_compile_args=["-O3", "-ffast-math"], extra_link_args=["-pthread"])])
