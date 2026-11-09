from pythonforandroid.recipe import IncludedFilesBehaviour, CompiledComponentsPythonRecipe


class TtgemmRecipe(IncludedFilesBehaviour, CompiledComponentsPythonRecipe):
    """Matmul float32 multi-thread propria (o numpy do APK vem sem BLAS e roda ~100x mais lento)."""
    name = "ttgemm"
    version = "1.0"
    url = None
    src_filename = "src"
    depends = ["setuptools"]
    call_hostpython_via_targetpython = False


recipe = TtgemmRecipe()
