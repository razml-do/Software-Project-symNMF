from setuptools import setup, Extension

ext = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"],
)

setup(
    name="symnmfmodule",
    ext_modules=[ext],
)