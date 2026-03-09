from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:
    raise SystemExit(
        "pybind11 is required to build flexaidds. "
        "Install it with `pip install pybind11` and retry."
    ) from exc

ROOT = Path(__file__).resolve().parent
LIB_DIR = ROOT.parent / "LIB"

ext_modules = [
    Extension(
        "flexaidds._core",
        sources=[
            str(ROOT / "flexaidds" / "_core.cpp"),
            str(LIB_DIR / "statmech.cpp"),
        ],
        include_dirs=[
            str(LIB_DIR),
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="flexaidds",
    version="0.1.0",
    description="Python bindings for the FlexAID∆S thermodynamic core",
    author="Louis-Philippe Morency",
    packages=["flexaidds"],
    package_dir={"": "."},
    ext_modules=ext_modules,
)
