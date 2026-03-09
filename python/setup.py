import re
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

# Read version from __version__.py without importing the package (which
# would require _core to be built already).
_version_file = ROOT / "flexaidds" / "__version__.py"
_version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']',
                           _version_file.read_text(), re.MULTILINE)
_version = _version_match.group(1) if _version_match else "0.0.0"

ext_modules = [
    Extension(
        "flexaidds._core",
        sources=[
            "flexaidds/_core.cpp",
            f"{_rel_lib}/statmech.cpp",
            f"{_rel_lib}/encom.cpp",
        ],
        include_dirs=[
            str(LIB_DIR),
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["-std=c++20", "-O3"],
    ),
]

setup(
    name="flexaidds",
    version=_version,
    description="Python bindings for the FlexAID∆S thermodynamic core",
    author="Louis-Philippe Morency",
    packages=["flexaidds"],
    package_dir={"": "."},
    ext_modules=ext_modules,
)
