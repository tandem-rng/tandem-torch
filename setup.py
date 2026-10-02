import os
import shutil
import subprocess
import sysconfig

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Sources must be /-separated paths relative to this file for the editable build. Include
# directories and objects must be absolute, since the compiler runs in a temporary directory.
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
pkg = "src/tandem_torch"
abs_pkg = os.path.join(here, pkg)


def tandem_object():
    """Compile the C99 reference with the C compiler. torch's extension builder passes C++
    flags to every source, which the C file rejects."""
    out = os.path.join(here, "build", "tandem.o")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    cmd = cc.split() + ["-std=c99", "-O2", "-fPIC", "-c", "-o", out, f"{pkg}/c/tandem.c"]
    subprocess.check_call(cmd)
    return out


with_cuda = os.environ.get("TANDEM_TORCH_CUDA", "auto")
if with_cuda == "auto":
    with_cuda = "1" if shutil.which("nvcc") or os.environ.get("CUDA_HOME") else "0"

sources = [f"{pkg}/_ext.cpp"]
kwargs = dict(
    include_dirs=[f"{abs_pkg}/c", f"{abs_pkg}/cuda"],
    extra_objects=[tandem_object()],
    extra_compile_args={"cxx": ["-O2"]},
)
if with_cuda == "1":
    sources.append(f"{pkg}/_cuda.cu")
    kwargs["define_macros"] = [("TANDEM_TORCH_CUDA", "1")]
    kwargs["extra_compile_args"]["nvcc"] = ["-O3"]
    ext = CUDAExtension("tandem_torch._ext", sources, **kwargs)
else:
    ext = CppExtension("tandem_torch._ext", sources, **kwargs)

setup(ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
