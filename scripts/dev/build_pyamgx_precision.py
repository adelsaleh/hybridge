"""Build the qualified PyAMGX binding with mode-aware real array types.

The extension is built into an explicit local directory, never installed into
site-packages. The source checkout remains unchanged. Pass the output directory
on PYTHONPATH when running the FP32 benchmark.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def build(source: Path, output: Path, amgx_source: Path, amgx_build: Path) -> None:
    """Copy and patch the qualified source, then build its Cython extension."""
    source, output = source.resolve(), output.resolve()
    staging = output / "source"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "setup.py", staging / "setup.py")
    shutil.copytree(source / "pyamgx", staging / "pyamgx", dirs_exist_ok=True)
    for name, handle, position in (("Matrix", "mtx", 2), ("Vector", "vec", 1)):
        path = staging / "pyamgx" / f"{name}.pyx"
        text = path.read_text()
        declaration = f"    cdef AMGX_{name.lower()}_handle {handle}"
        if declaration not in text:
            raise RuntimeError(f"Unexpected {name} layout in {source}")
        text = text.replace(declaration, declaration + "\n    cdef public object dtype")
        create = f"        check_error(AMGX_{name.lower()}_create"
        text = text.replace(create,
            f"        self.dtype = {{'D': np.dtype('float64'), 'F': np.dtype('float32')}}[mode[{position}]]\n" + create)
        text = text.replace('ptr_from_array_interface(data, "float64")',
                            'ptr_from_array_interface(data, self.dtype)')
        text = text.replace('check_for_dtype="float64"', 'check_for_dtype=self.dtype')
        text = '\n'.join(
            line if line.lstrip().startswith(('>>>', '...')) else line.replace('dtype=np.float64)', 'dtype=self.dtype)')
            for line in text.split('\n')
        )
        if name == "Vector":
            text = text.replace('n = data.size/block_dim', 'n = data.size//block_dim')
            text = text.replace('def download(self, double[:] data=None):', 'def download(self, data=None):')
            text = text.replace('n = self.get_size()[0]', 'size, block_dim = self.get_size()\n            n = size * block_dim')
            text = text.replace('self.download_raw(<uintptr_t> &data[0])',
                'size, block_dim = self.get_size()\n        if data.ndim != 1 or data.size != size * block_dim or not data.flags.c_contiguous:\n            raise ValueError("download buffer must be contiguous and match the vector size")\n        if hasattr(data.flags, "writeable") and not data.flags.writeable:\n            raise ValueError("download buffer must be writeable")\n        cdef uintptr_t ptr = ptr_from_array_interface(data, self.dtype)\n        self.download_raw(ptr)')
        path.write_text(text)
    module = staging / "pyamgx" / "pyamgx.pyx"
    with module.open("a") as stream:
        stream.write('\nHDGFEM_PRECISION_AWARE = True\n')
    env = dict(os.environ, AMGX_DIR=str(amgx_source.resolve()), AMGX_BUILD_DIR=str(amgx_build.resolve()))
    subprocess.run([sys.executable, "setup.py", "build_ext", "--build-lib", str(output),
                    "--build-temp", str(output / "objects")], cwd=staging, env=env, check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    (output / "source-revision.txt").write_text(revision + "\nMode-aware Matrix/Vector dtype patch; see scripts/dev/build_pyamgx_precision.py\n")
    print(f"Built local precision-aware PyAMGX in {output}")


def main() -> None:
    """Parse explicit source/build paths for the optional local extension."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amgx-source", type=Path, required=True)
    parser.add_argument("--amgx-build", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output, args.amgx_source, args.amgx_build)


if __name__ == "__main__":
    main()
