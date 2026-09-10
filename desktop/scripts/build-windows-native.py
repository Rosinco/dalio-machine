"""Cross-build with LLVM and an SDK extracted using native msiextract.

Fallback for cargo-xwin's slow per-file CAB decompression. Same official Microsoft
SDK and CRT downloads, with Rust targeting x86_64-pc-windows-msvc.
"""
import json
import os
import subprocess
from pathlib import Path

tools = Path(os.environ.get("ATLAS_BUILD_TOOLS", "/tmp/atlas-tools"))
llvm = tools / "llvm/usr/lib/llvm-18/bin"
sdk = tools / "windows-sdk/Program Files/Windows Kits/10"
crt = tools / "manual-crt"
sdk_version = os.environ.get("ATLAS_SDK_VERSION", "10.0.26100.0")
ucrt_version = os.environ.get("ATLAS_UCRT_VERSION", "10.0.10240.0")
includes = [crt / "include", sdk / "Include" / ucrt_version / "ucrt"] + [
    sdk / "Include" / sdk_version / n for n in ("shared", "um", "winrt")
]
libraries = [
    crt / "lib/x86_64",
    sdk / "Lib" / ucrt_version / "ucrt/x64",
    sdk / "Lib" / sdk_version / "um/x64",
]
for directory in [llvm, *includes, *libraries]:
    if not directory.is_dir():
        raise SystemExit(f"Build dependency is missing: {directory}")
env = dict(os.environ)
env["PATH"] = f"{llvm}:{Path.home() / '.cargo/bin'}:{env['PATH']}"
env["LD_LIBRARY_PATH"] = str(tools / "llvm/usr/lib/x86_64-linux-gnu")
env["CARGO_BUILD_JOBS"] = "4"
env["CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER"] = str(llvm / "lld-link")
env["CC_x86_64_pc_windows_msvc"] = str(llvm / "clang-cl")
env["CXX_x86_64_pc_windows_msvc"] = str(llvm / "clang-cl")
env["RC"] = str(llvm / "llvm-rc")
env["INCLUDE"] = ";".join(map(str, includes))
env["CC_SHELL_ESCAPED_FLAGS"] = "1"
# Windows SDK includes use case-insensitive names (for example DriverSpecs.h).
# LLVM's virtual filesystem supplies those semantics on Linux without altering
# the extracted Microsoft headers. SQLite is compiled into the portable binary.
def virtual_directory(path):
    return {"type": "directory", "name": str(path), "contents": [
        virtual_directory(child) if child.is_dir() else
        {"type": "file", "name": child.name, "external-contents": str(child)}
        for child in sorted(path.iterdir())
    ]}

overlay = tools / "atlas-sdk-case-overlay.json"
overlay.write_text(json.dumps({"version": 0, "case-sensitive": False,
                              "roots": [virtual_directory(p) for p in includes]}))
env["CFLAGS_x86_64_pc_windows_msvc"] = " ".join(f'/imsvc"{p}"' for p in includes) + f' /clang:-ivfsoverlay /clang:"{overlay}"'
env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(
    [f"-Lnative={p}" for p in libraries] + ["-Ctarget-feature=+crt-static"]
)
subprocess.run(
    [str(Path.home() / ".cargo/bin/cargo"), "build", "--release", "--target", "x86_64-pc-windows-msvc"],
    cwd=Path(__file__).resolve().parents[1] / "src-tauri", env=env, check=True,
)
