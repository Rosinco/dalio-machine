#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../src-tauri"
# The resulting executable has no dependency on these build tools or on WSL.
export PATH="${ATLAS_LLVM_BIN:-/tmp/atlas-tools/llvm/usr/lib/llvm-18/bin}:/tmp/atlas-tools/python/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="${ATLAS_LLVM_LIB:-/tmp/atlas-tools/llvm/usr/lib/x86_64-linux-gnu}:${LD_LIBRARY_PATH:-}"
export XWIN_ARCH=x86_64
export XWIN_CACHE_DIR="${ATLAS_XWIN_CACHE:-/tmp/atlas-tools/xwin}"
export CARGO_BUILD_JOBS=4
cargo xwin build --release --target x86_64-pc-windows-msvc
