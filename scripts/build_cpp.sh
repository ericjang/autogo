#!/bin/bash
# Build C++ pybind11 extension and install to venv site-packages.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/src/alpha_go/cpp/build"
# Use UV_PROJECT_ENVIRONMENT if set, otherwise fall back to uv python find
if [ -n "$UV_PROJECT_ENVIRONMENT" ] && [ -f "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
    VENV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
else
    VENV_PYTHON="$(uv python find)"
fi

# Detect Python root (FindPython3 + pybind11 auto-detect the rest)
PYTHON_ROOT=$($VENV_PYTHON -c "import sys; print(sys.base_prefix)")

# Build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$VENV_PYTHON" \
    -DPython3_ROOT_DIR="$PYTHON_ROOT" \
    -DFETCHCONTENT_BASE_DIR="/tmp/cmake-fetchcontent"
cmake --build . -j$(nproc)

# Install .so to site-packages
SO_FILE=$(find . -name "alpha_go_cpp*.so" | head -1)
SITE_PACKAGES=$($VENV_PYTHON -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cp "$SO_FILE" "$SITE_PACKAGES/"
ln -sf "$(basename "$SO_FILE")" "$SITE_PACKAGES/alpha_go_cpp.so" 2>/dev/null || true

echo "=== C++ extension built: $SO_FILE -> $SITE_PACKAGES ==="
