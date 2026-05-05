#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

build_type="${BUILD_TYPE:-Debug}"
build_dir="${BUILD_DIR:-${project_root}/build/${build_type}}"

format() {
  find "${project_root}/include" "${project_root}/src" "${project_root}/tests" \
    \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) \
    -print0 | xargs -0 clang-format -i
}

configure() {
  cmake -S "${project_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE="${build_type}"
}

build() {
  configure
  cmake --build "${build_dir}"
}

test() {
  build
  ctest --test-dir "${build_dir}" --output-on-failure
}

all() {
  format
  test
}

command="${1:-all}"

case "${command}" in
  format)
    format
    ;;
  build)
    format
    build
    ;;
  test)
    format
    test
    ;;
  clean)
    rm -rf "${build_dir}"
    ;;
  all)
    all
    ;;
  *)
    echo "usage: tools/make.sh [format|build|test|clean|all]" >&2
    exit 2
    ;;
esac
