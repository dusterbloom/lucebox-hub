#!/bin/bash
set -euo pipefail
R=/home/duster/qwen4exp-exact-stack
C=$R/build1151/CMakeCache.txt
OUT=$R/build-ko-configure.sh
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo 'R=/home/duster/qwen4exp-exact-stack'
  echo 'S=$R/src/server'
  echo 'B=$R/build-ko'
  echo 'mkdir -p $B'
  printf 'cmake -S "$S" -B "$B" -G "Unix Makefiles" \\\n'
  grep -E '^(GGML_|LUCE_)[A-Za-z0-9_]+:(BOOL|STRING|PATH|FILEPATH)=' "$C" | while IFS= read -r line; do
    name=${line%%:*}
    rest=${line#*:}
    type=${rest%%=*}
    val=${rest#*=}
    [ -z "$val" ] && continue
    printf '  -D%s:%s=%s \\\n' "$name" "$type" "$val"
  done
  echo '  -DCMAKE_BUILD_TYPE=Release \'
  echo '  -DCMAKE_C_COMPILER=/usr/bin/cc \'
  echo '  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \'
  echo '  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.2/lib/llvm/bin/clang++ \'
  echo '  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \'
  echo '  -DCMAKE_INSTALL_PREFIX=/usr/local'
} > "$OUT"
chmod +x "$OUT"
wc -l "$OUT"
