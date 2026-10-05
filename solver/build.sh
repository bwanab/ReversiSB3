#!/bin/sh
# Build the endgame solver library next to this script: libendgame.dylib (macOS) or libendgame.so.
# Plain sh + cc so it works on macOS and Linux alike.
set -e
cd "$(dirname "$0")"
case "$(uname)" in
  Darwin) out=libendgame.dylib; flags="-dynamiclib" ;;
  *)      out=libendgame.so;    flags="-shared -fPIC" ;;
esac
build() { ${CC:-cc} -O3 -march=native $flags -o "$out" endgame.c; }
if ! build 2>/dev/null; then
  # macOS: if the Command Line Tools linker is older than the default SDK (as with the Edax build,
  # see CLAUDE.md), try the installed SDKs from newest to oldest.
  [ "$(uname)" = Darwin ] || { build; exit 1; }
  ok=""
  for sdk in $(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX[0-9]*.sdk 2>/dev/null | sort -rV); do
    if SDKROOT=$sdk build 2>/dev/null; then echo "linked with SDKROOT=$sdk"; ok=1; break; fi
  done
  [ -n "$ok" ] || { echo "no installed SDK links; see the error below"; build; exit 1; }
fi
echo "built solver/$out"
