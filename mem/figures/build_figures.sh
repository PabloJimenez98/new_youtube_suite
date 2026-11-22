#!/usr/bin/env bash
# Pre-render every figure source (figures/<name>.tex) to figures/pdf/<name>.pdf,
# so the report compiles without running TikZ/pgfplots (fast enough for Overleaf).
#
#   cd mem && bash figures/build_figures.sh [name ...]      # default: all figures
#
# Needs a LaTeX engine: tectonic (TECTONIC=/path/to/tectonic) or pdflatex.
set -euo pipefail
cd "$(dirname "$0")/.."                      # the mem/ folder
mkdir -p figures/pdf
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
mkdir -p "$build/figures"
cp figures/*.tex "$build/figures/"

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  for f in figures/*.tex; do
    n=$(basename "$f" .tex)
    [ "$n" = fig-preamble ] || names+=("$n")
  done
fi

for n in "${names[@]}"; do
  cat > "$build/$n.tex" <<EOF
\\documentclass[12pt]{standalone}
\\input{figures/fig-preamble}
\\begin{document}
\\input{figures/$n}
\\end{document}
EOF
  if [ -n "${TECTONIC:-}" ] || command -v tectonic >/dev/null; then
    (cd "$build" && "${TECTONIC:-tectonic}" -X compile "$n.tex" >/dev/null 2>"$n.err" \
      || { cat "$n.err"; exit 1; })
  else
    (cd "$build" && pdflatex -interaction=nonstopmode -halt-on-error "$n.tex" >/dev/null)
  fi
  cp "$build/$n.pdf" "figures/pdf/$n.pdf"
  echo "figures/pdf/$n.pdf"
done
