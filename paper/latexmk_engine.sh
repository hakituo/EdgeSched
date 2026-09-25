#!/usr/bin/env bash
set -e
src="${@: -1}"
if [[ "$src" == *"main_zh_full.tex" ]]; then
  exec xelatex "$@"
else
  exec pdflatex "$@"
fi