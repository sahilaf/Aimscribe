#!/bin/sh
# report.md -> report.pdf (pandoc -> standalone HTML -> Edge headless print). Run from report/.
pandoc report.md -s --self-contained --css report.css --metadata title="AIMScribe RA Speech: technical report" -V title= -o report.html
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$PWD/report.pdf")" "file:///$(cygpath -m "$PWD/report.html")"
sleep 5   # Edge returns before the PDF is flushed
