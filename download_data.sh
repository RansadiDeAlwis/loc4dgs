#!/usr/bin/env bash
# Guide Section 4.3: get the Dynamic 3D Gaussians data (data.zip) and pretrained scenes (output.zip).
# Files land next to loc4dgs/:  ../data/<scene>/...   and   ../output/pretrained/<scene>/params.npz
set -e
cd "$(dirname "$0")/.."
[ -d Dynamic3DGaussians ] || git clone https://github.com/JonathonLuiten/Dynamic3DGaussians
echo "Download links found in the Dynamic3DGaussians README:"
grep -oE 'https?://[^ )"]+\.zip' Dynamic3DGaussians/README.md | sort -u | tee /tmp/loc4dgs_links.txt
DATA=$(grep -m1 'data\.zip' /tmp/loc4dgs_links.txt || true)
OUTP=$(grep -m1 'output\.zip' /tmp/loc4dgs_links.txt || true)
if [ -z "$DATA" ] || [ -z "$OUTP" ]; then
  echo "Could not find both links automatically; download data.zip and output.zip by hand from the README."; exit 1
fi
[ -f data.zip ] || wget -c "$DATA" -O data.zip
[ -f output.zip ] || wget -c "$OUTP" -O output.zip
unzip -n -q data.zip && unzip -n -q output.zip
echo "Now check the layout (guide 4.3):"; ls data output output/* | head -30
