#!/bin/bash
export PATH=./:$PATH

mkdir -p out
rm -f out/debug_variance.png
rm -f outTest.json

extractLights -d testData/graceCath.jpeg > outTest.json
DIFF=$(oiiotool --diff --failpercent 0.001 testData/debug_variance.png testData/out/debug_variance.png)

if [[ "$DIFF" != *PASS* ]]
then
   echo "[ERROR] extractlight output diff changed"
else
  rm -f outTest.json
  rm -f out/debug_variance.png
  echo "[OK] extractLights output diff test OK"
fi
