#!/bin/bash
export PATH=./:$PATH

rm -f out/*
mkdir -p out

./process_environment.py --fixedge --write-by-channel --nbSamples 512 --backgroundSamples 512 testData/graceCath.jpeg out/

DIFF=$(oiiotool --diff --failpercent 0.001 /tmp/specular_4.tif testData/specular_4.tif)


if [[ "$DIFF" != *PASS* ]]
then
   echo "[ERROR] process env cubemap specular output diff changed"
else
    rm -f out/*
    rmdir out
    echo "[OK] output diff test OK"
fi
