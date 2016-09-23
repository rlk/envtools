#!/bin/bash
export PATH=./:$PATH

rm -f out/*
mkdir -p out

./process_environment.py --fixedge --write-by-channel --nbSamples 512 --backgroundSamples 512 testData/graceCath.jpeg out/

DIFF=$(oiiotool --diff -a --fail 0.05 /tmp/specular_4.tif testData/specular_4.tif)


if [[ "$DIFF" == *FAILURE* ]]
then
   echo "[ERROR] process env cubemap specular output diff changed"
else
    rm -f out/*
    rmdir out
    echo "[OK] output diff test OK"
fi
