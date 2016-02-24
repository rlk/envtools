#!/bin/bash
export PATH=./:$PATH
extractLights testData/graceCath.jpeg > outTest.json
DIFF=$(diff -q outTest.json testData/graceCath.json)
rm outTest.json
if [ "$DIFF" != "" ]
then
   echo "[ERROR] extractlight output diff changed"
else
  echo "[OK] extractLights output diff test OK"
fi
