#!/usr/bin/env bash
echo "Copying important files ..."
# copy data files into AMPS module
mkdir target/easyAMPS.app/Contents/MacOS/AMPS/
# cp -vf src/main/python/delnatan.mplstyle target/easyAMPS.app/Contents/MacOS/AMPS/
cp -vf src/main/python/AMPS/*.pickle target/easyAMPS.app/Contents/MacOS/AMPS/
cp -vf src/main/python/AMPS/*.csv target/easyAMPS.app/Contents/MacOS/AMPS/