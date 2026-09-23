#!/bin/zsh
source ./venv-sb/bin/activate
for i in `seq 3 10`;
do
    echo level $i
    python sb-play.py -m edax_bc_pretrained_CNN_test -e 100 -o Edax --depth $i -d;
done
