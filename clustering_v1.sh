#!/bin/bash

source activate klusta;

c=0

for i in $(find . -type d -not -path '*/\.*');
  do
    ((++c))
done

p=0

for i in $(find $PWD -type d -not -path '*/\.*');
  do
    ((++p))
    cd $i
    echo -ne "\rProgress: $p/$c"
    klusta params.prm &> /dev/null
    rm test_signal.raw.kwd &> /dev/null
done

echo -e '\nDone.'

source deactivate klusta;
