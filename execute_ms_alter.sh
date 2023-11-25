# bash

for noise in iid ar; do
    for signal in 1.0 2.0 3.0 4.0; do
        python experiment/experiment.py \
            --signal $signal \
            --mode ms \
            --noise $noise
    done
done
