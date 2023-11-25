# bash

for noise in iid ar; do
    for signal in 0.1 0.2 0.3 0.4; do
        python experiment/experiment.py \
            --signal $signal \
            --mode lt \
            --noise $noise
    done
done
