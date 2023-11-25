# bash

for noise in iid ar; do
    for d in 100 80 60 40; do
        python experiment/experiment.py \
            --d $d \
            --mode lt \
            --noise $noise
    done
done
