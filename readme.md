# Selective Inference for Changepoint detection by Reccurent Neural Network
This pacakge is the implementation of the paper "Selective Inference for Changepoint detection by Reccurent Neural Network" for experiments.

## Installation & Requirements
This pacakage has the following dependencies:
- Python (version 3.10 or higher, we use 3.10.11)
- sicore (we use version 1.0.0)
- tensorflow (we use version 2.11.1)
- tqdm

Please install these dependencies by pip.
```
pip install sicore
pip install tensorflow
pip install tqdm
```

## Reproducibility

Since we have already got the results in advance, you can reproduce the figures by running following code. The results will be saved in "/image" folder.
```
sh plot.sh
```

To reproduce the results, please see the following instructions after installation step.
The results will be saved in "./results" folder as pickle file.

For the mean-shift case (type I error rate experiment).
```
sh execute_ms_null.sh
```

For the mean-shift case (power experiment).
```
sh execute_ms_alter.sh
```

For the linear-trend-shift case (type I error rate experiment).
```
sh execute_lt_null.sh
```

For the linear-trend-shift case (power experiment).
```
sh execute_lt_alter.sh
```

For visualization of the reproduced results.
```
sh plot.sh
```
