# Gaussian mixture model

This code reproduces experiments in section 4.1 in the paper.

## Train

To train a single model, run
```
python run.py
```

To run a sweep over models in the paper, run
```
python sweep.py
```

## PLot

To plot Figure 4, you must first run the sweep above and then run
```
python plot.py
```