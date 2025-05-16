## Experiment: Assess the claims of network similarity after fine-tuning

### Sub-Experiment 1: Compute the Mean Absolute Difference of Target Softmax Outputs between original and fine-tuned models

To run:
```
python softmax_mad_experiment.py --models_dir 'Path to the tuned models directory' --test_dir 'Path to the test set directory'
```

Results available in softmax_mad_experiment.csv

### Sub-Experiment 2: Compute the Mean Absolute Difference of Attribution Maps between original and fine-tuned models using different Attribution Methods

To run:
```
python am_mad_experiment.py --models_dir 'Path to the tuned models directory' --test_dir 'Path to the test set directory'
```

Results available in am_*_mad_experiment.csv
