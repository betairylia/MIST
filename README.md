# MIST

This is the implementation of "Deep Clustering via Mutual Information Maximization with Local Smoothness and Topological Invariant Constraints".

## Usage

1. Download the datasets used in the paper: https://drive.google.com/file/d/1h-1xx8MrgsRHwAKh4geltcn0-NeCVd_m/view?usp=sharing 

2. Run `train_mist.py` to run the MIST experiment. hyper-parameters can be assigned via command-line arguments.  
use `python train_mist.py -h` for instructions.

For example, use following command to run our experiments on MNIST:
```bash
python train_mist.py --dataset mnist --UseSuggestedK0Beta 1
```

Use `--UseSuggestedK0Beta 1` to use suggested values for `K0` and `beta` as shown in the paper. Their values varies from datasets.

3. `train_traditional_clustering.py` is provided to reproduce our results for K-means, Spectral Clustering and Gaussian Mixture Model Clustering.
