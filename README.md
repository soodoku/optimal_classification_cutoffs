## Optimal Cut-Offs

Probabilities from classification models can have two problems: 

1. Miscalibration: A p of .9 often doesn't mean a 90% chance of 1 (assuming a dichotomous y). (You can calibrate it using isotonic regression.)

2. Optimal cut-offs: For multi-class classifiers, we do not know what probability value will maximize the accuracy or F1 score. Or any metric for which you need to trade-off between FP and FN.

Here we share a solution for #2. It involves running the outputs through a brute-force optimizer. We provide a simple wrapper to make it yet easier to use.

### Function

The function `get_probability` takes the following arguments: 

1. `true_labs` (required): NumPy array or Pandas Series in which the true labels are stored. 
2. `pred_prob` (required): NumPy array or Pandas Series in which the predicted probabilities are stored.
3. `objective` (optional): `accuracy` (default) or `f1`
4. `verbose` (optional): `True` or `False` (default) to show/hide verbose messages.

The function outputs a numeric p-value that gives the lowest F1-score or FP+FN (max. accuracy).

### Usage

To use the [function](optimal_cut_offs.py), just download it and put it in the local directory and call import. 

```
import optimal_cut_offs

df = ...

p = optimal_cut_offs.get_probability(df.true_labs, df.pred_prob, 'accuracy')

```

### Illustration

Check out this [Jupyter notebook](comscore.ipynb) to see the script in action. 
For context, the notebook underlies the outputs you see [here](https://github.com/themains/domain_knowledge/blob/master/scripts/porn.ipynb).

### Authors

Suriyan Laohaprapanon and Gaurav Sood
