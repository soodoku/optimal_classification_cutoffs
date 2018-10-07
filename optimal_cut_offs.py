import numpy as np
from scipy import optimize


def _accuracy(prob, true_labs, pred_prob, verbose=False):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
        print("Probability: {0:0.4f} Accuracy: {1:0.4f}".format(prob[0], accuracy))
    return 1 - accuracy


def _f1(prob, true_labs, pred_prob, verbose=False):
    tp, tn, fp, fn = get_confusion_matrix(true_labs, pred_prob, prob[0])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("Probability: {0:0.4f} F1 score: {1:0.4f}".format(prob[0], f1))
    return 1 - f1


def get_confusion_matrix(true_labs, pred_prob, prob):
    pred_labs = pred_prob > prob
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp = np.sum(np.logical_and(pred_labs == 1, true_labs == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.sum(np.logical_and(pred_labs == 0, true_labs == 0))    
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(pred_labs == 1, true_labs == 0))    
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.sum(np.logical_and(pred_labs == 0, true_labs == 1))

    return tp, tn, fp, fn


def get_probability(true_labs, pred_prob, objective='accuracy', verbose=False):
    if objective == 'accuracy':
        prob = optimize.brute(_accuracy, (slice(0.1, 0.9, 0.1),), args=(true_labs, pred_prob, verbose), disp=verbose)
    elif objective == 'f1':
        prob = optimize.brute(_f1, (slice(0.1, 0.9, 0.1),), args=(true_labs, pred_prob, verbose), disp=verbose)
    else:
        raise ValueError('`objective` must be `accuracy` or `f1`')
    return prob[0]
