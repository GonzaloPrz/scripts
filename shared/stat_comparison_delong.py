import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import pandas as pd
from pathlib import Path
import itertools,pickle
import sys,json,os
from scipy.stats import chi2

def delong_auc_test(y_true, y_pred_1, y_pred_2):
    """
    Perform DeLong's test to compare AUCs of two models.

    Parameters:
        y_true (array): True binary labels (0 or 1)
        y_pred_1 (array): Predicted probabilities from Model 1
        y_pred_2 (array): Predicted probabilities from Model 2
    
    Returns:
        p-value: Significance level of the AUC difference
    """
    def compute_auc_var(y_true, y_scores):
        """ Compute AUC variance using DeLong method """
        n = len(y_true)
        auc = roc_auc_score(y_true, y_scores[:,1])
        pos = y_scores[y_true == 1,1]
        neg = y_scores[y_true == 0,1]
        n_pos, n_neg = len(pos), len(neg)
        v = np.var([np.mean(pos > neg[j]) for j in range(n_neg)])
        return auc, v / n

    # Compute AUC and variance for both models
    auc1, var1 = compute_auc_var(y_true, y_pred_1)
    auc2, var2 = compute_auc_var(y_true, y_pred_2)

    # Compute Z-score
    se = np.sqrt(var1 + var2)
    z_score = (auc1 - auc2) / se
    p_value = 2 * norm.sf(abs(z_score))  # Two-tailed test

    print(f"Model 1 AUC: {auc1:.4f}")
    print(f"Model 2 AUC: {auc2:.4f}")
    print(f"Z-score: {z_score:.4f}")
    print(f"p-value: {p_value:.4f}")

    return z_score, p_value


        #y_true = np.concatenate([y_true[j,r] for j,r in itertools.product(range(y_true.shape[0]),range(y_true.shape[1]))])

        #z_score, p_value = delong_auc_test(y_true, 
        #                                   np.concatenate([outputs_1[j,r] for j,r in itertools.product(range(outputs_1.shape[0]),range(outputs_1.shape[1]))]),
        #                                    np.concatenate([outputs_2[j,r] for j,r in itertools.product(range(outputs_2.shape[0]),range(outputs_2.shape[1]))]))
        p_values = []
        for j,r in itertools.product(range(outputs_1.shape[0]),range(outputs_1.shape[1])):
            z_score, p = delong_auc_test(y_true[j,r],outputs_1[j,r],outputs_2[j,r])
            p_values.append(p)
        
        chi2_stat = -2 * np.sum(np.log(p_values))
        combined_p_value = 1 - chi2.cdf(chi2_stat, 2 * len(p_values))
        
        stats_append = {'comparison':f'{model1} vs {model2}','z_score':np.round(chi2_stat,3),'p_value':np.round(combined_p_value,3)}
        
        if stats.empty:
            stats = pd.DataFrame(stats_append,index=[0])
        else:
            stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)

stats.to_csv(Path(results_dir,'stats_comparison_delong_bayes.csv' if bayes else 'stats_comparison_delong.csv'),index=False)