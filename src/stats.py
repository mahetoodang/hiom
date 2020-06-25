import unidip.dip as dip
import numpy as np

"""
File contains three methods to quantify the polarization within the population. 
1. Hartigan's D test (which is increasing when the distribution is less similar to unimodal distribution)
2. Fraction of the population holding a view in accordance with the minority.
3. Mean opinion. 
"""

def compute_hartigan_opinions(opinion):
    """
    Function to compute Hartigan's Dip test of unimodality in distribution of opinions in a selected step.

    Arguments
    ---------
    opinion : [ float ]
        Series of opinions defined as output of the data_collector, i.e., [[1, 0.5], [2, 0.2], ... ]

    Returns
    -------
    (dip, pvalue, indices) : tuple of floats
        Dip - Hartigans' dip statistic as defined in Hartigan, J. A.; Hartigan, P. M. The Dip Test 
        of Unimodality. The Annals of Statistics 13 (1985), no. 1, 70--84. doi:10.1214/aos/1176346577.
        http://projecteuclid.org/euclid.aos/1176346577
        pvalue - P-value specifying the similarity of the distribution to an unimodal distribution. In short: 
        The smaller the value, the more likely it is that the distribution is not unimodal.
        indices - left and center indices of the dip
    """

    raw_opinions = [x[1] for x in opinion]
    return dip.diptst(raw_opinions)

def compute_fractions_size(opinion):
    """
    Computing fraction of the population which holds the view closer to 1 (usually the minority view).

    Arguments
    ---------
    opinion : [ float ]
        Series of opinions defined as output of the data_collector, i.e., [[1, 0.5], [2, 0.2], ... ]

    Returns
    -------
    (fraction, n_plus, n_minus) : tuple of float, int and int
        fraction - the number of agents with opion > 0.0 divided by the total number of opinions 
        (i.e., population size).
        n_plus - number of agents with opion > 0.0
        n_minus - number of agents with opion <= 0.0 
    """
    raw_opinions = [x[1] for x in opinion]
    n_minus = np.sum([o <= 0.0 for o in raw_opinions])
    n_plus = np.sum([o > 0.0 for o in raw_opinions])
    return n_plus/len(raw_opinions), n_plus, n_minus

def compute_mean_opinion(opinion):
    """
    Computing mean opinion within the population. Also returns standard deviation.

    Arguments
    ---------
    opinion : [ float ]
        Series of opinions defined as output of the data_collector, i.e., [[1, 0.5], [2, 0.2], ... ]

    Returns
    -------
    (mean, stdev) : tuple of floats
        mean - average value of the opinions passed in the parameter
        stdev - standard deviation of the opinions passed in the parameter
    """
    raw_opinions = [x[1] for x in opinion]
    return np.mean(raw_opinions), np.std(raw_opinions)