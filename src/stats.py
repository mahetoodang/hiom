import unidip.dip as dip

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