import numpy as np
import scipy.stats as stats


def first_type():
    attention = 0.01 * np.random.random()
    opinion = 0
    lower, upper = -0.1, 0.1
    mu, sigma = 0, 1
    information = stats.truncnorm(
        (lower - mu) / sigma,
        (upper - mu) / sigma,
        loc=mu,
        scale=sigma
    ).rvs(1)[0]
    return {
        "attention": attention,
        "opinion": opinion,
        "information": information
    }


def second_type():
    attention = np.random.uniform(0, 0.5)
    opinion = 1
    information = 1
    return {
        "attention": attention,
        "opinion": opinion,
        "information": information
    }


agents = [
    {
        "n": 50,
        "generator": first_type
    },
    {
        "n": 10,
        "generator": second_type
    }
]
