import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.special import kolmogorov

###########################
########## DAY 1 ##########
###########################

def LCG(a:int, c:int, M:int, x0:int, n:int, as_int=False):
    """
    LCG generates a list of random numbers using the Linear Congruent Generation method.
    """
    # Preallocate and initialize array for pseudorandom numbers
    X = np.zeros(n+1, dtype=int)
    X[0] = x0

    # Generate pseudorandom numbers
    for i in range(1, n+1):
        X[i] = (a*X[i-1] + c) % M
    
    if as_int:
        return X
    else:
        return X/M
    

# Histogram
def plot_histogram(U, title="", n_bins=10): 
    counts, bins = np.histogram(U, bins=n_bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = fig = go.Figure(data=[go.Bar(x=bins, y=counts, marker_color='Green')])
    fig.update_layout(
        title_text=title,
        xaxis_title="Value",
        yaxis_title="Count",
        width=600,
        height=400,
        bargap=0.1,
    )
    fig.show()


# Correlation plot
def plot_correlation(U, title="Correlation plot of consequtive numbers."):
    n = len(U)
    fig = go.Figure(data=go.Scatter(x=U[:n-1], y=U[1:n], mode='markers', marker=dict(size=2, color='Blue')))
    fig.update_layout(
        title=title,
        xaxis_title=r"$U_{i-1}$",
        yaxis_title=r"$U_{i}$",
        width=600,
        height=600,
    )
    fig.show()


def chi_sq_test(U, n_classes=10):
    
    # Compute expected number of observations in each class
    n_expected = len(U) / n_classes
    
    # Count number of observations in each class
    n_obs, _ = np.histogram(U, bins=n_classes)

    # Compute test statistic
    T_obs = np.abs(np.sum((n_obs - n_expected)**2 / n_expected))

    # Compute p-value
    df = n_classes-1 # when number of estimated parameters is m=1
    p = 1 - chi2.cdf(T_obs, df)
    
    return T_obs, p


def kolmogorov_smirnov_test(U):

    # Get number of observations
    n = len(U)

    # Setup expected values of F
    F_exp = np.linspace(0, 1, n+1)[1:]

    # Compute test statistic
    Dn = max(abs(F_exp-np.sort(U)))

    # Compute p-value
    p = kolmogorov(Dn)

    return Dn, p


def above_below_runtest1(U):

    median = np.median(U)
    n1 = np.sum(U < median)
    n2 = np.sum(median < U)

    # Compute total number of observed runs
    temp = U > median
    T_obs = sum(temp[1:] ^ temp[:-1])

    # Compute p-value
    mean = 2*n1*n2/(n1 + n2) + 1
    log_expr = np.log(2) + np.log(n1) + np.log(n2) + np.log(2*n1*n2 - n1 - n2) - 2*np.log(n1 + n2) - np.log(n1 + n2 - 1)
    var = np.exp(log_expr)
    Z_obs = (T_obs - mean) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(np.abs(Z_obs)))

    return T_obs, p


def up_down_runtest2(U):

    n = len(U)

    # Get indeces where runs change (Append -1 and n-1 at ends to handle first and last run)
    idx = np.concatenate(([-1], np.where(U[1:]-U[:-1] < 0)[0], [len(U)-1]))

    # Compute run lengths and count them (clamp to 6)
    run_lengths = np.clip(idx[1:] - idx[:-1], 1, 6)
    R = np.array([np.count_nonzero(run_lengths == i) for i in range(1, 7)])

    # Compute test statistic
    A = np.array([
        [4529.4, 9044.9, 13568, 18091, 22615, 27892],
        [9044.9, 18097, 27139, 36187, 45234, 55789],
        [13568, 27139, 40721, 54281, 67852, 83685],
        [18091, 36187, 54281, 72414, 90470, 111580],
        [22615, 45234, 67852, 90470, 113262, 139476],
        [27892, 55789, 83685, 111580, 139476, 172860]
    ])
    B = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    Z_obs = (1/(n - 6)) * (R - n*B).T @ A @ (R - n*B)

    # Compute p-value
    p = 1 - chi2.cdf(Z_obs, 6)
    
    return Z_obs, p


def up_and_down_runtest3(U):

    n = len(U)

    # Find runs (Append 0 at ends to handle first and last run)
    seq = np.concatenate(([0], np.sign(U[1:] - U[:-1]), [0]))

    # Get indeces where runs change
    idx = np.flatnonzero(seq[:-1] != seq[1:])

    # Compute run lengths
    run_lengths = idx[1:] - idx[:-1]
    X_obs = len(run_lengths)

    # Compute test statistic
    Z_obs = (X_obs - (2*n-1)/3) / np.sqrt((16*n - 29) / 90)

    # Compute p-value
    p = 2*(1 - norm.cdf(np.abs(Z_obs)))

    return Z_obs, p


def corr_coef(U, h=2):
    n = len(U)
    ch = np.sum(U[:n-h]*U[h:])/(n-h)
    Z = (ch - 0.25)/(7/(144*n))
    p = 2*(1 - norm.cdf(np.abs(ch)))
    return ch, p


def test_random_numbers(U, verbose=True):

    tests = [chi_sq_test, kolmogorov_smirnov_test, above_below_runtest1, up_down_runtest2, up_and_down_runtest3, corr_coef]
    table = np.array([test(U) for test in tests])

    df = pd.DataFrame(np.round(table, 2),
                      index=["Chi squared", "Kol-Smi", "Above/Below (I)", "Up/Down (II)", "Up and Down (III)", "Correlation"],
                      columns=["Test statistic", "p-value"]
    )
    if verbose:
        print(df)
    return df


###########################
########## DAY 2 ##########
###########################

def direct(n, ps):
    # Generate uniform random numbers
    U = np.random.rand(n)

    # Convert to discrete random numbers using the given probabilities
    X = np.searchsorted(np.cumsum(ps), U)

    return X


def rejection(n, ps):
    c = max(ps)
    k = len(ps)

    X = np.zeros(n, dtype=int)
    for i in range(len(X)):
        while True: # Could theoretically run forever...
            U1, U2 = np.random.rand(2)
            I = np.floor(k * U1).astype(int)
            if U2 <= ps[I]/c:
                X[i] = I + 1
                break
    
    return X


def alias(N, ps):
    k = len(ps)

    # Generating Alias tables
    L = np.arange(k)
    F = k*ps
    G = np.where(F >= 1)[0]
    S = np.where(F <= 1)[0]

    while len(S) != 0:
        i = G[0]
        j = S[0]
        L[j] = i
        F[i] -= (1 - F[j])
        if F[i] < 1 - np.finfo(float).eps:
            G = np.delete(G, 0)
            S = np.append(S, i)
        S = np.delete(S, 0)

    # Computing values
    X = np.zeros(N, dtype=int)

    # Generate random numbers
    U1 = np.random.rand(N)
    U2 = np.random.rand(N)

    # Perform Alias method
    I = np.array(np.floor(k * U1)).astype(int)
    mask = U2 <= F[I]
    X[mask] = I[mask] + 1
    X[~mask] = L[I[~mask]] + 1

    return X