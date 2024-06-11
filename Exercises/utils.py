import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import numpy as np
import pandas as pd
import math
from scipy.stats import chi2, norm, t
from scipy.special import kolmogorov

#########################################
########## COMPUTER EXERCISE 1 ##########
#########################################

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
        return X[1:]
    else:
        return X[1:]/M


# Histogram
def plot_histogram(U, title="", n_bins=20): 
    fig = go.Figure(go.Histogram(x=U, xbins=dict(start=0, end=1, size=1/n_bins), histnorm='probability density', marker=dict(color='Green')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1], mode='lines', line=dict(color='Red', width=2)))
    fig.update_layout(title=title, xaxis_title="Value", yaxis_title="Count", width=600, height=400, bargap=0.1, showlegend=False)
    fig.show()


# Correlation plot
def plot_correlation(U, title="Correlation plot of consequtive numbers."):
    fig = go.Figure(go.Scatter(x=U[:-1], y=U[1:], mode='markers', marker=dict(size=2, color='Blue')))
    fig.update_layout(title=title, xaxis_title=r"$U_{i-1}$", yaxis_title=r"$U_{i}$", width=600, height=600)
    fig.show()


def chi_sq_test(U, ps=None):
    
    # Compute expected number of observations in each class
    if ps is None:
        n_classes = 10
        n_expected = len(U) / n_classes
    else:
        n_classes = len(ps)
        n_expected = ps * len(U)

    # Count number of observations in each class
    n_obs, _ = np.histogram(U, bins=n_classes)

    # Compute test statistic
    T_obs = np.abs(np.sum((n_obs - n_expected)**2 / n_expected))

    # Compute p-value
    df = n_classes-1 # when number of estimated parameters is m=1
    p = 1 - chi2.cdf(T_obs, df)
    
    return T_obs, p


def kolmogorov_smirnov_test(U, ps=None):

    # Get number of observations
    n = len(U)

    # Setup expected values of F
    if ps is None:
        F_exp = np.linspace(0, 1, n+1)[1:]
        F_obs = np.sort(U)
    else:
        F_exp = np.cumsum(ps)
        n_obs, _ = np.histogram(U, bins=len(ps))
        F_obs = np.cumsum(n_obs / n)

    # Compute test statistic
    Dn = max(abs(F_exp-F_obs))

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


#########################################
########## COMPUTER EXERCISE 2 ##########
#########################################

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


#########################################
########## COMPUTER EXERCISE 3 ##########
#########################################

def exp_pdf(x, y=1):
    return y*np.exp(-y*x)

def pareto_pdf(x, k=20, beta=1):
    return k*beta**k / x**(k+1)

def gaussian_pdf(x, mu=0, sigma=1):
    return 1/np.sqrt(2*np.pi)*np.exp(-((x-mu)**2)/(2*sigma**2))

def uniform_2_exponential(U, y=1):
    X = -np.log(U)/y
    return X

def uniform_2_pareto(U, k=20, beta=1):
    X = beta*(U**(-1/k))
    return X

def uniform_2_normal(U):
    n = len(U)
    U1, U2 = U[:int(n/2)], U[int(n/2):]
    theta = 2*np.pi*U2
    r = np.sqrt(-2*np.log(U1))
    Z1, Z2 = r*np.array([np.cos(theta), np.sin(theta)])
    return np.concatenate((Z1, Z2))


#########################################
########## COMPUTER EXERCISE 4 ##########
#########################################

def describe_sample(data, title=None, alpha=0.05):
    n = len(data)
    mu = np.mean(data)
    s = np.std(data, ddof=1)
    t_val = t.ppf([alpha/2, 1 - alpha/2], n-1)
    s = np.std(data, ddof=1)
    CI = mu + t_val * s / np.sqrt(n)
    
    if alpha*100 % 1 == 0:
        CI_fraction = int((1-alpha)*100)

    print("━"*41)
    print("        >>> SAMPLE STATISTICS <<<        ")
    if title:
        pad = " "*int(max(0, ((41-len(title))/2)))
        print(pad + title + pad)
    print("━"*41)
    print(f"Sample size: {n:2d}")
    print(f"Mean: {mu:.4f}")
    print(f"Standard deviation: {s:.4f}")
    print(f"{CI_fraction}% confidence interval: {np.round(CI, 4)}")
    print("━"*41)


# def blocking_system_simulation(
#         num_service_units,
#         num_customers,
#         arrival_sample_fun,
#         service_time_sample_fun,
#         num_samples=None
#     ):

#     def single_sample():
#         blocked_customers = 0
#         event_departure = []
#         clock = 0

#         for _ in range(num_customers):
#             arrival_time = clock + arrival_sample_fun()

#             # Remove departed customers
#             event_departure = [x for x in event_departure if x > arrival_time]

#             # Check if there are available spots
#             if len(event_departure) < num_service_units:
#                 # Add customer
#                 service_time = service_time_sample_fun()
#                 event_departure.append(arrival_time + service_time)
#                 event_departure.sort()
#             else:
#                 blocked_customers += 1
            
#             # Update clock
#             clock = arrival_time
        
#         return blocked_customers/num_customers
    
#     if num_samples is None:
#         return single_sample()
#     else:
#         return np.array([single_sample() for _ in range(num_samples)])


def blocking_system_simulation(
        num_service_units,
        num_customers,
        arrival_sample_fun,
        service_time_sample_fun,
        num_samples=None
    ):


    def single_sample():
        # Initialize the state of the system
        service_units_occupied = np.zeros(num_service_units)
        blocked_customers = 0

        # Main loop
        for _ in range(num_customers):

            # Sample arrival of a new customer
            arrival = arrival_sample_fun()

            # Update the state of the system
            service_units_occupied = np.maximum(0, service_units_occupied - arrival)

            # Check if a service unit is available
            if any(service_units_occupied == 0):
                # Sample the service time and assign the customer to the first available service unit
                service_time = service_time_sample_fun()
                service_unit = np.argmin(service_units_occupied)
                service_units_occupied[service_unit] = service_time
            else:
                # Block the customer
                blocked_customers += 1
    
        return blocked_customers/num_customers
    
    if num_samples is None:
        return single_sample()
    else:
        return np.array([single_sample() for _ in range(num_samples)])


def simulation_stats(theta_hats, alpha, verbose=False):
    n = len(theta_hats)
    theta_bar = np.mean(theta_hats)
    S = np.sqrt((np.sum(theta_hats**2) - n*theta_bar**2)/(n-1))
    CI = theta_bar + S/np.sqrt(n) * t.ppf([alpha/2, 1-alpha/2], n-1)

    if verbose:
        print(f"Simulated blocking probability: {np.round(theta_bar, 4)}")
        print(f"{(1-alpha)*100}% confidence interval: {np.round(CI, 4)}")

    return np.array([theta_bar, theta_bar-CI[0], CI[1]-theta_bar])


def analytic_block_prob(lam, s, m):
    A = lam * s
    return A**m/math.factorial(m)/np.sum([A**i/math.factorial(i) for i in range(m+1)])


#########################################
########## COMPUTER EXERCISE 5 ##########
#########################################




#########################################
########## COMPUTER EXERCISE 6 ##########
#########################################




#########################################
########## COMPUTER EXERCISE 7 ##########
#########################################




#########################################
########## COMPUTER EXERCISE 8 ##########
#########################################

