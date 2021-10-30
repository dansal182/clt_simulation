import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, gamma, pi
from matplotlib.animation import FuncAnimation


def get_t_val(simvalues):
    return (simvalues.mean()) / (simvalues.std() / sqrt(len(simvalues)))


def get_s_error(simvalues):
    return simvalues.std() / sqrt(len(simvalues))


def get_mu_est(simvalues):
    return simvalues.mean()


def sim_experiment(sample_size, nsim, mu, sigma):
    t_values = np.linspace(0, 100, nsim)
    x_values = np.linspace(0, 100, nsim)
    s_errors = np.linspace(0, 100, nsim)
    for i, i0 in enumerate(t_values):
        x = np.random.normal(mu, sigma, sample_size)
        t_values[i] = get_t_val(x)
        x_values[i] = get_mu_est(x)
        s_errors[i] = get_s_error(x)
    out = pd.DataFrame()
    out['valor_T'] = t_values
    out['Estimador'] = x_values
    out['error_std'] = s_errors
    return out


def t_den_fun(t, df):
    return gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2)) * (1 + t ** 2 / df) ** (-(df + 1) / 2)


def norm_den(x, mu, sig):
    return (1 / sqrt(2 * pi * sig * sig)) * np.exp(- 1 / (2 * sig * sig) * (x - mu) ** 2)


def gamma_den(x, alpha, lam):
    return lam**alpha / gamma(alpha) * x**(alpha - 1) * np.exp(-lam*x)


def get_hist(t_values, n_bin):
    fig, ax = plt.subplots()
    ax.hist(t_values, bins=n_bin, density=True)
    t_supp = np.linspace(min(t_values), max(t_values), len(t_values))
    y = t_den_fun(t_supp, len(t_values) - 1)
    ax.plot(t_supp, y, '--')
    ax.set_xlabel('Valor t')
    ax.set_ylabel('Likelihood/Verosimilitud')
    ax.set_title('Histograma del valor t')
    fig.tight_layout()
    plt.show()


class UpdateSample:
    def __init__(self, ax, mu, sigma, size):
        self.line, = ax.plot([], [], 'k-')
        self.ax = ax
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(0, 5)
        self.ax.grid(False)
        self.ax.axvline(mu, linestyle='--', color='red')
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.x = np.linspace(-2, 2, 250)

    def __call__(self, i):
        if i == 0:
            self.line.set_data([], [])
            return self.line,
        else:
            self.size = i
            y = norm_den(self.x, self.mu, self.sigma / sqrt(i))
            self.line.set_data(self.x, y)
            return self.line,


def create_movie(kseed, frms, ints):
    np.random.seed(kseed)
    fig, ax = plt.subplots()
    ud = UpdateSample(ax, 0, 1, 1)
    anm = FuncAnimation(fig, ud, frames=frms, interval=ints, blit=True)
    plt.show()
    return anm


class UpdateCLT:
    def __init__(self, ax, mu, sigma, size):
        self.line, = ax.plot([], [], 'k-')
        self.ax = ax
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 2)
        self.ax.grid(False)
        self.ax.axvline(mu, linestyle='--', color='red')
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.x = np.linspace(0, 5, 150)

    def __call__(self, i):
        if i == 0:
            self.line.set_data([], [])
            return self.line,
        else:
            self.size = i
            y = gamma_den(self.x, i, 1/self.mu * i)
            self.line.set_data(self.x, y)
            return self.line,


def create_movie_clt(kseed, frms, ints):
    np.random.seed(kseed)
    fig, ax = plt.subplots()
    fig.suptitle('CLT Animation', fontsize=16)
    ud = UpdateCLT(ax, 2, 1, 1)
    anm = FuncAnimation(fig, ud, frames=frms, interval=ints, blit=True)
    plt.show()
    return anm

