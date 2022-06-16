import numpy as np
import math
from scipy.special import psi
import matplotlib.pyplot as plt
from scipy.stats import wishart
from scipy.stats import dirichlet
import time

# データの次元数
D = 2
# カテゴリ数
K = 3
# 訓練データ数
N = 10000
# イテレーション数
iter = 100
# シード値
random_state = 42

np.random.seed(random_state)

def gen_dataset():
    k1 = np.random.rand(1000, 2) + 2
    k2 = np.random.rand(1000, 2)
    k3 = np.random.randn(1000, 2)
    for i in range(1000):
        k3[i, 1] += 2
        k3[i, 0] -= 2
    return [k1, k2, k3]

def gen_train_data():
    X = np.zeros((N, 2))
    for i in range(N):
        k = np.random.choice(a=[0, 1, 2], p=[1/3, 1/3, 1/3], size=1)
        if k == 0:
            x = np.random.rand(2) + 2
            X[i, :] = x
        elif k == 1:
            x = np.random.rand(2)
            X[i, :] = x
        else:
            x = np.random.randn(2)
            x[0] -= 2
            x[1] += 2
            X[i, :] = x
    
    return X

def calc_Es(nu_k, W_k, m_k, beta_k):
    E_Lambda_k = nu_k * W_k
    E_ln_det_Lambda_k = D * math.log(2) + math.log(np.linalg.det(W_k))
    for d in range(D):
        E_ln_det_Lambda_k += psi((nu_k + 1 - d) / 2)
    E_Lambda_k_mu_k = nu_k * W_k @ m_k
    s = m_k.reshape([1, D]) @ W_k @ m_k.reshape([D, 1])
    E_mu_k_Lambda_k_mu_k = nu_k * s[0][0] + D / beta_k
    return [E_Lambda_k, E_ln_det_Lambda_k, E_Lambda_k_mu_k, E_mu_k_Lambda_k_mu_k]


def update_eta(X, nu, W, m, beta, alpha):
    Es = []
    for k in range(K):
        Es.append(calc_Es(nu[k], W[k, :, :], m[k, :], beta[k]))
    eta = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            E_Lambda_k, E_ln_det_Lambda_k, E_Lambda_k_mu_k, E_mu_k_Lambda_k_mu_k = Es[k]
            x_n = X[n, :]
            s = x_n.reshape([1, D]) @ E_Lambda_k @ x_n.reshape([D, 1])
            t = x_n.reshape([1, D]) @ E_Lambda_k_mu_k
            eta[n, k] = math.exp(-(1/2) * s[0][0] + t[0] - (1/2) * E_mu_k_Lambda_k_mu_k + (1/2) * E_ln_det_Lambda_k + psi(alpha[k]) - psi(np.sum(alpha)))
        eta[n, :] = eta[n, :] / np.sum(eta[n, :])
    
    return eta

def update_beta_k_m_k_nu_k_W_k(eta, beta, X, m, W, nu, k):
    beta_k = beta + np.sum(eta[:, k])
    s = np.zeros(D)
    for n in range(N):
        s += eta[n, k] * X[n, :]
    m_k = (s + beta * m) / beta_k
    W_k = np.zeros((D, D))
    for n in range(N):
        W_k += eta[n, k] * X[n, :].reshape([D, 1]) @ X[n, :].reshape([1, D])
    W_k += beta * m.reshape([D, 1]) @ m.reshape([1, D]) - beta_k * m_k.reshape([D, 1]) @ m_k.reshape([1, D]) + np.linalg.inv(W)
    W_k = np.linalg.inv(W_k)
    nu_k = np.sum(eta[:, k]) + nu

    return [beta_k, m_k, W_k, nu_k]

def update_alpha(eta, alpha):
    alpha_ = np.zeros(K)
    for k in range(K):
        alpha_[k] = np.sum(eta[:, k]) + alpha[k]
    return alpha_

def hennbunn_suironn(X):
    beta = 3
    nu = 2
    w = np.random.randn(D, D)
    W = w.T @ w
    m = np.random.randn(D)
    beta_ = np.zeros(K)
    nu_ = np.zeros(K)
    W_ = np.zeros((K, D, D))
    m_ = np.zeros((K, D))
    for k in range(K):
        beta_[k] = 2
        nu_[k] = 3
        w = np.random.randn(D, D)
        W_[k, :, :] = w.T @ w
        m_[k, :] = np.random.randn(D)
    alpha = np.random.rand(K) * 10
    alpha_ = np.random.rand(K) * 10
    eta = np.zeros((N, K))
    for i in range(iter):
        eta = update_eta(X, nu_, W_, m_, beta_, alpha_)
        for k in range(K):
            beta_[k], m_[k, :], W_[k, :, :], nu_[k] = update_beta_k_m_k_nu_k_W_k(eta, beta, X, m, W, nu, k)
        alpha_ = update_alpha(eta, alpha)
    
    return [alpha_, beta_, nu_, W_, m_]


def plot_points(pred_data):
    k1, k2, k3 = gen_dataset()
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.scatter(k1[:, 0], k1[:, 1], label="cluster1", color="green")
    ax.scatter(k2[:, 0], k2[:, 1], label="cluster2", color="blue")
    ax.scatter(k3[:, 0], k3[:, 1], label="cluster3", color="grey")
    ax.scatter(pred_data[:, 0], pred_data[:, 1], label="prediction", color="red")
    plt.legend()
    plt.show()
    return

def main():
    X = gen_train_data()
    # サンプリングの回数
    M = 3000
    start_time = time.time()
    alpha_, beta_, nu_, W_, m_ = hennbunn_suironn(X)
    end_time = time.time()
    print(end_time - start_time)
    print(alpha_)
    pred_data = np.zeros((M, D))
    for i in range(M):
        pi = dirichlet.rvs(alpha_, size=1)
        pi = pi[0]
        k = np.random.choice(range(K), p=pi, size=1)
        Lambda_k = wishart.rvs(nu_[k][0], W_[k, :, :][0], size=1, random_state=random_state)
        mu_k = np.random.multivariate_normal(m_[k, :][0], np.linalg.inv(beta_[k][0] * Lambda_k))
        x = np.random.multivariate_normal(mu_k, np.linalg.inv(Lambda_k))
        pred_data[i, :] = x
    plot_points(pred_data)
    return
    





if __name__ == "__main__":
    main()
