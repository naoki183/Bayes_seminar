import numpy as np
import math
from scipy.stats import wishart
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import random
import time

# シード値
random_state = 42
# 訓練データ数
N = 10000
# 訓練時のクラスタ数
K = 4

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


def calc_eta_n(x_n, mu, Lambda, pi):
    eta_n = np.zeros(K)
    for k in range(K):
        mu_k = mu[k, :].reshape([2, 1])
        x_n = x_n.reshape([2, 1])
        Lambda_k = Lambda[k, :, :]
        s = (-1/2) * (x_n - mu_k).T @ Lambda_k @ (x_n - mu_k)
        eta_n[k] = math.exp(s[0][0] + (1/2) * math.log(np.linalg.det(Lambda_k)) + math.log(pi[k]))
    
    return eta_n / np.sum(eta_n)

def calc_beta_k_m_k_nu_k_W_k(beta, s_k, X, m, W, nu):
    beta_k = beta + np.sum(s_k)
    s = np.zeros(2)
    for i in range(N):
        s += s_k[i] * X[i, :]
    s += beta * m
    m_k = s / beta_k

    s = np.zeros((2, 2))
    for i in range(N):
        s += s_k[i] * X[i, :].reshape([2, 1]) @ X[i, :].reshape([1, 2])
    
    s += beta * m.reshape([2, 1]) @ m.reshape([1, 2]) - beta_k * m_k.reshape([2, 1]) @ m_k.reshape([1, 2]) + np.linalg.inv(W)
    W_k = np.linalg.inv(s)
    nu_k = np.sum(s_k) + nu

    return [beta_k, m_k, nu_k, W_k]

def calc_alpha_k(s_k, alpha_k):
    alpha_k = np.sum(s_k) + alpha_k
    return alpha_k

def sample_s_n(eta_n):
    i = np.random.choice(a=range(K), p=eta_n.tolist(), size=1)
    s_n = np.zeros(K)
    s_n[i] = 1
    return s_n

def sample_Lambda_k_mu_k(m_k, beta_k, nu_k, W_k):
    Lambda_k = wishart.rvs(nu_k, W_k, size=1, random_state=random_state)
    # Lambda_k = np.array(Lambda_k)
    mu_k = np.random.multivariate_normal(m_k, np.linalg.inv(beta_k * Lambda_k)) 
    return [Lambda_k, mu_k]

def sample_pi(alpha):
    pi = dirichlet.rvs(alpha, size=1)
    return pi[0]

def Gibbs_sampling(X, iter):
    # 初期化
    m = np.random.randn(2)
    beta = 3
    nu = 2
    w = np.random.randn(2, 2)
    W = w.T @ w
    Lambda = np.zeros((K, 2, 2))
    alpha = np.random.rand(K)
    mu = np.zeros((K, 2))
    for k in range(K):
        Lambda_k, mu_k = sample_Lambda_k_mu_k(m, beta, nu, W)
        Lambda[k, :, :] = Lambda_k
        mu[k, :] = mu_k
    pi = sample_pi(alpha)
    s = np.zeros((N, K))
    alpha_ = np.zeros(K)
    # サンプリングを繰り返す
    for i in range(iter):
        for n in range(N):
            eta_n = calc_eta_n(X[n, :], mu, Lambda, pi)
            s_n = sample_s_n(eta_n)
            s[n, :] = s_n
        for k in range(K):
            beta_k, m_k, nu_k, W_k = calc_beta_k_m_k_nu_k_W_k(beta, s[:, k], X, m, W, nu)
            Lambda_k, mu_k = sample_Lambda_k_mu_k(m_k, beta_k, nu_k, W_k)
            Lambda[k, :, :] = Lambda_k
            mu[k, :] = mu_k
        for k in range(K):
            alpha_k = calc_alpha_k(s[:, k], alpha[k])
            alpha_[k] = alpha_k
        pi = sample_pi(alpha_)
    
    return [Lambda, mu, pi]

def predict_distribution(M, Lambda, mu, pi):
    pred_data = np.zeros((M, 2))
    for i in range(M):
        k = np.random.choice(a=range(K), p=pi, size=1)
        Lambda_k = Lambda[k, :, :][0]
        mu_k = mu[k, :][0]
        x = np.random.multivariate_normal(mu_k, np.linalg.inv(Lambda_k))
        x = x.tolist()
        pred_data[i, :] = x
    return pred_data

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
    iter = 100
    M = 3000
    start_time = time.time()
    Lambda, mu, pi = Gibbs_sampling(X, iter)
    end_time = time.time()
    print(end_time - start_time)
    print(pi)
    pred_data = predict_distribution(M, Lambda, mu, pi)
    plot_points(pred_data)
    return

if __name__ == "__main__":
    main()