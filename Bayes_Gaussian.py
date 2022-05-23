import numpy as np
import math
from random import random
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_t

def p(x, mu, Lambda, nu, D):
    return math.pow(1 + (1 / nu) * ((x - mu).T @ Lambda @ (x - mu)), -(nu + D) / 2)

def sample_t(N, mu_s, Lambda_s, nu_s, D):
    sample = []
    while len(sample) != N:
        while True:
            # 規格化定数がわからないので棄却サンプリング
            z = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * 2
            if p(z, mu_s, Lambda_s, nu_s, D) > random():
                sample.append(z)
                break
    return sample

def main():
    #　学習データ数
    N = 10000
    # 次元数
    D = 2
    # ハイパーパラメータ
    _beta = 0.1
    _m = np.random.rand(D)
    _nu = 100
    A = np.random.rand(D, D)
    _W = A.T @ A

    X = np.random.rand(N, D)
    beta_ = N + _beta
    s = 0
    for i in range(N):
        s += X[i, :]
    m_ = (1 / beta_) * (s + _beta * _m)
    nu_ = N + _nu
    t = np.zeros((D, D))
    for i in range(N):
        t += X[i, 0:D].reshape([1, D]).T @ X[i, 0:D].reshape([1, D])
    print(t)
    W_ = np.linalg.inv(t + _beta * (_m.reshape([D, 1]) @ _m.reshape([1, D])) - beta_ * (m_.reshape([D, 1]) @ m_.reshape([1, D])) + np.linalg.inv(_W))
    mu_s = m_
    Lambda_s = (((1 + nu_ - D) * beta_) / (1 + beta_)) * W_
    nu_s = 1 + nu_ - D
    X_test = np.random.rand(1000, D)
    X_pred = sample_t(1000, mu_s, Lambda_s, nu_s, D)
    X_pred1 = multivariate_t(mu_s, np.linalg.inv(W_), df=nu_s, shape=[1000, 1])
    for i in range(1000):
        plt.scatter(X_test[i][0], X_test[i][1], c="red")
        plt.scatter(X_pred[i][0], X_pred[i][1], c="blue")
        plt.scatter(X_pred1[i][0], X_pred[i][1], c="green")
    plt.title("uniform_distribution")
    plt.show()

    return

if __name__ == '__main__':
    main()