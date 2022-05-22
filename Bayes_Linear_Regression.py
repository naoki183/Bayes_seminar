
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as LA

def gen_dataset(N, function):
    x = np.random.rand(N)
    if function == "sin":
        y = np.sin(2 * math.pi * x)
        return (x, y)
    elif function == "step":
        y = x * 10 // 1
        return (x, y)

def base_function(x, M, type):
    if type == "polynomial":
        y = np.zeros(M)
        for i in range(M):
            y[i] = math.pow(x, i)
        return y

def plot_pred_function(_Lambda, _m, _lambda, M, N, base_function_type, x, y):
    X = np.zeros((N, M))
    for i in range(N):
        X[i, :] = base_function(x[i], M, base_function_type)
    s = np.zeros((M, M))
    for i in range(N):
        s += X[i:i+1, :].T @ X[i:i+1, :]
    

    Lambda_ = _lambda * s + _Lambda
    t = np.zeros(M)
    for i in range(N):
        t += y[i] * X[i, :]
    m_ = np.linalg.inv(Lambda_) @ (_lambda * t + _Lambda @ _m)
    
    test_x = np.arange(0, 1, 0.01)
    test_y = np.sin(test_x * 2 * math.pi)
    pred_y_mu = np.zeros(100)
    pred_y_sigma = np.zeros(100)
    for i in range(len(test_x)):
        pred_y_mu[i] = m_.T @ base_function(test_x[i], M, base_function_type)
        pred_y_sigma[i] = 1 / np.sqrt(1 / _lambda + base_function(test_x[i], M, base_function_type).T @ (np.linalg.inv(Lambda_) @ base_function(test_x[i], M, base_function_type)))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_title(f"{M}dimension")
    # ax.plot(test_x, test_y, linestyle="solid", label="true function", color="red")
    # ax.plot(test_x, pred_y_mu, linestyle="solid", color="blue")
    # ax.plot(test_x, pred_y_mu - pred_y_sigma, linestyle="dashed", color="blue")
    # ax.plot(test_x, pred_y_mu + pred_y_sigma, linestyle="dashed", color="blue")
    # ax.legend()
    # plt.show()
    return [_m, _Lambda, m_, Lambda_, _lambda, y]

def calc_marginal_likelihood(_m, _Lambda, m_, Lambda_, _lambda, y):
    s = 0
    for i in range(len(y)):
        s += (_lambda / 2) * y[i] * y[i] - (1 / 2) * math.log(_lambda / (2 * math.pi))
    
    return -(1 / 2) * _m @ (_Lambda @ _m) + (1 / 2) * m_ @ (Lambda_ @ m_) - s + (1 / 2) * math.log(LA.det(_Lambda)) - (1 / 2) * math.log(LA.det(Lambda_))

def calc_generalization_error(m_, M, base_function_type):
    test_size = 100000
    test_x = np.random.rand(test_size)
    test_y = np.sin(2 * math.pi * test_x)
    pred_y = np.zeros(test_size)
    error = 0
    for i in range(test_size):
        pred_y[i] = m_.T @ base_function(test_x[i], M, base_function_type)
        error += (test_y[i] - pred_y[i]) ** 2
    error /= test_size
    return error


def main():
    # 学習データ数
    N = 1000
    # 回帰対象の関数
    regression_function_type = "sin"

    x, y = gen_dataset(N, regression_function_type)
    l_list = []
    error_list = []
    for i in range(2, 20, 1):
        # ハイパーパラメータ(_Lambda, _mの引数の項数とMが一致していないとエラー出るので注意)
        _Lambda = np.linalg.inv(np.cov(np.random.rand(i, 100)))
        _m = np.random.rand(i)
        _lambda = i
        # 基底関数の次元
        M = i
        # 基底関数
        base_function_type = "polynomial"

        _m, _Lambda, m_, Lambda_, _lambda, y = plot_pred_function(_Lambda, _m, _lambda, M, N, base_function_type, x, y)
        l = calc_marginal_likelihood(_m, _Lambda, m_, Lambda_, _lambda, y)
        error = calc_generalization_error(m_, M, base_function_type)
        l_list.append(-l)
        error_list.append(error * 10)
    error1_list = []
    for j in range(2, 20, 1):
        M = j
        base_function_type = "polynomial"
        x, y = gen_dataset(N, regression_function_type)
        X = np.zeros((N, M))
        for i in range(N):
            X[i, :] = base_function(x[i], M, base_function_type)
    
        theta = (np.linalg.inv(X.T @ X) @ X.T) @ y
        x_test, y_test = gen_dataset(100000, regression_function_type)
        error1 = 0
        for i in range(100000):
            y_pred = theta.T @ base_function(x_test[i], M, base_function_type)
            error1 += (y_test[i] - y_pred) ** 2
        error1 /= 100000
        error1_list.append(error1 * 10)



    fig = plt.figure()

    ax = fig.add_subplot(111)
    # ax.plot(np.arange(2, 50, 1), l_list, linestyle="solid", label="free_energy", color="red")
    ax.plot(np.arange(2, 20, 1), error_list, linestyle="solid", label="generalization_error(Bayes)", color="blue")
    ax.plot(np.arange(2, 20, 1), error1_list, linestyle="solid", label="generalization_error(Non-Bayes)", color="green")
    ax.legend()
    plt.show()

    return

def main1():
    # 学習データ数
    N = 100000
    # 回帰対象の関数
    regression_function_type = "sin"
    M = 10
    base_function_type = "polynomial"

    x, y = gen_dataset(N, regression_function_type)
    X = np.zeros((N, M))
    for i in range(N):
        X[i, :] = base_function(x[i], M, base_function_type)
    
    theta = (np.linalg.inv(X.T @ X) @ X.T) @ y




if __name__ == '__main__':
    main()
