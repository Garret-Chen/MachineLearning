import numpy as np


def prob(beta, X):
    return np.exp(np.matmul(beta.T, X)) / (1 + np.exp(np.matmul(beta.T, X)))  # (1,150)


def train_Newton(epoch, beta, X, Y, X_train, Y_train):
    for i in range(epoch):
        temp1 = np.matrix(Y) - prob(beta, X)  # (1,150)
        # temp2 = prob(beta, X) * (np.ones(150).reshape(1, -1) - prob(beta, X))  # 对位相乘 (1,150)
        temp2 = prob(beta, X) * (np.ones(len(Y)).reshape(1, -1) - prob(beta, X))  # 对位相乘 (1,150)
        first = 0
        second = 0
        for ii in range(Y.shape[1]):
            first -= np.matrix(X)[:, ii] * temp1[:, ii]  # (5,1)
            second += np.matmul(X, X.T) * temp2[:, ii][0]  # (5,5)
        beta = beta - np.matmul(np.linalg.pinv(second), first)

        if isinstance(beta, np.matrix):
            beta = beta.A
        if i == epoch - 1:
            print('EPOCH', i, '>>accuracy:', predict_ALL(beta, X_train, Y_train))
    return beta


def predict_ALL(beta, X, Y):
    target = 1 / (1 + np.exp(-np.matmul(beta.T, X)))
    error = np.mean(np.abs(target - Y))
    accuracy = 1 - error
    return accuracy


def predict_one(beta, x):
    target = 1 / (1 + np.exp(-np.matmul(beta.T, x)))
    return target


if __name__ == '__main__':
    p = r'iris.csv'
    with open(p, encoding='utf-8') as f:
        origin = np.loadtxt(f, str, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4, 6))
        origin = origin.astype(float)
        data = origin[:, :4]
        Y = origin[:, -1].reshape(-1, 1).T  # (1,150)

    # nomalization
    X = (data - np.mean(data, axis=1).reshape(-1, 1)) / np.std(data, axis=1).reshape(-1, 1)
    X1 = np.c_[X, np.ones((150,))].T
    beta = np.random.uniform(-2, 2, (5, 1))
    epoch = 400

    # 十折交叉验证
    # all_res1 = []
    # for i in range(0, 150, 15):
    #     print("第{}次交叉验证...".format(i // 15 + 1))
    #     X_ = np.concatenate([X1[:, :i], X1[:, i+15:]], axis=1)
    #     Y_ = np.concatenate([Y[:, :i], Y[:, i+15:]], axis=1)
    #     beta_result = train_Newton(epoch, beta, X_, Y_, X1[:, i:i+15], Y[:, i:i+15])
    #     all_res1.append(beta_result)
    #
    # res = sum(all_res1) / len(all_res1)
    # # print("十折交叉验证法的平均正确率为%f" % res, end=' ')
    # print(res)

    # 留一法
    all_res2 = [0 for i in range(150)]
    for i in range(0, 150):
        print("正在进行第{}次迭代...".format(i+1))
        X_ = np.concatenate([X1[:, :i], X1[:, i+1:]], axis=1)
        Y_ = np.concatenate([Y[:, :i], Y[:, i+1:]], axis=1)
        beta_result = train_Newton(epoch, beta, X_, Y_, X1[:, i], Y[:, i])
        all_res2[i] = beta_result
    res = sum(all_res2) / 150
    # print("留一法的平均正确率为%f" % res, end=' ')
    # print(res)