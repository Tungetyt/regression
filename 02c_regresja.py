import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# a = np.loadtxt('Sharp_char.txt')
def sinusoidal_equation(x):
    return np.sin(x * 1.6 + 1.5) * 80 + 10

# def regression(x, y, formula):
#     c = np.hstack([x, formula])
#     v = np.linalg.pinv(c) @ y
#     return v, sum((y - (v[0] * x + v[1])) ** 2) / x.size
def calc_error(x, y, v, equation_result):
    return sum((y - (v[0] * equation_result + v[1])) ** 2) / x.size


def main():

    a = np.loadtxt('dane3.txt')

    x = a[:, [0]]
    y = a[:, [1]]

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    c = np.hstack([x * x * x, x * x, x, np.ones(x.shape)])
    v = np.linalg.pinv(c) @ y

    # c1 = np.hstack([1 / x, np.ones(x.shape)])
    # v1 = np.linalg.pinv(c1) @ y

    c2 = np.hstack([x, np.ones(x.shape)])
    v2 = np.linalg.pinv(c2) @ y

    c2train = np.hstack([X_train, np.ones(X_train.shape)])
    v2train = np.linalg.pinv(c2train) @ y_train
    # e2train = sum((y_train - (v2train[0] * X_train + v2train[1])) ** 2) / X_train.size
    e2train = calc_error(X_train, y_train, v2train, X_train)
    print(f'linear training error: {e2train}')

    c2test = np.hstack([X_test, np.ones(X_test.shape)])
    v2test = np.linalg.pinv(c2test) @ y_test
    # e2test = sum((y_test - (v2test[0] * X_test + v2test[1])) ** 2) / X_test.size
    e2test = calc_error(X_test, y_test, v2test, X_test)

    print(f'linear testing error: {e2test}')

    print(f'linear error difference: {e2train[0] - e2test[0]}')

    c3 = np.hstack([sinusoidal_equation(x), np.ones(x.shape)])
    v3 = np.linalg.pinv(c3) @ y

    c3train = np.hstack([sinusoidal_equation(X_train), np.ones(X_train.shape)])
    v3train = np.linalg.pinv(c3train) @ y_train
    # e3train = sum((y_train - (v3train[0] * sinusoidal_equation(X_train) + v3train[1])) ** 2) / X_train.size
    e3train = calc_error(X_train, y_train, v3train, sinusoidal_equation(X_train))
    print(f'sinusoidal training error: {e3train}')
    # print(f'sinusoidal training error: {calc_error(X_train, y_train, v3train, sinusoidal_equation(X_train))}')

    c3test = np.hstack([sinusoidal_equation(X_test), np.ones(X_test.shape)])
    v3test = np.linalg.pinv(c3test) @ y_test
    # e3test = sum((y_test - (v3test[0] * sinusoidal_equation(X_test) + v3test[1])) ** 2) / X_test.size
    e3test = calc_error(X_test, y_test, v3test, sinusoidal_equation(X_test))
    print(f'sinusoidal testing error: {e3test}')

    print(f'sinusoidal error difference: {e3train[0] - e3test[0]}')

    plt.plot(x, y, 'ro')
    plt.plot(x, v3[0] * sinusoidal_equation(x) + v3[1])
    plt.plot(x, v2[0] * x + v2[1])
    plt.show()



if __name__ == '__main__':
    main()
