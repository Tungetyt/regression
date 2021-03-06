import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def transform_matrix_with_sinusoidal_formula(x):
    return np.sin(x * 1.6 + 1.5)


def calc_error(x, y, formula_result):
    return sum((y - formula_result) ** 2) / x.size


def determine_c_and_v(x, y):
    c = np.hstack([x, np.ones(x.shape)])
    v = np.linalg.pinv(c) @ y
    return v


def main():
    a = np.loadtxt('dane3.txt')

    x = a[:, [0]]
    y = a[:, [1]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    v_lin_train = determine_c_and_v(X_train, y_train)
    e_lin_train = calc_error(X_train, y_train, v_lin_train[0] * X_train + v_lin_train[1])
    print(f'linear training error: {e_lin_train}')

    v_lin_test = determine_c_and_v(X_test, y_test)
    e_lin_test = calc_error(X_test, y_test, v_lin_test[0] * X_test + v_lin_test[1])
    print(f'linear testing error: {e_lin_test}')

    print(f'linear error difference: {e_lin_train[0] - e_lin_test[0]}')

    v_sin_train = determine_c_and_v(transform_matrix_with_sinusoidal_formula(X_train), y_train)
    e_sin_train = calc_error(X_train, y_train, v_sin_train[0] * transform_matrix_with_sinusoidal_formula(X_train) + v_sin_train[1])
    print(f'sinusoidal training error: {e_sin_train}')

    v_sin_test = determine_c_and_v(transform_matrix_with_sinusoidal_formula(X_test), y_test)
    e_sin_test = calc_error(X_test, y_test, v_sin_test[0] * transform_matrix_with_sinusoidal_formula(X_test) + v_sin_test[1])
    print(f'sinusoidal testing error: {e_sin_test}')

    print(f'sinusoidal error difference: {e_sin_train[0] - e_sin_test[0]}')

    plt.plot(x, y, 'ro')
    v_sin = determine_c_and_v(transform_matrix_with_sinusoidal_formula(x), y)
    plt.plot(x, v_sin[0] * transform_matrix_with_sinusoidal_formula(x) + v_sin[1])
    v_lin = determine_c_and_v(x, y)
    plt.plot(x, v_lin[0] * x + v_lin[1])
    plt.show()


if __name__ == '__main__':
    main()
