# -*- coding: utf-8 -*-
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Examples from https://realpython.com/linear-regression-in-python/

def simple_linear():
    # x, the input, must be two-dimensional; one column and as many rows as necessary
    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([5, 20, 14, 32, 22, 38]).reshape((-1, 1))
    
    model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
    model.fit(x, y)
    r_sq = model.score(x, y)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    y_pred = model.intercept_ + model.coef_ * x.ravel() # vector [scalar] multiplication 
    # x.ravel() == x.reshape(-1) == x.reshape(6) == x.flatten()
    print(f"predicted response:\n{y_pred}")

    x_new = np.arange(5).reshape((-1, 1)) # just an in-order array
    y_new = model.predict(x_new)
    print(f"predicted response:\n{y_new}")

def multiple_linear():
    x = np.array([
        [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]
        ])
    y = np.array([4, 5, 20, 14, 32, 22, 38, 43])
    model = LinearRegression().fit(x, y)
    print(f"coefficient of determination: {model.score(x, y)}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1) # vector [scalar] multiplication 
    # x.ravel() == x.reshape(-1) == x.reshape(6) == x.flatten()
    print(f"predicted response:\n{y_pred}")

    x_new = np.arange(6).reshape((3, -1)) # just an in-order array
    y_new = model.predict(x_new)
    print(f"predicted response:\n{y_new}")

def polynomial():
    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([5, 20, 14, 32, 22, 38]).reshape((-1, 1))
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformer.fit(x)
    x_ = transformer.transform(x)  # now contains 2 columns--original inputs and their squared values
    print(x_)
    model = LinearRegression().fit(x_, y)

    # Or: include the intercept as part of input matrix (column of 1s). y = Ax + b = A(x w/ b included)
    x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
    model = LinearRegression(fit_intercept=False).fit(x_, y)
    
    print(f"coefficient of determination: {model.score(x_, y)}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

def mean(x):
    return np.sum(x) / len(x)

def median(x):
    return np.sort(x)[len(x) // 2]

def mode(x) -> tuple:
    d = {}
    max_val = None
    max_count = 0
    for val in x:
        if val not in d:
            d[val] = 0
        d[val] += 1
        if d[val] > max_count:
            max_val = val
            max_count = d[val]
    return max_val, max_count

def stddev(x):
    xm = mean(x)
    sum_sq_variance = 0
    for val in x:
        sum_sq_variance += (xm - val) ** 2
    return (sum_sq_variance/len(x)) ** (0.5)

def stderr(x, n):
    # x is a random sample of larger population of size n
    return stddev(x) / (n ** (0.5))


def stats_review():
    x = np.array([5, 15, 12, 25, 70, 35, 37, 45, 55, 18, 27, 27, 29, 15, 15])

    sorted_x = np.sort(x)
    x_mean = mean(x)
    x_median = median(x)
    x_mode, x_mode_count = mode(x)
    x_stddev = stddev(x)
    x_stderr = stderr(x, 100)
    print(f"x: {x}\nsorted: {sorted_x}\n mean: {x_mean}, median: {x_median}, mode: {x_mode}, mode_count: {x_mode_count}, stddev: {x_stddev}, stderr: {x_stderr}")


def main() -> int:
    # stats_review()
    simple_linear()
    polynomial()
    # multiple_linear()

if __name__ == '__main__':
    sys.exit(main())
