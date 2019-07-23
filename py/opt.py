import numpy as np


def fun(b, i):
    bb = b[i]*b 
    bb[i] = 0
    y = b.sum()/2
    
    f = (bb / (2*y - bb)).sum()
    
    numerator = 2*y*b - bb
    numerator[i] = 0
    
    f_ = (numerator/((2*y-bb)**2)).sum()
    return(f, f_)


def newton_round(b, d, alpha = .01, epsilon = .01):
    n = len(d)

    alpha = .01
    eps = .01
    
    for i in range(n):

        while True:
            f, f_ = fun(b, i)
            update = (f - d[i])/f_
            if np.abs(update) < eps:
                break
            b[i] -= alpha*update
    return(b)

def compute_b(d, alpha = 0.01, epsilon = 0.01, outer_epsilon = 0.001, max_steps = 10):
    '''
    Works best when d is sorted
    '''
    n = len(d)
    b0 = np.ones(n)
    b = b0
    approx = np.array([fun(b, i)[0] for i in range(n)])
    err_old = ((approx - d)**2).mean()
    k = 0
    while True:
        print('round ' + str(k) + ', current error = ' + str(round(err_old, 4)))
        b = newton_round(b, d, alpha, epsilon)
        approx = np.array([fun(b, i)[0] for i in range(n)])
        err = ((approx - d)**2).mean()
        if np.abs(err - err_old) < outer_epsilon: 
            break
        err_old = err
        k += 1
        if k > max_steps:
            break
    return b

def W_from_b(b):
    y = 0.5*b.sum()
    BB = np.outer(b, b)
    np.fill_diagonal(BB, 0)
    W = BB / (2*y - BB)
    return(W)
