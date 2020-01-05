import numpy as np

def degree_estimate(b):
    return((W_from_b(b).sum(axis = 1)))

def mse(b, d):
    return(((degree_estimate(b) - d)**2).mean())

def coord_obj(b, i, return_deriv = True):
    bb = b[i]*b 
    bb[i] = 0
    y = b.sum()/2
    
    f = (bb / (2*y - bb)).sum()
    
    if return_deriv: 
        numerator = 2*y*b - bb
        numerator[i] = 0

        f_ = (numerator/((2*y-bb)**2)).sum()
        return(f, f_)
    else:
        return(f)

def newton_round(b, d, alpha = .01, eps = .01, require_feasible = True, feasible_value = 0):
    n = len(d)
    
    for i in range(n):

        while True:
            f, f_ = coord_obj(b, i)
            update = np.zeros(n)
            update[i] = (f - d[i])/f_
            b_ = b - alpha * update
            
            improvement = np.abs(f - d[i]) - np.abs(coord_obj(b_, i, False) - d[i])
            
            if (improvement > eps) and ((check_feasible(b_, feasible_value) or not require_feasible)):
                b = b_
            else:        
                break
    return(b)

def symmetrize(b, d):
    for val in np.unique(d):
        b[d == val] = b[d == val].mean()
    return(b)

def compute_b(d, alpha = 0.01, eps = 0.01, tol = 0.01, max_steps = 100, sort = True, b0 = None, message_every = 10, feasible_value = 0):
    
    # sort and keep order to return in original order provided 
    if sort:
        ord = np.argsort(d)
    else:
        ord = np.arange(len(d))
    d_ = d[ord]
    
    un_ord = np.argsort(ord)
    
    n = len(d_)
    
    # initialize first guess
    if b0 is None:
        b = np.ones(n)
    else:
        b = b0

    obj = mse(b, d)
    
    i = 0
    while True:
        i += 1
        b = newton_round(b, d, eps = eps, alpha = alpha, feasible_value = feasible_value)
        b = symmetrize(b, d)
        obj = mse(b, d)
        if obj < tol:
            print('Successfully converged within tolerance ' + str(tol) + ' in ' + str(i) + ' steps.')
            break
        elif i > max_steps:
            print('Warning: convergence within tolerance ' + str(tol) + ' not reached within ' + str(max_steps) + ' steps. MSE = ' + str(obj))
            break
        elif i % message_every == 0:
            print('Step ' + str(i) + ': MSE = ' + str(obj))   
    return(b, mse(b, d))

def X_from_b(b):
    y = 0.5*b.sum()
    X = np.outer(b, b) / (2*y)
    np.fill_diagonal(X, 0)
    return(X)

def W_from_b(b):
    X = X_from_b(b)
    W = X / (1-X)
    return(W)

def check_feasible(b, feasible_value = 0):
    largest_pair = b[np.argpartition(b, -2)[-2:]]
    nonsingular = largest_pair[0]*largest_pair[1] < b.sum()
    
    nonnegative = np.all(b >= feasible_value)
    return(nonnegative & nonsingular)