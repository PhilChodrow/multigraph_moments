import numpy as np
from scipy.optimize import root_scalar

def degree_estimate(b):
    return((W_from_b(b).sum(axis = 1)))

def mse(b, d):
    return(((degree_estimate(b) - d)**2).mean())

def mae(b, d):
    return(np.abs((degree_estimate(b) - d)).mean())

def partial_derivative(b, i):
    
    y = b.sum()/2

    xi = b[i]*b/(2*y)
    xi[i] = 0
    ai = xi / (1-xi)**2
    V = ai.sum()
    
    dhdi = (1/b) * ai - 1/(2*y)*V
    dhdi[i] = (1/b[0] - 1/(2*y))*V
    return(dhdi)

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

def newton_round(b0, d, alpha = .01, eps = .01, require_feasible = True, feasible_value = 0):
    b_out = b0.copy()
    n = len(d)
    for i in range(n):
        b = b0.copy()
        while True:
            f, f_ = coord_obj(b,i)
            update = np.zeros(n)
            update[i] = (f-d[i])/f_
            proposal = b - alpha*update
            b = proposal
            b_out -= alpha*update
            if alpha*update[i] < eps:
                break
    return(b_out)

def built_in_round(b, d, **kwargs):
    n = len(d)
    b_new = np.zeros(n)
    for i in range(n):

        def h(x,b):
            b_ = b.copy()
            b_[i] = x
            return(coord_obj(b_, i, True))
        
        def objective(x, b):
            out = h(x,b)
            obj = d[i] - out[0]
            deriv = out[1]
            return(obj, deriv)
        
        b_ = b.copy()
        b_[i] = 0
        y_ = b_.sum()/2

        B = b_.max()
        
        bracket = (0,(2*y_)/(B - 1 + .0001))    
        
        res = root_scalar(lambda x: objective(x,b), 
                          bracket = bracket, 
                          fprime = True, 
                          x0 = b[i], 
                          **kwargs)
        b_new[i] = res.root
    return(b_new)

def symmetrize(b, d):
    for val in np.unique(d):
        b[d == val] = b[d == val].mean()
    return(b)

def compute_b(d, method = 'default', tol = 0.01, max_rounds = 100, sort = True, b0 = None, message_every = 10,message_at_end = True, **kwargs):
    
    if method == 'default':
        update = built_in_round
    else:
        update = newton_round
    
    
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
        b = update(b, d, **kwargs)
        b = symmetrize(b, d)
        obj = mse(b, d)
        if obj < tol:
            if message_at_end:
                if check_feasible(b):
                    print('Successfully converged within tolerance ' + str(tol) + ' in ' + str(i) + ' rounds, b is feasible.')
                else:
                    print('Successfully converged within tolerance ' + str(tol) + ' in ' + str(i) + ' rounds, b is NOT feasible.')
            break
        elif i > max_rounds:
            if message_at_end:
                print('Warning: convergence within tolerance ' + str(tol) + ' not reached within ' + str(max_rounds) + ' rounds. MSE = ' + str(obj))
            break
        elif i % message_every == 0:
            print('Round ' + str(i) + ': MSE = ' + str(obj))   
    return(b, mse(b, d))

def interpolator(x, b):
    y = b.sum()/2
    return((x*b/(2*y - x*b)).sum() - x**2 / (2*y - x**2))

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

def jacobian(b):
    n = len(b)
    X = X_from_b(b)
    S = X / (1-X)**2
    y = b.sum()/2
    
    B_inv = np.diag(1 / b)
    E = np.ones((n,n))
    
    D = np.diag(S.sum(axis = 0))

    J = (S + D).dot(B_inv - 1/(4*y)*E) 
    
    return(J)