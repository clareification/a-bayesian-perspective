import numpy as np 
import matplotlib.pyplot as plt


def build_random_features(n=100, d=100, num_informative_features = 20):
    y = np.random.randn(n)
    X = np.random.randn(n, min(d, num_informative_features)) + y.reshape(-1,1)
    
    if d > num_informative_features:
        X = np.hstack((X, np.random.randn(n,d - num_informative_features)))
    
    return X,y

def build_gaussian_features(n=200, d=10):
    y = np.random.rand(n)
    X = np.zeros((n, d))
    for i in range(int(d/2)):
        X[:, 2*i] = y + np.random.normal(0, 0.1 * i, n)
    return X, y

def build_quadratic_fn(n=100, xmax=10):
    x = xmax*np.random.rand(n)
    y = x **2
    return x, y

def build_oned_features(n=200, d=10):
    y = np.random.rand(n)
    X = np.random.rand(n,d)
    X[:, 0] = y
    return X, y

def marginal_likelihood(x, y, n=200, l=1.0, prior=1.0):
    # Assume default prior for the moment?
    prior_sigma = prior
    N = x.shape[0]
    S0 = prior_sigma * np.eye(x.shape[1])
    SN = np.linalg.inv(np.linalg.inv(S0) + 1/l * x.T@x)
    mN = SN @ np.dot(x.T,y)/l
    log = -N/2*np.log(2*np.pi) - N/2*np.log(l)
    log = log - 1/2*( stable_log_determinant(S0) - stable_log_determinant(SN))
    #print(log, ' unstable', np.log(np.linalg.det(SN )), 'stable:', stable_log_determinant(SN))
    log = log - 1/2 * np.dot(y,y)/l

    log = log + 1/2 * mN@ np.linalg.inv(SN) @mN
    return log 

def stable_log_determinant(m):
    eigs = np.linalg.eigvals(m)
    #print('eigs', eigs[-10:])
    ls = np.log(eigs)
    #print('log sum', np.sum(ls))
    return np.sum(ls)

# def posterior_dist(prior_sigma, X, y, noise_sigma):


def train_one_epoch_iterative(w, X, y, num_steps, step_size=0.001, log_interval=1000, noise_sigma=1.0, prior_sigma=0.1):
    ''' num_steps: number of gradient steps per datum
        step_size: gradient descent step size
        w: w init 
    '''
    loss_gd = []
    count = 0
    x = X
    t = y
    g=100*np.linalg.norm(y)

    while(np.linalg.norm(g) > 0.01):
        count += 1
        d = (x @ w - t)/ np.linalg.norm(y)
        #print(np.linalg.norm(d))
        loss_gd.append(d)
        g = d @ x + w /np.linalg.norm(y) * 1/prior_sigma 
        wold = w
        w = w - step_size * g
        if not count % log_interval:
            print('abs error: ', np.linalg.norm(d), np.linalg.norm(w))
            print(np.linalg.norm(g), 'norm diff', np.linalg.norm(w - wold))
        
    return w, loss_gd


def get_posterior_samples(x, y, prior_sigma=1.0, l=0.01):
    N = x.shape[0]
    # Posterior covariance
    S0 = prior_sigma * np.eye(x.shape[1])
    SN = np.linalg.inv(np.linalg.inv(S0) + 1/l * x.T@x)
    # Posterior mean
    mN = SN @ np.dot(x.T,y)/l
    # Return a function that generates samples from the posterior.
    return lambda : np.random.multivariate_normal(mN, SN)

def get_posterior_mean(x, y, prior_sigma=1.0):
    l=0.01
    N = x.shape[0]
    S0 = prior_sigma * np.eye(x.shape[1])
    SN = np.linalg.inv(np.linalg.inv(S0) + 1/l * x.T@x)
    mN = SN @ np.dot(x.T,y)/l
    return lambda : mN

def iterative_estimator(xtrain, ytrain, xtest, ytest, l=1.0, k=10, prior_sigma=0.1):
  n = len(xtrain)
  test_errors = [0]
  mean_errors = [0]
  sample_errors = [0]
  sample_vars = [0]
  for i in range(n-1):
    trains = xtrain[:i]
    train_ys = ytrain[:i]
    w = np.linalg.lstsq(trains, train_ys, rcond=None)[0]
    err = np.linalg.norm(xtrain[i+1] @ w - ytrain[i+1])/l
    mean_errors.append(err)
    sampler = get_posterior_samples(trains, train_ys, prior_sigma, l)
    
    samples= []
    for _ in range(k): 
        w = sampler()
        samples.append(np.linalg.norm(xtrain[i+1]@ w - ytrain[i+1])/l)
    sample_err = np.mean(samples)
    test_err = np.linalg.norm(xtest @ w - ytest)/np.linalg.norm(ytest)
    test_errors.append(test_err)
    sample_var = np.var(samples)
    sample_errors.append(sample_err)
    sample_vars.append(sample_var)
    
  
  return test_errors, mean_errors, sample_errors, sample_vars
 

def sample_then_optimize(prior_sampler, xtrain, ytrain, l=0.1, k=1):
    w = prior_sampler()
    prior_sigma = 1000 #1/(xtrain.shape[1])
    print(prior_sigma) # close enough lol
    noise_sigma = l
    n = len(xtrain)
    d = xtrain.shape[1]
    ls = []
    dists = []
    opts = []
    ws = [w]
    winit = w.copy()
    a = winit[0]
    for i in range(n-1):
        l = (w @ xtrain[i] - ytrain[i])**2/noise_sigma + np.linalg.norm(w)**2/prior_sigma#/np.linalg.norm(ytrain)
        wopt = np.linalg.lstsq(xtrain[:i], ytrain[:i])[0]
        lopt = np.abs(wopt @ xtrain[i] - ytrain[i])**2/noise_sigma
        opts.append(lopt)
        #print('curr loss: ', np.abs(l), 'opt',  lopt, np.linalg.norm(a - w[0]))
        ls.append(l)
        for _ in range(k):
            
            dist = (np.linalg.norm(w - wopt))
            wnew, epoch_l = train_one_epoch_iterative(w, xtrain[:i+1], ytrain[:i+1], 50, prior_sigma=prior_sigma)
            
            w = wnew
        #print('distance from optimal: ', dist)
        ws.append(w)
        dists.append(dist)
    return ls, dists, opts, ws

def generate_lb_ml_plot():
    n = 200
    d=100
    x, y = build_random_features(d=d, n=n)
    x2, y2 = build_random_features(d=d)
    te, me, se, sv = iterative_estimator(x, y, x2, y2)
    true_mls = []
    for i in range(n):
        true_mls.append(marginal_likelihood(x[:i], y[:i]))
    true_mls[0] = 0
    deltas = [-true_mls[i+1] + true_mls[i] for i in range(n-1)]
    
    se = np.array(se)
    sv = np.array(sv)
    plt.plot(te, label='test errs', color='red')
    plt.plot(me, alpha=0.5, label='mean errs')
    plt.plot(se + np.log(2*np.pi), color='green', label='sample loglik')
    sample_ml = [-1*sum(se[:i]) for i in range(n)]
    sample_ml_var = [1/(i+1) * sum(sv[:i]) for i in range(n)]
    plt.fill_between(range(n), se + np.sqrt(sv) + np.log(2*np.pi), se-np.sqrt(sv) + np.log(2*np.pi),color='green', alpha=0.2)
    #plt.plot(sample_ml, color='green', label='sample errs')

    # plt.fill_between(range(len(se)), sample_ml - np.sqrt(sample_ml_var), sample_ml + np.sqrt(sample_ml_var),
    #              color='green', alpha=0.2)
    plt.plot(deltas, color='blue', label='ml deltas')
    # plt.ylim(0, 10)
    plt.axvline(x=d + 1, color='black', linestyle='-.')
    # Plot marginal likelihood corrected for constant factors
    #plt.plot(np.array(true_mls) - np.array([- np.log(2*np.pi)  * i/2 for i in range(n)]))
    plt.title('Change in evidence as a function of number of data points')
    plt.legend()

    plt.show()

def sample_opt_plot():
    d=50
    n=100
    x, y = build_random_features(d=d, n=n)
    x2, y2 = build_random_features(d=d)
    te, me, se, sv = iterative_estimator(x, y, x2, y2)
    #plt.plot(se + np.log(2*np.pi), color='green', label='sample loglik')
    sample_ml = [-1*sum(se[:i]) for i in range(n)]
    sample_ml_var = [1/(i+1) * sum(sv[:i]) for i in range(n)]
    sample_ml = [-1*sum(se[:i]) for i in range(n)]
    plt.plot(sample_ml, label='sample ml sigma=1.0')
    for i in range(1):
        ls , ds, opts, ws = sample_then_optimize(lambda : np.random.normal(np.zeros(d), 1/d), x, y, l=1.0)
        # plt.plot(ds, label='dist from lstsq solution')
        # plt.plot(np.abs(ls), label='loss from sgd')
        
        # plt.legend()
        # plt.show()
        # print(ws[0].shape)
        # print(len(ws))
        # [plt.scatter(range(len(we)), we, alpha=0.2 + 0.5*(i/len(ws)), color='green') for i, we in enumerate(ws)]
        # plt.show()
        estim_mls = [-1*(np.sum(ls[:i])) for i in range( n)]
        print(ls)
        plt.plot(estim_mls, label='seed ' + str(i))

    plt.plot([-1*sum(opts[:i]) for i in range(n)], label='estim from lstsq solution')
    
    for sigma in [0.1, 0.01, 0.001, 1.0]:
        true_mls = []
        for i in range(n):
            true_mls.append(marginal_likelihood(x[:i], y[:i], l=sigma)  + i/2 * np.log(sigma*2*np.pi))
        
        plt.plot(true_mls, label='true, sigma=' + str(sigma))
    plt.ylim(min(np.min(estim_mls), np.min(sample_ml)), 4)
    plt.legend()
    plt.show()
    plt.plot(ls, label='sgd losses')
    plt.plot(opts, label='lstsq losses')
    plt.legend()
    plt.show()

    plt.plot([true_mls[i+1] - true_mls[i] for i in range(n-1)], label='true ml deltas')
    plt.plot(ls, label='sample-then-optimize deltas')
    plt.plot(se, label='posterior sampling deltas')
    plt.ylim(-10, 10)
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    print('Welcome, you diligent and resourceful scientist!')
    #generate_lb_ml_plot()
    sample_opt_plot()

''' want to answer the following questions....

1. 

Using the following parameters:
Features: RFF model (?)
Predictions: first try with posterior means, then try sampling from posterior
Loglik function: || wTX - y||^2/sigma2 for some noise variance

''' 

