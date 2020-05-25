import numpy as np 
import matplotlib.pyplot as plt
import torch
import seaborn as sns
class BLRModel():
    def __init__(self, prior_sigma, noise_sigma, feature_map):
        self.prior_sigma = prior_sigma
        self.noise_sigma = noise_sigma 
        self.feature_map = feature_map
    
    def posterior_weight_sampler(self, x, y):
        phi = self.feature_map(x)
        sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma)
        return sampler

    def posterior_pred_sample(self, x, y, xtest):
        phi = self.feature_map(x)
        if len(x) == 0:
            sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma, N=len(xtest))
        else:
            sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma)
        w = sampler() 
        return w @ self.feature_map(xtest).T
    
    def get_marginal_likelihood(self, x, y):
        phi = self.feature_map(x)
        n = len(x)
        ml = marginal_likelihood(phi, y, n, self.noise_sigma, self.prior_sigma)
        return ml

    def get_elbo(self, x, y, custom_noise=False):
        phi = self.feature_map(x)
        noise = 1.0 if custom_noise else self.noise_sigma
        return iterative_estimator(phi, y, phi, y, prior_sigma=self.prior_sigma, l=noise)

def build_random_features(n=100, d=100, num_informative_features = 50, rand_variance=1.0):
    y = np.random.randn(n)
    X = 1/rand_variance *np.random.randn(n, min(d, num_informative_features)) + y.reshape(-1,1)
    
    if d > num_informative_features:
        X = np.hstack((X, rand_variance * np.random.randn(n,d - num_informative_features)))
    
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


def train_one_epoch_iterative(w, X, y, num_steps, step_size=0.0001, log_interval=10000, noise_sigma=1.0, prior_sigma=0.1):
    ''' num_steps: number of gradient steps per datum
        step_size: gradient descent step size
        w: w init 
    '''
    loss_gd = []
    count = 0
    x = X
    t = y
    g=100*np.linalg.norm(y)

    while(np.linalg.norm(g) > 0.01 and count < num_steps):
        count += 1
        d = (x @ w - t)/ np.linalg.norm(y)
        #print(np.linalg.norm(d))
        loss_gd.append(d)
        g = d @ x #+ w /np.linalg.norm(y) * 1/prior_sigma 
        wold = w
        w = w - step_size * g
        # if not count % log_interval*100:
        #     print('abs error: ', np.linalg.norm(d), np.linalg.norm(w))
        #     print(np.linalg.norm(g), 'norm diff', np.linalg.norm(w - wold))
        
    return w, loss_gd


def get_posterior_samples(x, y, prior_sigma=1.0, l=0.01, N=None):
    
    # Posterior covariance
    if len(x) > 0:
        S0 = prior_sigma * np.eye(x.shape[1])
        SN = np.linalg.inv(np.linalg.inv(S0) + 1/l * x.T@x)
        # Posterior mean
        mN = SN @ np.dot(x.T,y)/l
    else:
        mN = np.zeros(N)
        S0 = prior_sigma * np.eye(N)
        SN = S0
    sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(mN), torch.tensor(SN))
    # Return a function that generates samples from the posterior.
    return lambda : sampler.sample().cpu().numpy()

def get_posterior_mean(x, y, prior_sigma=1.0, l=0.01):
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
    err = - ( np.linalg.norm(xtrain[i+1] @ w - ytrain[i+1]))**2/(2*l) - 1/2 * np.log(2 * np.pi * l)
    mean_errors.append(err)
    sampler = get_posterior_samples(trains, train_ys, prior_sigma, l, N=xtrain.shape[1])
    
    samples= []
    for _ in range(k): 
        w = sampler()
        samples.append(-np.linalg.norm(xtrain[i+1]@ w - ytrain[i+1])**2/(2*l) - 1/2* np.log(2*np.pi*l))
    sample_err = np.mean(samples)
    test_err = np.linalg.norm(xtest @ w - ytest)/np.linalg.norm(ytest)
    test_errors.append(test_err)
    sample_var = np.var(samples)
    sample_errors.append(sample_err)
    sample_vars.append(sample_var)
    
  
  return test_errors, mean_errors, sample_errors, sample_vars
 
def sample_then_optimize(prior_sampler, xtrain, ytrain, l=1.0, k=1):
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
        l =- (w @ xtrain[i] - ytrain[i])**2/(2*noise_sigma) - 1/2 * np.log(np.pi * 2 * noise_sigma) #/np.linalg.norm(ytrain)
        #print(l, (w @ xtrain[i] - ytrain[i])**2/(2*noise_sigma), 1/2 * np.log(np.pi * 2 * noise_sigma))
        wopt = np.linalg.lstsq(xtrain[:i], ytrain[:i])[0]
        lopt = np.abs(wopt @ xtrain[i] - ytrain[i])**2/noise_sigma
        opts.append(lopt)
        #print('curr loss: ', np.abs(l), 'opt',  lopt, np.linalg.norm(a - w[0]))
        ls.append(l)
        for _ in range(k):
            
            dist = (np.linalg.norm(w - wopt))
            wnew, epoch_l = train_one_epoch_iterative(w, xtrain[:i+1], ytrain[:i+1], 500, prior_sigma=0.)
            
            w = wnew
        #print('distance from optimal: ', dist)
        ws.append(w)
        dists.append(dist)
    return ls, dists, opts, ws

# Figures 1a and 1b
def generate_lb_ml_plot(prior_sigma = 1.0, noise_var = 1.0):
    n = 200
    d= 100
    
    
    k = 5
    x, y = build_random_features(d=d, n=n)
    x2, y2 = build_random_features(d=d)
    te, me, se, sv = iterative_estimator(x, y, x2, y2, k=k, l=noise_var, prior_sigma=prior_sigma)
    sto_means = []
    sto_vars = []
    sto_ws = []
    for _ in range(k):
        ls , ds, opts, ws = sample_then_optimize(lambda : np.random.normal(np.zeros(d), prior_sigma), x, y, l=noise_var)
        sto_ws.append(ls)
    sto_means = np.mean(np.array(sto_ws), axis=0)
    sto_vars = np.var(np.array(sto_ws), axis=0)

    true_mls = []
    for i in range(n):
        true_mls.append(marginal_likelihood(x[:i], y[:i], l=noise_var, prior=prior_sigma))
    true_mls[0] = 0
    deltas = [true_mls[i+1] - true_mls[i] for i in range(n-1)]
    
    se = np.array(se)
    sv = np.array(sv)
    #plt.plot(te, label='test errs', color='red')
    #plt.plot(me, alpha=0.5, label='mean errs')
    
    # Plot change in ML
    plt.plot(se, color='green', label='Exact sampling ELBO')
    sto_ml =  np.array([sum(sto_means[:i]) for i in range(n)])
    sto_var = np.array( [1/(i+1) * sum(sto_vars[:i]) for i in range(n)])
    
    sample_ml = [1*sum(se[:i]) for i in range(n)]
    sample_ml_var = [1/(i+1) * sum(sv[:i]) for i in range(n)]
    plt.fill_between(range(n), se + np.sqrt(sv), se-np.sqrt(sv),color='green', alpha=0.2)
    plt.fill_between(range(n-1), sto_means + np.sqrt(sto_vars), sto_means-np.sqrt(sto_vars),color='orange', alpha=0.2)
    plt.plot(sto_means, color='orange', label='Sample-then-optimize ELBO')
    #sto_var = [0] + sto_vars
    plt.plot(deltas, color='blue', label='Log Evidence')
    # plt.ylim(0, 10)
    plt.axvline(x=d + 1, color='black', linestyle='-.')
    plt.title('Change in estimators after seeing a new data point \n (prior=' + str(prior_sigma) + ")")
    plt.xlabel("Index of new data point")
    plt.ylabel("Change in estimator after seeing data point k")
    plt.legend()

    plt.savefig('dummy_deltas.png')
    plt.clf()

    # Plot marginal likelihood corrected for constant factors
    plt.plot(sample_ml, color='green', label='Exact Sampling ELBO')
    plt.plot(np.array(true_mls), label='Log Evidence')
    plt.fill_between(range(len(se)), sample_ml - np.sqrt(sample_ml_var), sample_ml + np.sqrt(sample_ml_var),
                 color='green', alpha=0.2)
    plt.plot(sto_ml, color='orange', label='Sample-then-optimize ELBO')
    plt.fill_between(range(n), sto_ml + np.sqrt(sto_var), sto_ml - np.sqrt(sto_var),color='orange', alpha=0.2)       
    print(sto_var.shape, sto_ml.shape, np.min(sto_var), np.max(sto_var))
    plt.ylabel('Likelihood')
    plt.xlabel('Number of data points used in estimate')
    plt.title('Illustration of Marginal Likelihood Estimators for Linear Regression \n (prior=' + str(prior_sigma) + ")")
    plt.legend()
    
    plt.savefig('dummy.png')
    
def sample_opt_plot():
    d=100
    n=200
    prior_sigma = 1.0
    x, y = build_random_features(d=d, n=n)
    x2, y2 = build_random_features(d=d)
    te, me, se, sv = iterative_estimator(x, y, x2, y2, prior_sigma=prior_sigma)
    print('done iterative estimator')
    #plt.plot(se + np.log(2*np.pi), color='green', label='sample loglik')
    sample_ml = [-1*sum(se[:i]) for i in range(n)]
    sample_ml_var = [1/(i+1) * sum(sv[:i]) for i in range(n)]
    sample_ml = [-1*sum(se[:i]) for i in range(n)]
    plt.plot(sample_ml, label='sample ml sigma=1.0')
    for i in range(1):
        ls , ds, opts, ws = sample_then_optimize(lambda : np.random.normal(np.zeros(d), 1.0 ), x, y, l=1.0)
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
    plt.savefig('sammpleopt.png')
    plt.clf()
    plt.plot([true_mls[i+1] - true_mls[i] for i in range(n-1)], label='true ml deltas')
    plt.plot(ls, label='sample-then-optimize deltas')
    plt.plot(se, label='posterior sampling deltas')
    plt.ylim(np.min(estim_mls), 10)
    plt.legend()
    
    plt.savefig('sampleto.png')

def model_selection_plot():
    '''
    Generate a figure showing
    '''
    torch.manual_seed(0)
    np.random.seed(0)

    marg_liks = []
    lbs = []
    model_ls = []
    sto_lbs = []
    d = 30
    d_inf = 15

    # One: construct elbo and ml for each scale
    n_models = 6
    
    lengthscales=[2**(-i ) for i in range(5,n_models+5)]
    num_features = [(i+1)*(int(d/n_models)) for i in range(n_models)]
    xtrain, ytrain = build_random_features(n=2*d, d=d, num_informative_features=d_inf, rand_variance=1.0)

    for _ in range(5):
        linear_models = [BLRModel(1/k**2, 1/d_inf, lambda x : x) for k in num_features]

        marg_liks.append([m.get_marginal_likelihood(xtrain[:, :k], ytrain) for m, k in zip(linear_models, num_features)])
        # print('done ml')
        lbs.append([np.sum(m.get_elbo(xtrain[:, :k], ytrain, custom_noise=False)[2]) for m, k in zip(linear_models, num_features)])
        sto_results = [sample_then_optimize(lambda : np.random.normal(np.zeros(k), 1/k), xtrain[:, :k], ytrain, l=0.1) for k in num_features]
        sto_lbs.append([np.sum(s[0]) for s in sto_results])
    # print(sto_lbs)

    plt.clf()
    plt.xlabel('Number of features (x5)')
    plt.ylabel('Negative likelihood')
    #print(len(num_features), len(marg_liks))
    #ax1.set_xlabels(lengthscales)
    sns.tsplot( marg_liks, condition='log evidence', )
    sns.tsplot(lbs, condition='elbo', color='green', )
    print(np.argmax(lbs[0]), np.argmax(marg_liks[0]), np.argmax(sto_lbs[0]))
    sns.tsplot(sto_lbs, condition='sample then optimize', color='purple')
    plt.title('Selecting Number of Features for Bayesian Linear Regression')
    plt.tight_layout()
    plt.savefig('feature_dim_selection.png')
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None


if __name__ == '__main__':
    print('Welcome, you diligent and resourceful scientist!')
    #generate_lb_ml_plot(prior_sigma=1., noise_var=0.1)
    model_selection_plot()
    print('sample then opt')
    #sample_opt_plot()

''' want to answer the following questions....

1. 

Using the following parameters:
Features: RFF model (?)
Predictions: first try with posterior means, then try sampling from posterior
Loglik function: || wTX - y||^2/sigma2 for some noise variance

''' 

