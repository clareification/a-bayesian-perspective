import numpy as np 
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy.special import logsumexp 
import pickle as pkl

class BLRModel():
    def __init__(self, prior_sigma, noise_sigma, feature_map):
        self.prior_sigma = prior_sigma
        self.noise_sigma = noise_sigma 
        self.feature_map = feature_map
    
    def posterior_weight_sampler(self, x, y, d=None):
        phi = self.feature_map(x)
        sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma, d=d)
        return sampler

    def posterior_pred_sample(self, x, y, xtest, d=None):
        phi = self.feature_map(x)
        dim = d if d is not None else self.feature_map(x).shape[1]
        if len(x) == 0:
            sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma, d=dim)
        else:
            sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma)
        w = sampler() 
        return w @ self.feature_map(xtest).T
    
    def get_marginal_likelihood(self, x, y):
        phi = self.feature_map(x)
        n = len(x)
        ml = marginal_likelihood(phi, y, n, l=self.noise_sigma, prior=self.prior_sigma)
        return ml

    def get_elbo(self, x, y, custom_noise=False, n_samples=10, lse=False, integrate_over_params=True):
        phi = self.feature_map(x)
        noise = 1.0 if custom_noise else self.noise_sigma
        return iterative_estimator(phi, y, phi, y, prior_sigma=self.prior_sigma, l=noise, k=n_samples, lse=lse, integrate_over_params=integrate_over_params)

    def get_posterior_mean_and_var(self, x, y, d=None):   
        if len(x) > 0:
            S0 = self.prior_sigma * np.eye(x.shape[1])
            SN = np.linalg.inv(np.linalg.inv(S0) + 1/self.noise_sigma * x.T@x)
            # Posterior mean
            mN = SN @ np.dot(x.T,y)/self.noise_sigma
        else:
            mN = np.zeros(d)
            S0 = self.prior_sigma * np.eye(d)
            SN = S0
        return mN, SN
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
    # Formula taken from http://www.utstat.utoronto.ca/~radford/sta414.S11/week4a.pdf
    # Assume diagonal prior
    prior_sigma = prior
    N = x.shape[0]
    S0 = prior_sigma * np.eye(x.shape[1])
    SN = np.linalg.inv(np.linalg.inv(S0) + 1/l * x.T@x)
    
    mN = SN @  x.T @ y/l
    log = -N/2*np.log(2*np.pi) - N/2*np.log(l)
    log = log - 1/2*( stable_log_determinant(S0) - stable_log_determinant(SN))
    log = log - 1/2 * np.dot(y,y)/l

    log = log + 1/2 * mN.T @ np.linalg.inv(SN) @mN
    return log 

def stable_log_determinant(m):
    eigs = np.linalg.eigvals(m)
    ls = np.log(eigs)
    return np.sum(ls)

# def posterior_dist(prior_sigma, X, y, noise_sigma):


def train_one_epoch_iterative(w, X, y, num_steps, step_size=0.0001, log_interval=10000, noise_sigma=0.0, prior_sigma=0.1, w0=None):
    ''' num_steps: number of gradient steps per datum
        step_size: gradient descent step size
        w: w init 
    '''
    loss_gd = []
    count = 0
    x = X
    t = y
    g=100*np.linalg.norm(y)
    # In noiseless setting, suffices to just run GD on 
    # \|x@w - y\|^2
    if noise_sigma == 0.:
        while(np.linalg.norm(g) > 0.01 and count < num_steps):
            count += 1

            d = (x @ w - t)/ np.linalg.norm(y) # normalize gradient stepsize so that we don't diverge.
            loss_gd.append(d)
            g = d @ x 
            wold = w
            w = w - step_size * g
            
        return w, loss_gd
    
    # Otherwise, we'll need to run GD on the following
    # problem: \|x@w - y\|^2 + noise_sigma^2/prior_sigma^2 \|\theta - \theta_0\|^2
    # where y = target + N(0, noise_sigma^2)
    # (assuming y is noiseless)
    else:
        t = y + np.random.randn(*y.shape) * noise_sigma 
        while(np.linalg.norm(g) > 0.01 and count < num_steps):
            count += 1
            d = ((x @ w - t) )
            loss_gd.append(d)
            g = d @ x 
            g = g + noise_sigma/prior_sigma * (w - w0)
            g = g/np.linalg.norm(y) # normalize step size so we don't diverge
            wold = w
            w = w - step_size * g
        return w, loss_gd

        



def get_posterior_samples(x, y, prior_sigma=1.0, l=0.01, d=None):
    
    # Posterior covariance
    if len(x) > 0:
        S0 = prior_sigma * np.eye(x.shape[1])
        SN = np.linalg.inv(np.linalg.inv(S0) + 1/l * x.T@x)
        # Posterior mean
        mN = SN @ np.dot(x.T,y)/l
    else:
        mN = np.zeros(d)
        S0 = prior_sigma * np.eye(d)
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

def iterative_estimator(xtrain, ytrain, xtest, ytest, l=1.0, k=500, prior_sigma=0.1, lse=False, integrate_over_params=True):
  n = len(xtrain)
  test_errors = []
  mean_errors = []
  sample_errors = []
  sample_vars = []
  target = ytrain + np.random.randn(*ytrain.shape)* np.sqrt(l)
  for i in range(n):
    trains = xtrain[:i]
    train_ys = target[:i]
    w = np.linalg.lstsq(trains, train_ys, rcond=None)[0]
    err = - ( np.linalg.norm(xtrain[i] @ w - target[i]))**2/(2*l) - 1/2 * np.log(2 * np.pi * l)
    mean_errors.append(err)
    sampler = get_posterior_samples(trains, train_ys, prior_sigma=prior_sigma, l=l, d=xtrain.shape[1])
    
    samples= []

    if integrate_over_params:
        for _ in range(k): 
            w = sampler()
            samples.append(-np.linalg.norm(xtrain[i]@ w - target[i])**2/(2*l) - 1/2* np.log(2*np.pi*l))

        sample_err = np.mean(samples) if not lse else np.log(np.sum (1/k * np.exp(samples)))
        test_err = np.linalg.norm(xtest @ w - ytest)/np.linalg.norm(ytest)
        test_errors.append(test_err)
        sample_var = np.var(samples)
        sample_errors.append(sample_err)
        sample_vars.append(sample_var)
    
    else:
        for _ in range(k):
            # Get posterior predictive samples
            w = torch.tensor(sampler()).cuda()
            samples.append(xtrain[i] @ w.cpu().numpy())
        # Use the mean and variance of the predictive samples to estimate
        # the parameters of the predictive distribution
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)   *(k/(k-1))  
        print(sample_mean,sample_var, samples)
        # Compute p(y) under posterior          
        log_posterior_likelihood =  -(sample_mean - target[i])**2/(2*sample_var )- np.log(2*np.pi*sample_var)/2
        sample_errors.append(log_posterior_likelihood)

  print(sample_errors)
  return test_errors, mean_errors, sample_errors, sample_vars
 
def sample_then_optimize(prior_sampler, xtrain, ytrain, l=1.0, k=1, prior_sigma=None, noisy_posterior=False):
    w = prior_sampler()
    noise_sigma = l
    if prior_sigma is None:
        prior_sigma = 1/len(w)**2
    n = len(xtrain)
    d = xtrain.shape[1]
    ls = []
    preds = []
    opts = []
    ws = [w]
    winit = w.copy()
    a = winit[0]
    for i in range(n):
        l =- (w @ xtrain[i] - ytrain[i])**2/(  2*noise_sigma) - 1/2 * np.log(np.pi * 2 * noise_sigma) 
        p = w @ xtrain[i] 
        preds.append(p)
        wopt = np.linalg.lstsq(xtrain[:i], ytrain[:i])[0]
        lopt = np.abs(wopt @ xtrain[i] - ytrain[i])**2/noise_sigma
        opts.append(lopt)

        ls.append(l)
        for _ in range(k):
            dist = (np.linalg.norm(w - wopt))
            ns = noise_sigma if noisy_posterior else 0.
            wnew, epoch_l = train_one_epoch_iterative(w, xtrain[:i+1], ytrain[:i+1], 500, prior_sigma=prior_sigma, noise_sigma=ns, w0=winit)            
            w = wnew
        ws.append(w)
        dists.append(dist)
    return ls, preds, opts, ws


def train_gd_ensemble(prior_sampler, xtrain, ytrain, num_models=4, l=1.0, k=1, prior_var=None, integrate_over_params=True):
    w_ensemble = []
    noise_sigma = l
    n = len(xtrain)
    d = xtrain.shape[1]
    l_ensemble = []
    if prior_var is none:
        prior_var = 1/d**2
    # Collect posterior samples from num_models models 
    # with initial params sampled from prior.
    for j in num_models:    
        w = prior_sampler()
        
        ls = []
        dists = []
        opts = []
        ws = [w]
        winit = w.copy()
        a = winit[0]
        if l > 0.:
            # Add noise to y to sample from noisy posterior
            target = ytrain + np.random.randn(ytrain.shape) * np.sqrt(l)
        else:
            target = ytrain
        
        for i in range(n):
            l = - (w @ xtrain[i] - target[i])**2/(  2*noise_sigma) - 1/2 * np.log(np.pi * 2 * noise_sigma) 
            wopt = np.linalg.lstsq(xtrain[:i], target[:i])[0]
            lopt = np.abs(wopt @ xtrain[i] - target[i])**2/noise_sigma
            opts.append(lopt)

            ls.append(l)
            for _ in range(k):
                dist = (np.linalg.norm(w - wopt))
                wnew, epoch_l = train_one_epoch_iterative(w, xtrain[:i+1], target[:i+1], 500, prior_sigma=prior_var, noise_sigma=l, w0=winit)            
                w = wnew
            ws.append(w)
            dists.append(dist)
        
        w_ensemble.append(ws)
        l_ensemble.append(ls)
    
    if integrate_over_params:
        return np.mean(l_ensemble, axis=0)

    else:
        # Now w_ensemble of form [ [w_trajectory(m)] for m in ensemble]
        w_ensemble = np.array(w_ensemble)
        w_mean = np.mean(w_ensemble, axis=0)
        w_var = np.var(w_ensemble, axis=0)*(k/k-1)
        log_p = -(w_mean @ xtrain - ytrain)**2/(2*w_var) - np.log(2*np.pi*sample_var)/2 
        return np.sum(log_p)

# Figures from the appendix
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
        #print(ls)
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

# Generate plot for LHS of Figure 2
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

    for _ in range(4):
        linear_models = [BLRModel(1/k**2, 1/d_inf, lambda x : x[:, :k]) for k in num_features]

        marg_liks.append([m.get_marginal_likelihood(xtrain[:, :k], ytrain) for m, k in zip(linear_models, num_features)])
        # print('done ml')
        lbs.append([np.sum(m.get_elbo(xtrain[:, :k], ytrain, custom_noise=False, integrate_over_params=True)[2]) for m, k in zip(linear_models, num_features)])
        sto_results = [sample_then_optimize(lambda : np.random.normal(np.zeros(k), 1/k), xtrain[:, :k], ytrain, l=0.1) for k in num_features]
        
        sto_lbs.append([np.sum(s[0]) for s in sto_results])
    # print(sto_lbs)

    plt.clf()
    font = {'family':'serif', 'size':16}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    plt.xlabel('Number of Features')
    plt.ylabel('Log Likelihood Estimate')
    #print(len(num_features), len(marg_liks))
    #ax1.set_xlabels(lengthscales)
    sns.tsplot( marg_liks, condition='Log Evidence', marker='o', ax=ax)
    sns.tsplot(np.array(lbs), condition='ELBO', color='green', marker='o', ax=ax)
    print(np.argmax(lbs[0]), np.argmax(marg_liks[0]), np.argmax(sto_lbs[0]))
    sns.tsplot(sto_lbs, condition='Sample-then-optimize', color='purple', marker='o', ax=ax)
    plt.title('Feature Selection')
    
    locs, labels = plt.xticks()
    plt.xticks(locs, num_features)

    plt.tight_layout()
    plt.savefig('feature_dim_selection.png')

    return None

# Generate plot for LHS of Figure 2
def model_selection_integrate_posterior():
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

    num_samples = 4
    for _ in range(num_samples):
        linear_models = [BLRModel(1/k**2, 1/d_inf, lambda x : x[:, :k]) for k in num_features]

        marg_liks.append([m.get_marginal_likelihood(xtrain[:, :k], ytrain) for m, k in zip(linear_models, num_features)])
        # print('done ml')
        lbs.append([np.sum(m.get_elbo(xtrain[:, :k], ytrain, custom_noise=False, integrate_over_params=False)[2]) for m, k in zip(linear_models, num_features)])
        sto_results = [sample_then_optimize(lambda : np.random.normal(np.zeros(k), 1/k), xtrain[:, :k], ytrain, l=0.1) for k in num_features]
        
        sto_preds.append([s[1] for s in sto_results])
    
    # Evaluate sto predictive mean and variance
    preds = np.array(sto_preds)
    mean = np.mean(sto_preds, axis=0)
    var = np.var(sto_preds, axis=0)

    # print(sto_lbs)

    plt.clf()
    font = {'family':'serif', 'size':16}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    plt.xlabel('Number of Features')
    plt.ylabel('Log Likelihood Estimate')
    #print(len(num_features), len(marg_liks))
    #ax1.set_xlabels(lengthscales)
    sns.tsplot( marg_liks, condition='Log Evidence', marker='o', ax=ax)
    sns.tsplot(np.array(lbs), condition='ELBO', color='green', marker='o', ax=ax)
    print(np.argmax(lbs[0]), np.argmax(marg_liks[0]), np.argmax(sto_lbs[0]))
    sns.tsplot(sto_lbs, condition='Sample-then-optimize', color='purple', marker='o', ax=ax)
    plt.title('Feature Selection')
    
    locs, labels = plt.xticks()
    plt.xticks(locs, num_features)

    plt.tight_layout()
    plt.savefig('feature_dim_selection.png')

    return None

# Generate data with sample plot for RHS of Figure 2
def generate_sampling_gap_data():
    torch.manual_seed(0)
    np.random.seed(0)

    
    d = 30
    d_inf = 15

    # One: construct elbo and ml for each scale
    n_models = 6
    
    #lengthscales=[2**(-i ) for i in range(5,n_models+1)]
    num_features = [(i+1)*(int(d/n_models)) for i in range(n_models)]
    xtrain, ytrain = build_random_features(n=2*d, d=d, num_informative_features=d_inf, rand_variance=1.0)
    samples = [1, 3, 10, 50]
    linestyles = ['-', '-.', ':', '--']
    res_dicts = []
    deltas = []
    data = pkl.load(open('gap_data.pkl', 'rb'))
    for i, s in enumerate(samples):
        print("num samples: ", s)
        marg_liks = []
        lbs = []
        lse_lbs = []
        model_ls = []
        sto_lses = []
        sto_lbs = []
        for j in range(2):
            print(j)
            sto = []
            sto_lb = []
            for k in num_features:
                print(k)
                prior_sampler = lambda : 1/k* np.random.randn(k)
                ls = np.array([sample_then_optimize(prior_sampler, xtrain[:, :k], ytrain, l=1/d_inf)[0] for _ in range(s)])
                mean_ps = np.sum(np.log(np.mean(np.exp(ls), axis=0)))
                mean_logs = np.sum(np.mean(ls, axis=0))
                sto.append(mean_ps)
                sto_lb.append(mean_logs)
                
            sto_lses.append(sto) 
            sto_lbs.append(sto_lb)
            lbs.append(sto_lb)
            linear_models = [BLRModel(1/k**2, 1/d_inf, lambda x : x) for k in num_features]

            marg_liks.append([m.get_marginal_likelihood(xtrain[:, :k], ytrain) for m, k in zip(linear_models, num_features)])
            
            lse_lbs.append([np.sum(m.get_elbo(xtrain[:, :k], ytrain, custom_noise=False, n_samples=s, lse=True)[2]) for m, k in zip(linear_models, num_features)])
            #lse_lbs = data[i]['lse']
            lbs.append([np.sum(m.get_elbo(xtrain[:, :k], ytrain, custom_noise=False, n_samples=s, lse=False)[2]) for m, k in zip(linear_models, num_features)])
            
             
        deltas.append(np.mean(np.array(lse_lbs) - np.array(marg_liks)))
        sns.tsplot(sto_lbs, condition='s-t-o, k=' + str(s),  alpha=.5 + 0.5 *(i)/len(samples), linestyle=linestyles[i], color=(i/len(samples), 0.2, i/len(samples)))
        sns.tsplot(lbs, condition='lb' + str(s), color='orange',  alpha=.5 + 0.5 *(i)/len(samples), linestyle=linestyles[i])
        sns.tsplot(lse_lbs, condition='ELBO, k=' + str(s),  alpha=.5 + 0.5 *(i)/len(samples), linestyle=linestyles[i], color=(0.2, 0.5, i/len(samples)))
        res_dicts.append({'lse_sto': sto_lses, 'lse_exact':lse_lbs, 'ml':marg_liks, 'lb_sto':sto_lbs, 'lb_exact':lbs})
        pkl.dump(res_dicts, open('gap_data.pkl', 'wb'))


    sns.tsplot(marg_liks, condition='Log Evidence', color='green', alpha=.5 + 0.5 *(i)/len(samples))
    res_dicts.append(marg_liks)
    
    ticks, labels = plt.xticks()
    plt.xticks(ticks, [str(int(5*(i+1))) for i in ticks])

    plt.legend(loc='lower right')
    plt.title('Gap between ELBO and Evidence with Multi-Sample Estimator')
    plt.xlabel('Number of features')
    plt.ylabel('Value of estimator')
    plt.savefig('gap2.png')
    
    plt.clf()
    plt.plot(samples, deltas)
    plt.savefig('deltas_by_k.png')

# Plot to illustrate gap between ML and ELBO estimator (Figure 2)
def plot_lse_gap(file_name):
    font = {'family':'serif', 'size':16}
    plt.rc('font', **font)
    d_list = pkl.load(open(file_name, 'rb'))
    #print(len(d_list))
    #print(d_list)
    samples = [1, 3, 10, 50]
    linestyles = ['-', '-.', ':', '--']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, d in enumerate(d_list):
        s = samples[i]
        #sns.tsplot(d['lb_sto'], condition='s-t-o, k=' + str(s),  alpha=.5 + 0.5 *(i)/len(samples), linestyle=linestyles[i], color=(i/len(samples), 0.2, i/len(samples)))
        #sns.tsplot(d['lb_exact'], condition='s-t-o, k=' + str(s),  alpha=.5 + 0.5 *(i)/len(samples), linestyle=linestyles[i], color=(i/len(samples), 0.2, i/len(samples)))
        sns.tsplot(d['lse_exact'], condition='ELBO (exact), k=' + str(s), color=(0.5 + 0.5 * i/len(samples), 0.2, 0.2),  marker='o', alpha=.2 + 0.8 *(i/len(samples))**2, linestyle=linestyles[i], ax=ax)
        sns.tsplot(d['lse_sto'], condition='ELBO (s-t-o), k=' + str(s), color=(0.2, 0.5 + 0.5 * i/len(samples), 0.2), marker='o', alpha=.2 + 0.8 *(i/len(samples))**2, linestyle=linestyles[i], ax=ax)
    
    sns.tsplot(d['ml'], condition='ML',  alpha=.5 + 0.5 *(i)/len(samples), linestyle=linestyles[i], color='green', ax=ax)
    plt.xlabel('Feature dimension')
    n_models = 6
    
    num_features = [(i+1)*(int(30/n_models)) for i in range(n_models)]
    locs, labels = plt.xticks()
    plt.xticks(locs, num_features)
    plt.legend(fontsize=13, loc='lower right', ncol=2)
    plt.ylabel('Estimator Value')
    plt.title('Closing the Gap in the ELBO')   
    plt.tight_layout()
    plt.savefig('gap2.png')
    


if __name__ == '__main__':
    print('Welcome, you diligent and resourceful scientist!')
    #generate_lb_ml_plot(prior_sigma=1., noise_var=0.1)
    model_selection_plot()
    #plot_lse_gap('gap_data.pkl')
    #gap_plot()
    print('sample then opt')
    #sample_opt_plot()

