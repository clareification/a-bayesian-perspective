import numpy as np 
import torch
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
from linear_regression import BLRModel
from matplotlib import rc 

rc('text', usetex=True)
params= {'text.latex.preamble' : [r'\usepackage{amsfonts}']}
matplotlib.rc('font', **{'family' : "serif"})
plt.rcParams.update(params)

def generate_data(n, d):

    x = 1/d*np.random.rand(n, d)
    w =  np.ones(d)
    y = x @ w + 0.1 * np.random.randn(n)

    return x, y, w

def generate_easy_data(n):
    y = np.random.rand(n)
    x = np.concatenate([y.reshape(1,-1), np.diag(2*np.random.rand(n))]).T
    return x, y, None

def exact_elbo(muN, SN, x, y):
    return y**2 - 2 * muN.T @ x * y + x.T @ SN @ x + x.T @ muN.reshape(-1, 1) @ muN.reshape(-1,1).T @ x

if __name__ == "__main__":
    n = 10
    d = n+1

    x, y, w = generate_easy_data(n)
    xtest, ytest, _ = generate_easy_data(n)
    xtest2 = xtest[:, :1]
    x2 = x[:, :1]
    prior_sigma = 1.0
    noise_sigma = 1.0 

    losses = []
    model = BLRModel(prior_sigma, noise_sigma, lambda x : x)
    model2 = BLRModel(prior_sigma, noise_sigma, lambda x : x)
    opt_losses = []
    test_perfs = []
    test_perfs2 = []
    
    for i in range(n):
        mN, SN = model.get_posterior_mean_and_var(x[:i], y[:i], d=d)
        
        #print(mN.shape, SN.shape, w, mN)
        losses.append(exact_elbo(mN, SN, x[i], y[i])/2/noise_sigma - 1/2 * np.log(2*np.pi*noise_sigma))
        mN2, SN2 = model.get_posterior_mean_and_var(x2[:i], y[:i], d=1)
        opt_losses.append(exact_elbo(mN2, SN2, x[i][:1], y[i])/2/noise_sigma - 1/2 * np.log(2*np.pi*noise_sigma))
        test_perfs.append(np.linalg.norm(xtest @ mN.reshape(-1) - ytest.reshape(-1))**2)
        test_perfs2.append(np.linalg.norm((xtest2 @ mN2).reshape(-1) - ytest.reshape(-1))**2)
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(losses, color=(0., 0.6, 0.8), label=r"$\ell(\phi_1(x_i), y_i) = -\mathbb{E}[\log p(y|x, \theta)]$")
    
    k= int(n/2)
    lb = [np.sum(losses[:k]) for k in range(n)]
    opt_lb = [np.sum(opt_losses[:k]) for k in range(n)]
    ax.fill_between(range(k), losses[:k], np.zeros(k), alpha=0.3, color=(0., 0.6, 0.6))
    ax.plot(lb, color=(0., 0.4, 0.4), label=r"$\sum_{i=1}^k \ell(\phi_1(x_i), y_i) = -\mathcal{L}(x_{<k}, y_{<k})$")
    ax.axvline(x=k-1, ymin=0, ymax=lb[k]/np.max(lb) - 0.02, color='black', linestyle='-.')
    plt.plot([-1*model.get_marginal_likelihood(x[:i], y[:i]) for i in range(n)], label=r'$-\log p(D_{<k}; M_1)$')
    plt.legend()
    
    plt.title('Online Computation of Evidence')
    plt.xlabel('Data point seen')
    plt.ylabel('Log posterior predictive value')
    plt.tight_layout()
    plt.savefig('illustration.png')
    plt.clf()

    fig, ax = plt.subplots(figsize=(6,3))

    plt.plot([-1*model2.get_marginal_likelihood(x2[:i], y[:i]) for i in range(n)], label=r'$-\log p(D_{<k}; M_2)$')
    ax.plot(opt_lb, color='purple', label=r"$\sum_{i=1}^k \ell(\phi_2(x_i), y_i) = -\mathcal{L}(x_{<k}, y_{<k})$")
    ax.plot(opt_losses, color='pink', label=r"$\ell(\phi_2(x_i), y_i) = -\mathbb{E}[\log p(y|x, \theta)]$")
    ax.fill_between(range(k), opt_losses[:k], np.zeros(k), alpha=0.3, color=(0.6, 0.2, 0.6))
    ax.axvline(x=k-1, ymin=0, ymax=opt_lb[k]/np.max(opt_lb) - 0.02, color='black', linestyle='-.')

    plt.legend()
    
    plt.title('Online Computation of Evidence')
    plt.xlabel('Data point seen')
    plt.ylabel('Log posterior predictive value')
    plt.tight_layout()
    plt.savefig('illustration2.png')


    plt.clf()
    '''
    fig, ax = plt.subplots(figsize=(8,4))
    ax2 = ax.twinx()
    print(test_perfs, test_perfs2)
    ax.plot(test_perfs, label=r'Test error $\mathcal{M}_1$')
    ax.plot(test_perfs2, label=r'Test error $\mathcal{M}_2$')
    ax.set_ylabel('Test Loss')
    ax2.plot(ml, color='pink', label=r"$\mathcal{L}(X_{<k}, y_{<k}; \mathcal{M}_1)$")
    ax2.plot(opt_ml, color='green', label=r"$\mathcal{L}(X_{<k}, y_{<k}; \mathcal{M}_2)$")
    ax2.set_ylabel('SoTL')
    plt.xlabel('Number of datapoints seen')
    ax.legend()
    ax2.legend()
    plt.title("Test error of posterior vs SoTL")
    plt.savefig('illustration_test.png')
    '''


