import numpy as np 
import torch
from linear_regression import * 
import matplotlib.pyplot as plt
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

def optimize_linear_combo(w, linear_models, w_optimizer, x, y, num_epochs=1, training_type='concurrent'):
    criterion = torch.nn.MSELoss() 
    model_losses = []
    losses = []
    for i in range(1, len(x)-1):

        for _ in range(num_epochs):        
            w_optimizer.zero_grad()
            data = x[:i]
            labels = y[:i] 

            if training_type == 'concurrent':
                print('building post samples')
                post_samples = torch.tensor([m.posterior_pred_sample(data, labels, x[i+1]) for m in linear_models], dtype=torch.float64)
            elif training_type == 'post':
                post_samples = torch.tensor([m.posterior_pred_sample(x, y, x[i+1]) for m in linear_models], dtype=torch.float64)
            else:
                post_samples = torch.tensor([m.posterior_pred_sample([], [], x[i+1]) for m in linear_models], dtype=torch.float64)

            pred = w @ post_samples 
            loss = criterion(pred, torch.tensor([y[i+1]], dtype=torch.float64))
            loss.backward()
            losses.append(loss.item())
            w_optimizer.step()
            model_losses.append([(y[i+1] - ps)**2 for ps in post_samples])
        print(w)
        
    return w, losses, model_losses

def build_feature_subset_map(num_features):
    return lambda x : x.T[:num_features].T       

def tsplot(x, y,**kw):
    x = x
    data = y
    est = np.mean(y, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw)
    ax.margins(x=0)

def build_rff(d):
    def f(x):
        out = np.zeros(2*d)
        for k in range(d):
            out[2*k] = np.cos((d+1)*x)
            out[2*k + 1] = np.sin((d+1)*x)
        return out
    return f

def linear_ensembles_sgd_elbo_ml_plot():
    '''
    Generate a figure showing
    '''
    torch.manual_seed(0)
    np.random.seed(1)
    weights = []
    marg_liks = []
    lbs = []
    model_ls = []
    weights_posttraining = []
    weights_pretraining = []
    # One: construct elbo and ml for each scale
    n_models = 10
    
    lengthscales=[4**(-i ) for i in range(n_models)]
    for _ in range(5):
        xtrain, ytrain = build_random_features(n=50, d=25)
        xtest, ytest = build_random_features(n=50, d=25)
        linear_models = [BLRModel(ps, 0.1, lambda x : x) for ps in lengthscales]
        w = torch.tensor(np.ones(n_models)*1/n_models)
        
        w.requires_grad = True
        print(w)
        opt = torch.optim.SGD([w], lr=0.005)
        wnew, ls, model_losses = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=2)
        model_ls.append(np.sum(model_losses, axis=0))
        weights.append(w.detach().numpy())
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        opt = torch.optim.SGD([w], lr=0.005)
        wpost, _, _ = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=2, training_type='post')
        weights_posttraining.append(wpost.detach().numpy())
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        opt = torch.optim.SGD([w], lr=0.005)
        wpre, _, _ = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=2, training_type='pre')
        weights_pretraining.append(wpre.detach().numpy())


        #print('model l shape', model_ls[-1].shape)
        
        
    marg_liks.append([m.get_marginal_likelihood(xtrain, ytrain) for m in linear_models])
    print('done ml')
    lbs.append([np.sum(m.get_elbo(xtrain, ytrain, custom_noise=False)[2]) for m in linear_models])
        # plt.plot(ls)
    # plt.show()
    #sns.tsplot(model_losses)
    #plt.savefig('losses.png')
    plt.clf()
    [plt.scatter(ml, w) for ml, w in zip(lbs, weights)]
    #plt.scatter(marg_liks, weights)
    plt.title('weight as fn of elbo of model')
    plt.xlabel('L(M)')
    plt.ylabel('weight found by sgd')
    plt.savefig('weightvelbo.png')
    print(len(model_losses))
    plt.clf()
    fig, ax1 = plt.subplots()
    color='red'
    sns.tsplot(weights, condition='weights (concurrent sampling)', color='red', ax=ax1)
    sns.tsplot(weights_pretraining, condition='weights (prior sampling)', ax=ax1, linestyle='-.', color='orange')
    sns.tsplot(weights_posttraining, condition='weights (posterior sampling)', ax=ax1, color='purple')
    ax2 = ax1.twinx()
    #[plt.scatter(range(len(ml)), ml/(-1*np.mean(ml)), label='normalized evidence') for ml in marg_liks]
    plt.savefig('weights.png')
    ax1.set_xlabel('Negative log prior variance (base 4)')
    ax1.set_ylabel('Weight')
    ax2.set_ylabel('Negative likelihood')
    #ax1.set_xlabels(lengthscales)
    sns.tsplot(marg_liks, condition='log evidence', ax = ax2)
    sns.tsplot(lbs, condition='elbo', color='green', ax = ax2)
    plt.title('Selecting Prior Variance for Bayesian Linear Regression')
    plt.tight_layout()
    plt.savefig('nle.png')
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None

if __name__ == "__main__":
    linear_ensembles_sgd_elbo_ml_plot()
    
