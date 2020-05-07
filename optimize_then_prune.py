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
        sampler = get_posterior_samples(phi, y, self.prior_sigma, self.noise_sigma)
        w = sampler() 
        return w @ self.feature_map(xtest)
    
    def get_marginal_likelihood(self, x, y):
        phi = self.feature_map(x)
        n = len(x)
        ml = marginal_likelihood(phi, y, n, self.noise_sigma, self.prior_sigma)
        return ml

    def get_elbo(self, x, y):
        phi = self.feature_map(x)
        return iterative_estimator(x, y, x, y, prior_sigma=self.prior_sigma, l=self.noise_sigma)

def optimize_linear_combo(w, linear_models, w_optimizer, x, y, num_epochs=1):
    criterion = torch.nn.MSELoss() 
    model_losses = []
    losses = []
    for i in range(len(x)-1):
        for _ in range(num_epochs):
        
            w_optimizer.zero_grad()
            data = x[:i]
            labels = y[:i] 
            post_samples = torch.tensor([m.posterior_pred_sample(data, labels, x[i+1]) for m in linear_models], dtype=torch.float64)
            pred = w @ post_samples 
            loss = criterion(pred, torch.tensor([y[i+1]], dtype=torch.float64))
            loss.backward()
            losses.append(loss.item())
            w_optimizer.step()
            model_losses.append([(y[i+1] - ps)**2 for ps in post_samples])
        
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

if __name__ == "__main__":
    weights = []
    marg_liks = []
    lbs = []
    model_ls = []
    for _ in range(2):
        n_models = 6
        xtrain, ytrain = build_random_features(n=100, d=50)
        xtest, ytest = build_random_features(n=50, d=50)
        feature_maps = []
        l = [1, 4, 10, 20]

        for i in  range(len(l)):
            f = build_feature_subset_map(l[i])
            feature_maps.append(f)
        lengthscales=[3**(-i ) for i in range(n_models)]
        linear_models = [BLRModel(ps, ps, f) for ps in lengthscales]
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        print(w)
        opt = torch.optim.SGD([w], lr=0.005)
        wnew, ls, model_losses = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=25)
        model_ls.append(np.sum(model_losses, axis=0))
        print('model l shape', model_ls[-1].shape)
        marg_liks.append([m.get_marginal_likelihood(xtrain, ytrain) for m in linear_models])
        lbs.append([np.sum(m.get_elbo(xtrain, ytrain)[2]) for m in linear_models])
        weights.append(w.detach().numpy())
        # plt.plot(ls)
    # plt.show()

    [plt.scatter(ml, w) for ml, w in zip(lbs, weights)]
    print(len(marg_liks), len(weights), len(marg_liks[0]), len(weights[0]), marg_liks[0], weights[0])
    #plt.scatter(marg_liks, weights)
    plt.title('weight as fn of elbo of model')
    plt.xlabel('log evidence')
    plt.ylabel('weight found by sgd')
    plt.show()
    # [plt.scatter(range(len(ml)), ml/(-1*np.mean(ml)), color='blue') for ml in marg_liks]
    # [plt.scatter(range(len(ml)), ml/(-1*np.mean(ml)), color='green') for ml in lbs]
    # plt.title('Log evidence and elbo for models')
    # plt.show()
    print(len(model_losses))
    sns.tsplot(weights, condition='weights')
    #[plt.scatter(range(len(ml)), ml/(-1*np.mean(ml)), label='normalized evidence') for ml in marg_liks]
    plt.show()
    sns.tsplot(marg_liks, condition='normalized log evidence')
    sns.tsplot(lbs, condition='elbo', color='green')
    #plt.title('weight vs marginal likelihood of models')
    plt.show()
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()