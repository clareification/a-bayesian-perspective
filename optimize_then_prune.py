import numpy as np 
import torch
from linear_regression import * 
import matplotlib.pyplot as plt
import seaborn as sns


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
def generate_relu_features(d, X):
    enc = ReluEncoder(X.shape[1], d)
    xt = torch.tensor(X, dtype=torch.float32)
    return enc(xt).detach().numpy()

def build_rff(d):
    def f(x):
        out = np.zeros(2*d)
        for k in range(d):
            out[2*k] = np.cos((d+1)*x)
            out[2*k + 1] = np.sin((d+1)*x)
        return out
    return f

def prior_selection_plot():
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
    plt.title('Model Weight as function of ELBO')
    plt.xlabel('L(M)')
    plt.ylabel('Weight found by SGD')
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
    ax1.set_xlabel('Feature Dimensionality')
    ax1.set_ylabel('Weight')
    ax2.set_ylabel('Likelihood')
    #ax1.set_xlabels(lengthscales)
    sns.tsplot(marg_liks, condition='log evidence', ax = ax2)
    sns.tsplot(lbs, condition='elbo', color='green', ax = ax2)
    plt.title('Selecting Prior Variance for Bayesian Linear Regression')
    plt.tight_layout()
    plt.savefig('variance_selection.png')
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None


def get_feature_subset(d):
    def f(x):
        if len(x) == 0:
            return x
        if len(x.shape) > 1:
            return x[:, :d]
        else: 
            return x[:d]
    return f

def feature_dim_selection_plot():
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
    d = 100
    d_inf = 50

    # One: construct elbo and ml for each scale
    n_models = 10
    
    lengthscales=[2**(-i ) for i in range(5,n_models+5)]
    num_features = [(i+1)*(int(d/n_models)) for i in range(n_models)]
    xtrain, ytrain = build_random_features(n=2*d, d=d, num_informative_features=d_inf, rand_variance=1.0)
    lengthscales=[4**(-i ) for i in range(n_models)]
    for _ in range(5):
        linear_models = [BLRModel(1/d**2, 1/d_inf, get_feature_subset(d)) for d in num_features]
        w = torch.tensor(np.ones(n_models)*1/n_models)
        
        w.requires_grad = True
        print('w: ', w)
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
    plt.title('Model Weight as function of ELBO')
    plt.xlabel('L(M)')
    plt.ylabel('Weight found by SGD')
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
    ax1.set_xlabel('Feature dimensionality')
    ax1.set_ylabel('Weight')
    ax2.set_ylabel('Likelihood')
    #ax1.set_xlabels(lengthscales)
    sns.tsplot(marg_liks, condition='log evidence', ax = ax2)
    sns.tsplot(lbs, condition='elbo', color='green', ax = ax2)
    plt.title('Selecting Feature Dimension for Bayesian Linear Regression')
    plt.tight_layout()
    plt.savefig('feature_selection_weights.png')
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None


if __name__ == "__main__":
    feature_selection_plot()
    

