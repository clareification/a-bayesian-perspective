import numpy as np 
import torch
from linear_regression import * 
import matplotlib.pyplot as plt
import seaborn as sns


def optimize_linear_combo(w, linear_models, w_optimizer, x, y, num_epochs=1, training_type='concurrent'):
    criterion = torch.nn.MSELoss() 
    model_losses = []
    losses = []
    for i in range(1, len(x)):

        for _ in range(num_epochs):        
            w_optimizer.zero_grad()
            data = x[:i]
            labels = y[:i] 

            if training_type == 'concurrent':
                print('building post samples')
                post_samples = torch.tensor([m.posterior_pred_sample(data, labels, x[i]) for m in linear_models], dtype=torch.float64)
            elif training_type == 'post':
                post_samples = torch.tensor([m.posterior_pred_sample(x, y, x[i]) for m in linear_models], dtype=torch.float64)
            else:
                post_samples = torch.tensor([m.posterior_pred_sample([], [], x[i], d=m.feature_map(x[i]).shape[0]) for m in linear_models], dtype=torch.float64)

            pred = w @ post_samples 
            loss = criterion(pred, torch.tensor([y[i]], dtype=torch.float64))
            loss.backward()
            losses.append(loss.item())
            w_optimizer.step()
            model_losses.append([(y[i] - ps)**2 for ps in post_samples])
        print(w)
        
    return w, losses, model_losses

def build_feature_subset_map(num_features):

    return lambda x : np.array(x).T[:num_features].T       

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

# Generate plot and data for prior selection in Figure 3
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
    plt.title('Model Weight as function of ELBO')
    plt.xlabel('L(M)')
    plt.ylabel('Weight found by SGD')
    plt.savefig('weightvelbo.png')
    print(len(model_losses))
    plt.clf()
    font = {'family':'serif', 'size':16}
    plt.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(5.8, 4.8))
    color='red'
    sns.tsplot(weights, condition='Weights (concurrent sampling)', marker='o', color='red', ax=ax1)
    sns.tsplot(weights_pretraining, condition='Weights (prior sampling)', marker='o', ax=ax1, linestyle='-.', color='orange')
    sns.tsplot(weights_posttraining, condition='Weights (posterior sampling)', marker='o', ax=ax1, color='purple')
    locs, labels = plt.xticks()
    plt.xticks(locs, np.log(lengthscales)/np.log(4))
    ax2 = ax1.twinx()
    #[plt.scatter(range(len(ml)), ml/(-1*np.mean(ml)), label='normalized evidence') for ml in marg_liks]
    
    plt.savefig('weights2.png')
    ax1.set_xlabel('Log Prior Variance')

    #ax1.set_ylabel('Weight')
    #ax2.set_ylabel('Log Likelihood')
    #ax1.set_xlabels(lengthscales)
    sns.tsplot(marg_liks, condition='Log Evidence', marker='o', ax = ax2)
    sns.tsplot(lbs, condition='ELBO', color='green', marker='o', ax = ax2)
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    plt.title('Prior Variance Selection')
    plt.tight_layout()
    plt.legend(handles + handles2, labels + labels2, loc='lower right', fontsize=10)
    plt.savefig('feature_selection_weights2.png')

    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None

# Generate data for feature selection problem
def save_feature_dim_selection_data():
    torch.manual_seed(0)
    np.random.seed(0)
    weights = []
    marg_liks = []
    lbs = []
    model_ls = []
    weights_posttraining = []
    weights_pretraining = []
    # One: construct elbo and ml for each scale
    d = 30
    d_inf = 15

    # One: construct elbo and ml for each scale
    n_models = 6
    
    lengthscales=[2**(-i ) for i in range(5,n_models+5)]
    num_features = [(i+1)*(int(d/n_models)) for i in range(n_models)]
    

    for _ in range(5):
        xtrain, ytrain = build_random_features(n=2*d, d=d, num_informative_features=d_inf, rand_variance=1.0)
        linear_models = [BLRModel(1/k**2, 1/d_inf, build_feature_subset_map(k)) for k in num_features]
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        print(w)
        opt = torch.optim.SGD([w], lr=0.005)
        wnew, ls, model_losses = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=12)
        model_ls.append(np.sum(model_losses, axis=0))
        weights.append(w.detach().numpy())
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        opt = torch.optim.SGD([w], lr=0.005)
        wpost, _, _ = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=12, training_type='post')
        weights_posttraining.append(wpost.detach().numpy())

        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        opt = torch.optim.SGD([w], lr=0.005)
        wpre, _, _ = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=12, training_type='pre')
        weights_pretraining.append(wpre.detach().numpy())
        lbs.append([np.sum(m.get_elbo(xtrain[:, :k], ytrain, custom_noise=False)[2]) for m, k in zip(linear_models, num_features)])


        #print('model l shape', model_ls[-1].shape)
        
        marg_liks.append([m.get_marginal_likelihood(xtrain[:, :k], ytrain) for m, k in zip(linear_models, num_features)])
    res_dict = {'ml':marg_liks, 'wpre':weights_pretraining, 'wpost':weights_posttraining, 'lb':lbs, 'w':weights}
    pkl.dump(res_dict, open('feature_dim_data.pkl', 'wb')) 

# Generate plot for feature selection in Figure 3
def feature_dim_selection_ensemble_plot():
    '''
    Generate a figure showing
    '''
    n_models = 6
    d=30
    num_features = [(i+1)*(int(d/n_models)) for i in range(n_models)]

    res_dict = pkl.load(open('feature_dim_data.pkl', 'rb'))
    marg_liks = res_dict['ml']
    weights_pretraining = res_dict['wpre']
    weights_posttraining = res_dict['wpost']
    lbs = res_dict['lb']
    weights = res_dict['w']
    plt.clf()
    [plt.scatter(ml, w) for ml, w in zip(lbs, weights)]
    #plt.scatter(marg_liks, weights)
    plt.title('Model Weight as function of ELBO')
    plt.xlabel('L(M)')
    plt.ylabel('Weight found by SGD')
    plt.savefig('weightvelbo.png')

    plt.clf()
    font = {'family':'serif', 'size':16}
    plt.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(5.8, 4.8))
    color='red'
    sns.tsplot(weights, condition='Weights (concurrent sampling)', marker='o', color='red', ax=ax1)
    sns.tsplot(weights_pretraining, condition='Weights (prior sampling)', marker='o', ax=ax1, linestyle='-.', color='orange')
    sns.tsplot(weights_posttraining, condition='Weights (posterior sampling)', marker='o', ax=ax1, color='purple')
    
    locs, labels = plt.xticks()
    plt.xticks(locs, num_features)
    ax2 = ax1.twinx()
    #[plt.scatter(range(len(ml)), ml/(-1*np.mean(ml)), label='normalized evidence') for ml in marg_liks]
    
    plt.savefig('weights2.png')
    ax1.set_xlabel('Number of Features')

    ax1.set_ylabel('Weight')
    #ax2.set_ylabel('Log Likelihood')
    #ax1.set_xlabels(lengthscales)
    sns.tsplot(marg_liks, condition='Log Evidence', marker='o', ax = ax2)
    sns.tsplot(lbs, condition='ELBO', color='green', marker='o', ax = ax2)
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    max_marg = np.mean([w[2] for w in marg_liks])
    max_lb = np.mean([w[2] for w in lbs])
    # ax2.plot([2,2,2], )
    # ax2.add_artist(circle1)
    
    #ax2.annotate('Maximum Model Evidence', xy=(2, max_marg), arrowprops=dict(facecolor='black', shrink=0.05))
    ax2.plot(2, max_marg, 'o')
    # ax2.plot(2, ), 'o', ms=15)
    plt.title('Feature Selection')
    plt.tight_layout()
    plt.legend(handles + handles2, labels + labels2, loc='lower right', fontsize=10)
    plt.savefig('weights_feature_selection.png')
    plt.axvline(2, linestyle='-.')
    ax2.plot([2, 2], [max_marg, max_lb], marker='o')
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None

if __name__ == "__main__":
    # Figure 
    feature_dim_selection_ensemble_plot()
    #linear_ensembles_sgd_elbo_ml_plot()
    

