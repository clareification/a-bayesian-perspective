import torch
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from model_zoo import ReluEncoder, ConvEncoder
from data_loaders import get_MNIST
from torchvision import transforms, datasets
import pickle as pkl
from linear_regression import *
from optimize_then_prune import *
from tqdm import tqdm

def marginal_likelihood(x, y, n=200, l=1.0, prior=1.0):
    # Assume default prior for the moment?
    prior_sigma = prior
    x = torch.tensor(x).cpu().numpy()

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

class ApproxBLRModel():
    def __init__(self, prior_sigma, noise_sigma, feature_map, learning_rate=0.1):
        self.prior_sigma = prior_sigma
        self.noise_sigma = noise_sigma 
        self.feature_map = feature_map
        self.learning_rate = learning_rate
        self.features = None
        
        
    def prior_weight_sampler(self, N):
        sampler = self.get_posterior_samples([], [], N, use_own_features=False)
        return sampler

    def posterior_weight_sampler(self, x, y):
        phi = self.feature_map(x)
        sampler = self.get_posterior_samples(phi, y, use_own_features=False)
        return sampler

    def get_posterior_samples(self, x, y,  N=None, use_own_features=True):
        
        prior_sigma = self.prior_sigma
        l = self.noise_sigma
        
        if len(x) == 0:
            mN = torch.zeros(N)
            S0 = prior_sigma * torch.eye(N)
            SN = S0

        else:
            if use_own_features:
                if self.features is None:
                    print(x.shape)
                    self.features = self.feature_map(x)
                x = self.features[:len(x)]
            else:
                x = torch.tensor(x).cuda()
                x = self.feature_map(x)
            # Posterior covariance
            y = torch.tensor(y).double().cuda()
            x = torch.tensor(x).double().cuda()
            S0 = prior_sigma * torch.eye(x.shape[1]).cuda()
            SN = (torch.inverse(S0) + 1/l * x.T@x).cuda()
            SN = torch.inverse(SN)

            # Posterior mean
            mN = SN @ torch.matmul(x.T, y.double())/l
            mN = mN
            SN = SN

        # Return a function that generates samples from the posterior.
        m = torch.distributions.multivariate_normal.MultivariateNormal(mN, SN)
        return lambda : m.sample().cpu().numpy()

    def posterior_pred_sampler(self, x, y, xtest):
        phi = torch.tensor(x).cuda()#self.feature_map(x)
        if len(x) == 0:
            sampler = self.get_posterior_samples(phi, y, N=self.feature_map(xtest).shape[1])
        else:
            sampler = self.get_posterior_samples(phi, y, use_own_features=False)
        w = sampler() 
        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, -1])
        return lambda : torch.tensor(sampler()).double().cuda() @ torch.tensor(self.feature_map(xtest).T).double().cuda()
    
    def get_marginal_likelihood(self, x, y):
        phi = self.feature_map(x)
        n = len(x)
        ml = marginal_likelihood(phi, y, n, self.noise_sigma, self.prior_sigma)
        return ml

    def get_elbo_opt(self, x, y, batch_size=32, logging=True):
        phi = self.feature_map(x).float()
        self.weight = torch.tensor(self.prior_weight_sampler(phi.shape[1])(), dtype=torch.float32).cuda()
        self.weight.requires_grad = True
        
        opt = torch.optim.SGD([self.weight], lr=self.learning_rate)
        yt = torch.tensor(y, dtype=torch.float32).cuda()
        criterion = torch.nn.MSELoss()
        losses = []

        for i in range(int(phi.shape[0]/batch_size)):
            grad = torch.tensor([100.])
            
            l = criterion( (phi[i*batch_size:(i+1)*batch_size] @ self.weight ).reshape(-1), yt[i*batch_size:(i+1)*batch_size].reshape(-1)).cpu().detach().numpy()
            losses.append(l)
            grad_steps = 0
            max_steps = 2000
            data = phi[:(i+1)*batch_size]
            labels = yt[:(i+1)*batch_size].reshape(-1)
            while(torch.norm(grad) > 0.01  and grad_steps < max_steps):
                grad_steps +=1 
                opt.zero_grad()
                
                preds = self.weight @ data.T 
                loss = criterion(preds.reshape(-1), labels)
                loss.backward()
                opt.step()
                grad = self.weight.grad

                #print('grad', grad)
            if logging and not i%10 : print(grad_steps, loss.item(), l.item(), i/(phi.shape[0]/batch_size))
        print('sum losses: ', np.sum(losses))
        return losses, None, None, None 
            
    def get_elbo(self, x, y, k=2):
        phi = torch.tensor(x)# self.feature_map(x)
        return self.iterative_estimator(phi, y, phi, y, k)

    def set_features(self, x):
        self.features = self.feature_map(x)
    
    def iterative_estimator(self, xtrain, ytrain, xtest, ytest,k=2):
        n = len(xtrain)
        l = self.noise_sigma
        prior_sigma = self.prior_sigma
        test_errors = [0]
        mean_errors = [0]
        sample_errors = [0]
        sample_vars = [0]
        ytest = torch.tensor(ytest, dtype=torch.float32).cuda()
        self.set_features(xtrain)
        xtrain = torch.tensor(self.features).cuda()
        for i in tqdm(range(n-1)):
            trains = xtrain[:i]
            train_ys = ytrain[:i]
            # w = np.linalg.lstsq(trains, train_ys, rcond=None)[0]
            # err = - ( np.linalg.norm(xtrain[i+1] @ w - ytrain[i+1])/l) - 1/2 * np.log(2 * np.pi * l)
            # mean_errors.append(err)
            
            sampler = self.get_posterior_samples(trains, train_ys, N=xtrain.shape[1], use_own_features=True)
            
            samples= []
            
            for _ in range(k): 
                w = torch.tensor(sampler()).cuda()
                
                samples.append( (-torch.norm(xtrain[i+1].double()@ w.double() - ytrain[i+1])**2/l - 1/2* np.log(2*np.pi*l)).cpu().numpy())
            
            sample_err = np.mean(samples)
            # test_err = torch.norm(xtest @ w.double() - ytest)/torch.norm(ytest)
            # test_errors.append(test_err)
            sample_var = np.var(samples)
            sample_errors.append(sample_err)
            sample_vars.append(sample_var)
            
        
        return test_errors, mean_errors, sample_errors, sample_vars

def generate_rff(d, X, l=1.0):    
    if len(X.shape)==1:
        X = X.reshape([1, -1])
    X = torch.tensor(X, dtype=torch.float32).cuda()
    k = X.shape[1]
    n = X.shape[0]
    # sample biases uniformly from 0, 2pi
    b = 2*np.pi * np.random.rand(d)
    # Sample w according to normal(0, 1/l**2)
    W = 1/np.sqrt(l) * torch.rand(d, k).cuda()
    print('types', type(X), type(W))
    fs = (W @ X.T) .T + torch.tensor(np.repeat([b], n, axis=0)).cuda()
    return np.sqrt(2/d) * torch.cos(fs)#np.bmat([np.cos(fs), np.sin(fs)])


    enc = ConvEncoder(d, n_channels=1)
    xt = torch.tensor(X, dtype=torch.float32)
    xt = xt.reshape([-1, 1, 28, 28])
    return enc(xt).detach().numpy()

def build_embedding(d, feature_map, k=None, l=None, filters=None):
    if feature_map == ConvEncoder:
        enc = feature_map(d, n_channels=1, num_filters=filters)
        return lambda x : enc(torch.tensor(x, dtype=torch.float32).reshape([-1, 1, 28, 28])).detach().numpy()
    elif feature_map == ReluEncoder:
        enc = feature_map(784, d)
        return lambda x : enc(torch.tensor(x, dtype=torch.float32).reshape([-1, 784])).detach().numpy()
    else:
        w = 1/np.sqrt(l) * torch.rand(d, k).cuda()
        b = 2*np.pi * np.random.rand(d)

        def f(X):
            n = X.shape[0]
            X = torch.tensor(X, dtype=torch.float32).cuda()
            fs = (w @ X.T) .T + torch.tensor(np.repeat([b], n, axis=0)).cuda()
            return np.sqrt(2/d) * torch.cos(fs)
        return f
    

def load_subset_mnist(max_label=2, max_size=1000, reshape=True):
    root = '.'
    input_size = 28
    channels = 1
    num_classes = 10
    k = max_size
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.LinearTransformation(torch.eye(input_size**2), torch.zeros(input_size**2))
    ])
    mnist = datasets.MNIST(root + "data/MNIST", train=True, transform=input_transform, target_transform=None, download=True)
    
    x = mnist.data.reshape(-1, input_size**2)
    y = mnist.targets.numpy().reshape(-1)
    test = datasets.MNIST(root + "data/MNIST", train=False, transform=input_transform, target_transform=None, download=True)
    xtest = test.data.reshape(-1, input_size**2)
    ytest = test.targets.numpy().reshape(-1)
    x = x.numpy()

    # Subset generation

    x = x[np.where(y <max_label)]#[:k]
    y = y[np.where(y < max_label)]#[:k]
    x = x[:k]
    y = y[:k]
    xtest= xtest.numpy()
    xtest = xtest[np.where(ytest <max_label)]
    ytest = ytest[np.where(ytest < max_label)]

    return xtest, ytest, x, y

def load_subset_cifar(max_label=2, max_size=1000, reshape=True):
    root = '.'
    input_size = 32
    channels = 3
    num_classes = 10
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.LinearTransformation(torch.eye(input_size**2), torch.zeros(input_size**2))
    ])
    mnist = datasets.CIFAR10(root + "data/CIFAR10", train=True, transform=input_transform, target_transform=None, download=True)
    x = mnist.data.reshape(-1, input_size**2 * channels)
    y = mnist.targets
    test = datasets.CIFAR10(root + "data/CIFAR10", train=False, transform=input_transform, target_transform=None, download=True)
    xtest = test.data.reshape(-1, input_size**2 * channels)
    ytest = test.targets.numpy().reshape(-1)
    x = x.numpy()
    y = y.numpy().reshape(-1)

    k = max_size

    # Subset generation
    x = x[np.where(y <max_label)][:k]
    y = y[np.where(y < max_label)][:k]
    xtest= xtest.numpy()
    ytest = ytest.numpy().reshape(-1)
    xtest = xtest[np.where(ytest <max_label)]
    ytest = ytest[np.where(ytest < max_label)]

    return xtest, ytest, x, y

def generate_rff_mnist(lengthscale, fourier_dim, max_label=2):
    n = len(y)
    xtest, ytest, x, y = load_subset_mnist(max_label)
    fx = l(fourier_dim, np.concatenate([x, xtest], axis=0))

    # Want to use same random fourier features to produce train and test set
    fx = generate_rff(fourier_dim, np.concatenate([x, xtest], axis=0), l=lengthscale)   
    ftest = fx[n:]
    fx = fx[:n]
    return ftest, ytest, fx, y

def get_feature_subset(d):
    def f(x):
        if len(x) == 0:
            return x
        if len(x.shape) > 1:
            return x[:, :d]
        else: 
            return x[:d]
def rff_lengthscale_selection(feature_maps, num_samples, x, y, xtest, ytest, learning_rates=None, train_linear_combo=False,
                                noise_sigma=0.1):
    
    marg_liks = []
    elbos = []
    weights = []
    n_models = len(feature_maps)
    tests = []
    stos = []
    if learning_rates is None:
        learning_rates = [0.1 for _ in feature_maps]
    n = len(x)
    
    for i in range(num_samples):    
        ml = []
        elbo = []
        weight = []
        test = []
        models = []
        sto = []
        for i, feature_map in enumerate(feature_maps):
            fx = feature_map(np.concatenate([x, xtest], axis=0))   
            ftest = fx[n:]
            fx = fx[:n]
            
            linear_model = ApproxBLRModel(1.0, noise_sigma, lambda x : x, learning_rate = learning_rates[i])
            # get ml estimates
            curr_ml = linear_model.get_marginal_likelihood(fx, y)
            ml.append(curr_ml)
            print('ml:', curr_ml)
            te, me, se, sv = linear_model.get_elbo(fx, y)
            print('elbo', np.sum(se))
            lb = np.sum(se)
            elbo.append(lb)
            approx_elbo, _, _, _ = linear_model.get_elbo_opt(fx, y, batch_size=1, logging=False)
            approx_elbo = np.sum(approx_elbo), -np.sum(approx_elbo)/linear_model.noise_sigma + -1/2 * n * np.log(np.pi * 2 *linear_model.noise_sigma)
            sto.append(approx_elbo)
            print('sample then opt ', approx_elbo)
            #print('approx elbo', np.sum(approx_elbo))#, np.sum(te))
            # elbo.append(np.sum(se))
            # print(np.sum(se))
            # Add the model that uses these features to the set
            linear_model = ApproxBLRModel(1.0, noise_sigma, feature_map)
            linear_model.set_features(x)
            #print('lin model feature shape', linear_model.feature_map(x).shape)
            models.append(linear_model)
        marg_liks.append(ml)
        elbos.append(elbo)
        stos.append(sto)    
        if train_linear_combo:    
            w = torch.tensor(np.ones(n_models)*1/n_models)
            
            w.requires_grad = True
            opt = torch.optim.SGD([w], lr=0.1)
            #wnew = None
            #('feature map shapes: ',[ m.feature_map(x).shape for m in models])
            wnew, ls, model_losses = optimize_linear_combo(w, models, opt, x, y, num_epochs=1, batch_size=1)
            print(wnew)
            
            
            weights.append(wnew.detach().numpy())
            tests.append(test)
            print(np.sum(model_losses, axis=0))

    return marg_liks, elbos, weights, stos

def optimize_linear_combo(w, linear_models, w_optimizer, x, y,
                            num_epochs=1, training_type='concurrent', batch_size=10,
                            logging=False):
    w.cuda()
    criterion = torch.nn.MSELoss() 
    model_losses = []
    losses = []
    x = torch.tensor(x).cuda()
    y = torch.tensor(y).cuda()
    for i in range(1, int(len(x)/batch_size)-1):        
        data = x[:i*batch_size].cuda()
        labels = y[:i*batch_size].cuda() 
        i = i * batch_size
        if data is None:
            data = []
            labels = []
        if training_type == 'concurrent':
                post_samplers = [m.posterior_pred_sampler(data, labels, x[i+1]) for m in linear_models]
                
        elif training_type == 'post':
            post_samplers = [m.posterior_pred_sampler(x, y, x[i+1]) for m in linear_models]
        else:
            post_samplers = [m.posterior_pred_sampler([], [], x[i+1]) for m in linear_models]
        for j in range(num_epochs):        
            w_optimizer.zero_grad()
            
            post_samples = torch.tensor( [s() for s in post_samplers], dtype=torch.float64).cuda()
            
            pred = w.cuda() @ post_samples 
            
            loss = criterion(pred, torch.tensor([y[i+1]], dtype=torch.float64).cuda())
            loss.backward()
            losses.append(loss.item())
            w_optimizer.step()
            model_losses.append([(y[i+1] - ps).cpu().numpy()**2 for ps in post_samples])
        if not i%100 and logging: 
            print(w)
            print(post_samples, y[i+1])
        
    return w, losses, model_losses
def old_plotting():
    # x = np.random.rand(n, 1)

    # y = (x)**2

    # y = np.array(list(map(lambda x : 1 if x > 0 else -1, y)))
    # w = np.linalg.lstsq(x, y.reshape(-1))[0]
    # for i in range(10):
    #     print(w @ x[i], y[i])
    # plt.scatter(y[:1000], x[:1000]@ w)
    # #print(np.linalg.norm(x[:1000]@w - y.reshape(-1)[:1000])/1000)
    # plt.savefig('gap.png')
    # plt.clf()

    tests = []
    
    for l in [generate_rff]:#, generate_conv_features, generate_relu_features]:   
        
        residualss = []
        widths = [32]
        #widths = [2,4,8,16,32, 64, 128, 256, 512, 1024, 2048]
        
        for _ in range(3):
            residuals = []
            rtest = []
            fourier_dim = 256
            #,  1e9, 1e11, 1e15]#, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
            for i, freq in enumerate(widths):
                fx = l(fourier_dim, np.concatenate([x, xtest], axis=0))
                if l == generate_rff:
                    fx = l(fourier_dim, np.concatenate([x, xtest], axis=0), l=freq)
                u, s, v = np.linalg.svd(fx)
                #print(s/np.max(s), u.shape, v.shape)
                #u2, s2, v2 = np.linalg.svd(x)ç
                #print(s2, u2.shape, v2.shape)
                #print(np.max(v[0]), np.min(v[0]), np.max(v2[0]), np.min(v[0]))
                #plt.imsave(str(l) + 'top singular.png', v2[0].reshape([28, 28]))
                # plt.plot(np.linalg.svd(fx)[1])
                # plt.savefig(str(fourier_dim) + '.png')
                ftest = fx[n:]
                fx = fx[:n]

                #fx = generate_rff(fourier_dim, x)
                # w_train = torch.ones(fourier_dim)
                # mnist = zip(fx, y)
                w, r, _, _  = np.linalg.lstsq(fx, y.reshape(-1))
                # print(w.shape, fx.shape)
                #acc = np.sum( np.equal((fx @ w ).astype(np.uint8).reshape(-1), y.reshape(-1)))
                plt.scatter(y+ i/len(widths), fx@ w, label=fourier_dim, alpha=0.2)
                
                # for i in range(10):
                #     print(w @ fx[i], y[i])
                
                r = np.linalg.norm((fx @ w).reshape(-1) - y.reshape(-1))**2/n 
                print('mse: ', r)# = MSE(w, xtrain)
                residuals.append(r)
                rt = np.sum(np.linalg.norm(ftest@w - ytest.reshape(-1))**2)/len(ytest) 
                rtest.append(rt)
                plt.scatter(ytest + 5.1 + i/len(widths), ftest @ w, label='test', alpha=0.2)
                # print(rtest, residuals)

            tests.append(rtest)
            residualss.append(residuals)
    with open('results.pkl', 'wb') as f:

        pkl.dump([residualss, tests], f)
        #plt.plot(residuals, label='l = ' + str(l))
    
    plt.savefig('residuals2.png')
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # conv_trains = residualss[:len(widths)]
    # relu_trains = residualss[len(widths):]
    # sns.tsplot(conv_trains, condition='conv train', ax=ax1)
    # sns.tsplot(relu_trains, condition='relu train', ax=ax1)
    #[ax1.plot(range(len(residuals)), residuals, label=l)for residuals, l in zip (residualss, ['conv', 'relu'])]
    ax1.legend()
    #plt.savefig('residuals1.png')
    # sns.tsplot(tests[:len(widths)], condition='conv test', color='purple', linestyle='-.', ax=ax2)
    # sns.tsplot(tests[len(widths):], condition='relu test', color='green', linestyle='-.', ax=ax2)
    # ax1.set_xticks(widths)
    # ax2.set_xticks(widths)
    sns.tsplot(tests, condition='tests')
    sns.tsplot(residualss, condition='train', color='orange')
    print(np.max(residualss[0]))
    plt.legend()
    plt.savefig('residuals.png')

def map_from_param(fn, dim, p=None):
    if p is not None:
        return lambda x : fn(dim, x, p)
    else:
        return lambda x : fn(dim, x)
def plot_and_save_data(mls, els, ws, ts, save=False):
    if save:
        pkl.dump([mls, els, ws], open('rffdata.pkl', 'wb'))
    print(ws, np.sum(els[0],axis=0 ), mls)
    fig, ax1 = plt.subplots()
    ax3 = ax1.twinx()
    sns.tsplot(ws/np.max(ws), condition='weight', ax=ax1)    
    sns.tsplot(np.sum(els, axis=1)/np.max(els), condition='elbo', color='orange', ax=ax1)   
    sns.tsplot(mls/np.max(mls), condition='ml', color='green', linestyle='-.', ax=ax3)
    ax1.legend()
    ax3.legend()
    plt.title('Normalized weight/ml/elbo for rff model')
    plt.savefig('rff_selection_nn.png')

def rff_selection_plot(train_linear_combo=False):
    print(train_linear_combo, 'Train linear combo?')
    torch.manual_seed(0)
    np.random.seed(0)
    fourier_dim = 784
    k =  200
    xtest, ytest, x, y = load_subset_mnist(2, k)

    ls = [ 10**(i) for i in range(5, 14)]#, 1e16, 1e32]#, 1e20] # 1e7 is the best
    dims = [2, 8]
    maps = []
    for l in ls:
            input_dim = xtest.shape[1]
            m = build_embedding(1000, generate_rff, input_dim, l)
            maps.append(m)
    res_dict = {}
    for k in [200, 2000]:
        xtest, ytest, x, y = load_subset_mnist(2, k)
        
        for ns in [10**(-i) for i in range(6)]:
            mls, els, ws, stos = rff_lengthscale_selection(maps, 1, x, y, xtest, ytest, train_linear_combo=train_linear_combo, noise_sigma=ns)
            curr_dict = {'mls':mls, 'els': els, 'ws':ws, 'stos':stos}
            res_dict[ns] = curr_dict
        with open('data_v2_' + str(k) + '.pkl', 'wb') as f:
            pkl.dump(res_dict, f)

def random_nn_selection_plot():
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

    # Load data
    k = 2000
    xtest, ytest, xtrain, ytrain = load_subset_mnist(2, k)

    # One: construct elbo and ml for each scale
    n_models = 5
    dims = [8, 16, 32, 64, 128]
    embeddings = [build_embedding(256, ConvEncoder, filters=d) for d in dims] # + [build_embedding(256, ReluEncoder) for d in dims]
    for _ in range(1):
        linear_models = [ApproxBLRModel(1/d**2, 1/d_inf, em) for em in embeddings]
        w = torch.tensor(np.ones(n_models)*1/n_models)
        
        w.requires_grad = True
        print('w: ', w)
        opt = torch.optim.SGD([w], lr=0.005)
        wnew, ls, model_losses = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=2)
        print('done concurrent', wnew)
        model_ls.append(np.sum(model_losses, axis=0))
        weights.append(wnew.detach().numpy())
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        opt = torch.optim.SGD([w], lr=0.005)
        wpost, _, _ = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=2, training_type='post')
        print('done post', wpost)
        weights_posttraining.append(wpost.detach().numpy())
        w = torch.tensor(np.ones(n_models)*1/n_models)
        w.requires_grad = True
        print('starting the pre network')
        opt = torch.optim.SGD([w], lr=0.005)
        wpre, _, _ = optimize_linear_combo(w, linear_models, opt, xtrain, ytrain, num_epochs=2, training_type='pre')
        print('done pre', wpre)
        weights_pretraining.append(wpre.detach().numpy())
        
        
    marg_liks.append([m.get_marginal_likelihood(xtrain, ytrain) for m in linear_models])
    print('done ml')
    lbs.append([np.sum(m.get_elbo(xtrain, ytrain)[2]) for m in linear_models])
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
    plt.savefig('weightvelbo_nn.png')
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
    #sns.tsplot(lbs, condition='elbo', color='green', ax = ax2)
    plt.title('Selecting Feature Dimension for Bayesian Linear Regression')
    plt.tight_layout()
    plt.savefig('feature_selection_weights_nn.png')
    #[plt.scatter(ml, w) for w,ml  in zip(weights, model_losses))]
    #plt.show()
    return None
   

if __name__ == "__main__":
    #rff_selection_plot(train_linear_combo=True)
    d = pkl.load(open('data_v2_200.pkl', 'rb'))
    print(d.keys())
    mls = []
    els =[] 
    ws = []
    stos = []
    fig, axs = plt.subplots(1,len(d.keys()), figsize=(18,2))
    print(len(axs))
    def normalize_data(data):
        d = (data - np.mean(data))/np.var(data)
        return d

    for k in list(d.keys()):
        stos.append(normalize_data(d[k]['stos'][0]))
        els.append(normalize_data(d[k]['els'][0]))
        ws.append(normalize_data(d[k]['ws'][0]))
        mls.append(normalize_data(d[k]['mls'][0]))

    #sns.tsplot(els, ax=axs[1])
    print(ws)
    for i in range(len(stos)):
        #axs[i].plot(stos[i]/np.abs(np.max(stos[i])), color='red')
        axs[i].plot(els[i]/np.abs(np.max(els[i])), color='blue')
        axs[i].plot(ws[i]/np.abs(np.max(ws[i])), color='green')
        axs[i].plot(mls[i]/np.abs(np.max(mls[i])), color='purple')

    # sns.tsplot(ws, ax=axs[0])
    # sns.tsplot(els, ax=axs[1])
    # sns.tsplot(mls, ax=axs[2])
    print(len(stos), stos[0][0])
    # sns.tsplot(stos, ax=axs[3])

    plt.savefig('check.png')
    #plot_and_save_data([mls, els, ws, stos, save=False)
    
    
    

