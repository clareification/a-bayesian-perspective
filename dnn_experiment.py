import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import matplotlib.pyplot as plt
import h5py
from argparse import ArgumentParser
from model_zoo import MLP, ConvEncoder
from data_loaders import get_FMNIST

def train_step(x, y, m, o, criterion):
    if len(x.shape)==3:
        x  = x.unsqueeze(0)
    o.zero_grad()
    out = m(x)
    if isinstance(y, int):
        y = torch.tensor([y]).reshape(-1)
    l = criterion(out, y.cuda())
    l.backward()
    o.step()
    out2 = m(x)
    l2 = criterion(out2, y)

    return out, l

def linear_train_step(outs, w, y, criterion, model_backprop=False, model_opts=None, models=None, x=None, step_size=0.001):
    outs = torch.stack(outs, axis=1)
    if not model_backprop:
        outs = outs.detach()

    w.requires_grad = True
    w_out = torch.einsum('abc, b->ac', outs, w)

    if isinstance(y, int):
        y = torch.tensor([y]).reshape(-1)

    l = criterion(w_out, y)
    l.backward()
    g = w.grad.data

    with torch.no_grad():
        w = w - step_size*g 
        if np.random.rand() > 0.999:
            print('grad norm', torch.norm(step_size*g), l)

    if model_backprop:
        for i, o in enumerate(model_opts):
            o.step()
    return w, l
    

def train_parallel_one_epoch(w, ms, opts, training_data, step_size=0.001, loss_fn=torch.nn.CrossEntropyLoss):
    # Train weights and models concurrently.
    # Don't backprop linear loss to models
    M = len(ms)
    losses = [[] for m in ms]
    criterion = loss_fn()

    lin_losses = []

    #counter =0
    for _, data in enumerate(training_data):     
        y = data[1]

        x = data[0]
        outs = []
        # Take one optimization step for each model
        for i, (m, o) in enumerate(zip(ms, opts)):
            out, l = train_step(x.cuda(), y.cuda(), m, o, criterion)
            outs.append(out.cpu())
            losses[i].append(l.detach().item())

        w, l = linear_train_step(outs, w, y, criterion)
        w.requires_grad = True
        
        lin_losses.append(l.detach().item())

        

    return lin_losses, losses, w

#only train models, not w
def train_models_one_epoch(w, ms, opts, training_data, step_size=0.001, loss_fn=torch.nn.CrossEntropyLoss):
    M = len(ms)
    losses = [[] for m in ms]
    criterion = loss_fn()
    lin_losses = []

    for _, data in enumerate(training_data):    
        y = data[1]

        x = data[0]
        outs = []
        # Take one optimization step for each model
        for i, (m, o) in enumerate(zip(ms, opts)):
            out, l = train_step(x.cuda(), y, m, o, criterion)
            outs.append(out.cpu())
            losses[i].append(l.detach().item())
        

    return lin_losses, losses, w

#only train w, not models
def train_w_one_epoch(w, ms, opts, training_data, step_size=0.001, loss_fn=torch.nn.CrossEntropyLoss):
    # Train weights and models concurrently.
    # DO backprop linear loss to models

    M = len(ms)
    losses = [[] for m in ms]
    criterion = loss_fn()

    lin_losses = []
    for j, data in enumerate(training_data):     
        y = data[1]

        x = data[0] #.reshape([-1, 784])
        outs = []

        # Compute model outputs
        for i, (m, o) in enumerate(zip(ms, opts)):
            o.zero_grad()
            if len(x.shape)==3:
                x  = x.unsqueeze(0)
            out = m(x.cuda())
            outs.append(out.cpu())

        # Compute loss of linear layer and backprop         
        w, l = linear_train_step(outs, w, y, criterion, False, model_opts=opts, models=ms)

        w.requires_grad = True
        
        lin_losses.append(l.detach().item())

        
    print(j)
    return lin_losses, [], w

def train_combo_one_epoch(w, ms, opts, training_data, step_size=0.001, loss_fn=torch.nn.CrossEntropyLoss):
    # Train weights and models concurrently.
    # DO backprop linear loss to models

    M = len(ms)
    losses = [[] for m in ms]
    criterion = loss_fn()

    lin_losses = []
    for j, data in enumerate(training_data):     
        y = data[1]

        x = data[0] #.reshape([-1, 784])
        print(x.shape)
        outs = []

        # Compute model outputs
        if isinstance(y, int):
            y = torch.tensor([y]).reshape(-1).cuda()
        for i, (m, o) in enumerate(zip(ms, opts)):
            o.zero_grad()
            if len(x.shape)==3:
                x  = x.unsqueeze(0)
            out = m(x.cuda())
            with torch.no_grad():
                l_est = criterion(out.cpu(), y)
                losses[i].append(l_est.detach().item())
            
            outs.append(out.cpu())

        # Compute loss of linear layer and backprop         
        w, l = linear_train_step(outs, w, y, criterion, True, model_opts=opts, models=ms)
        w.requires_grad = True
        lin_losses.append(l.detach().item())

        
    print(j)
    return lin_losses, losses, w

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--train_type', type=str, required=True)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    _, _, _, _, train_loader, test_loader = get_FMNIST()

    input_size = 28*28
    model_names  = ["mlp2_200","mlp2_100","mlp2_10", "mlp1_200","mlp1_100","mlp1_400", "conv200_5","conv200_3", "conv100_5","conv100_3" ] 
    models = [MLP(input_size,200).cuda(),MLP(input_size,100).cuda(), MLP(input_size,10).cuda(),
            single_MLP(input_size,200).cuda(), single_MLP(input_size,100).cuda(), single_MLP(input_size,400).cuda(),
            ConvEncoder(out_dim=10, input_shape=[1,28,28], n_filter=200, kernel_size=5, padding=0).cuda(),
            ConvEncoder(out_dim=10, input_shape=[1,28,28], n_filter=100, kernel_size=3, padding=1).cuda(),
            ConvEncoder(out_dim=10, input_shape=[1,28,28], n_filter=200, kernel_size=5, padding=0).cuda(),
            ConvEncoder(out_dim=10, input_shape=[1,28,28], n_filter=100, kernel_size=3, padding=1).cuda()]

    optims = [] 
    for i, m in enumerate(models):  
        o = torch.optim.SGD(m.parameters(), lr=0.01)
        optims.append(o)

    lin = 1/len(models) * torch.ones(len(models))
    lin.requires_grad = True


    lin_opt = torch.optim.SGD([lin], lr=0.01)
    l2s = [[] for _ in models]
    for i in range(args.reps):
        print('iteration ', i)
        print(args.train_type)
        if args.train_type =='concurrent':
            # do concurrent training 
            lins, l2, lin = train_combo_one_epoch(lin, models, optims, train_loader)

        else:
            if args.train_type =='parallel':
                #do parallel training
                lins, l2, lin = train_parallel_one_epoch(lin, models, optims, train_loader)
            else:
                #do model training
                lins, l2, _ = train_models_one_epoch(lin, models, optims, train_loader)
                _, _, lin = train_w_one_epoch(lin, models, optims, train_loader)
        
        print(l2)
        l2s = [s + l2[i] for i, s in enumerate(l2s)]
        for j,m in enumerate(models):
            torch.save(m.state_dict(), model_names[j]+"_reps"+str(args.reps)  +"_seed"+ str(args.seed) + args.train_type)

    l2 = l2s
    sums = [sum(z) for z in l2]
    mean_l = []

    criterion = torch.nn.CrossEntropyLoss()
    for m in models:
        l = 0
        n=1000
        for i, dat in enumerate(test_loader):  #(list(tl)[:n]):
            x = dat[0]
            if len(x.shape) ==3:
                x = x.unsqueeze(0)
            out = m(x.cuda())
            l += criterion(out, dat[1].cuda()).detach().item()
        mean_l.append(l/n)


    with h5py.File("./res_cifar_reps"+str(args.reps)  +"_seed"+ str(args.seed) + args.train_type, 'w') as f:
      tr = f.create_group('res')
      tr.create_dataset('sums', data=sums) #cumulative XE loss 
      tr.create_dataset('xe', data=l2) #xe loss
      tr.create_dataset('mean_l', data=mean_l) #test error (figure 3a)
      tr.create_dataset('weights', data=lin.detach().numpy().reshape(-1)) #concurrently trained linear weights (figure 3a)
