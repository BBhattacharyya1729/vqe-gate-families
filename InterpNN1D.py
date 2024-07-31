import pandas as pd
import numpy as np
import torch
import copy
from torch import nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset

pi=np.pi

###Setup and print the device. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("Final 1D Interp NN")

PAULIS = {
    "I" : np.array([[1,0],[0,1]],dtype='cfloat'),
    "X" : np.array([[0,1],[1,0]],dtype='cfloat'),
    "Y" : np.array([[0,1j],[-1j,0]],dtype='cfloat'),
    "Z" : np.array([[1,0],[0,-1]],dtype='cfloat'),
}


### Neural Network Class
class ControlNN(nn.Module):
    def __init__(self,layers,width,n_controls,n_steps):

        super().__init__()

        # Create a ModuleList to store the linear layers
        self.layers = nn.ModuleList()

        # Add some linear layers to the ModuleList
        self.layers.append(nn.Linear(1,width))
        self.layers.append(nn.ReLU())

        
        for i in range(layers):
            self.layers.append(nn.Linear(width,width))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(width,n_controls*n_steps))


    def forward(self, x):
        x=x.float()
        for layer in self.layers:
            #print(layer.weight.shape)
            x = layer(x)

        return x


class InterpolationNetwork1D():
    def __init__(self,N,layers,width,dt,n_steps,operator,H_drives):
        self.N = N
        self.n_controls = len(H_drives)
        self.n_qubits = int(np.log2(H_drives.shape[-1]))
        self.dt = dt
        self.n_steps = n_steps
        self.operator = operator
        self.H_drives=H_drives
        self.model = ControlNN(layers,width,self.n_controls,n_steps).to(device) ###Randomized network 

    ###Method to integrate accels.
    def euler(self,dda):
        da_init=-torch.sum(torch.cat((torch.tensor([[0 for i in range(self.n_controls)]]).to(device),torch.cumsum(dda[:-1]*self.dt,dim=0)))[:-1],dim=0)/(self.n_steps-1)
        da=torch.cat((torch.tensor([[0 for i in range(self.n_controls)]]).to(device),torch.cumsum(dda[:-1]*self.dt,dim=0)))+torch.Tensor.repeat(da_init,(self.n_steps,1))
        return torch.cat((torch.tensor([[0 for i in range(self.n_controls)]]).to(device),torch.cumsum(da[:-1]*self.dt,dim=0)))

    ###Get Accels from current Model (NN)
    def get_accels(self,theta):
        x = self.model(theta.unsqueeze(0))
        return (x.reshape(self.n_steps,self.n_controls))

    ###Get Controls from current Model (NN)
    def get_controls(self,theta):
        return self.euler(self.get_accels(theta))

    ###Do the rollout (for given controls)
    def get_rollout(self,controls):
        c_vals=controls.cfloat()
        G=torch.einsum('ij,jkl -> ikl',c_vals,self.H_drives.clone().cfloat().to(device))
        G=torch.linalg.matrix_exp(-1j*G*self.dt).cfloat()
        U = torch.tensor(np.eye(2**self.n_qubits)).cfloat().to(device)
        for j in range(0,self.n_steps-1):
             U = torch.einsum('ij,jk->ik',G[j],U)
        return U

    def get_infid(self,control,theta):
        rollout_op = self.get_rollout(control)
        true_op = self.operator(theta).H.cfloat()
        p = torch.matmul(rollout_op,true_op)
        l = 1-abs(torch.trace(p))/2**self.n_qubits
        return torch.max(l,torch.tensor(1e-7))
    
    def get_model_infid(self,theta):
        return(self.get_infid(self.get_controls(theta),theta))

    ###Loss 
    def infid_loss(self,theta_list): 
        infidelity = torch.vmap(func = lambda t: self.get_model_infid(t))
        return torch.mean(infidelity(theta_list))
    
    ###Get log10 data
    def infid_data(self,theta_list):
        infidelity = torch.vmap(func = lambda t: self.get_model_infid(t))
        return(torch.log10(infidelity(theta_list)))

    def L1(self,theta_list):
        accels = torch.vmap(func = lambda t: self.get_accels(t))(theta_list)
        return torch.norm(accels,p=1)/(self.n_controls * self.n_steps*theta_list.shape[0])

    ##Pretraining function. Returns Losses
    def pretrain(self,pretraining_path,lr = 1e-3,iters=10**5,thresh=1e-5):

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.n_controls,self.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)
        vmap_euler = torch.vmap(func = lambda controls: self.euler(controls))
        gpu_pretraining_data=vmap_euler(gpu_pretraining_data).float()
        opt = Adam(self.model.parameters(), lr=lr)
        pretraining_loss_fn  = nn.MSELoss()

        losses=[]

        for epoch in range(iters):
            yhat = torch.vmap(func = lambda t: self.get_controls(t))(torch.linspace(0,2*pi,self.N).to(device))
            loss=pretraining_loss_fn(yhat,gpu_pretraining_data)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            
            if(epoch % 1000 == 0):
                print(f"Epochs: {epoch} Loss: {loss.item()}")
                if(loss.item()<thresh):
                    break
                print("-------------------")
        ###Returns the loss history
        return np.array(losses)

    ###Compare Integrated Controls 
    def get_comparison_controls(self,pretraining_path):
        get_controls_list = torch.vmap(func = lambda t: self.get_controls(t))
        vmap_euler = torch.vmap(func = lambda controls: self.euler(controls))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.n_controls,self.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)

        c_vals=torch.Tensor.cpu(get_controls_list(torch.linspace(0,2*pi,self.N).to(device))).detach().numpy()
        c_vals1=torch.Tensor.cpu(vmap_euler(gpu_pretraining_data)).detach().numpy()

        return c_vals,c_vals1

    def get_comparison_accels(self,pretraining_path):
        get_accels_list = torch.vmap(func = lambda t: self.get_accels(t))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.n_controls,self.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)

        c_vals=torch.Tensor.cpu(get_accels_list(torch.linspace(0,2*pi,self.N).to(device))).detach().numpy()
        c_vals1=torch.Tensor.cpu(gpu_pretraining_data).detach().numpy()

        return c_vals,c_vals1



    ###Compare Sample Infidelities
    def get_comparison_samples(self,pretraining_path):
        temp_f = torch.vmap(lambda t: self.get_model_infid(t))
        model_infid = temp_f(torch.linspace(0,2*pi,self.N).to(device))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.n_controls,self.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)

        pretrain_infid=[]
        for i,v in enumerate(np.linspace(0,2*pi,self.N)):
            pretrain_infid.append(self.get_infid(self.euler(gpu_pretraining_data[i]),v))

        return model_infid,torch.tensor(pretrain_infid)

    def train(self,lr=5e-5,iters=10**5,train_size=500,test_size=4500,batch_size=100,c=1e-5):
        opt = Adam(self.model.parameters(), lr=5e-4)
        val_thetas = torch.linspace(0,2*pi,test_size).to(device)
        train_thetas = torch.linspace(0,2*pi,train_size).to(device)
        losses=[]
        test_losses = []
        batched_thetas = DataLoader(train_thetas.clone().detach().float().to(device), batch_size=batch_size, shuffle=True)
        for epochs in range(iters):
          for x in batched_thetas:
            loss=self.infid_loss(x) + c*self.L1(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            test_loss = torch.Tensor.cpu(self.infid_loss(val_thetas)).detach().numpy()
            test_losses.append(test_loss)
          if(epochs % 100 ==0):
              test_loss = torch.Tensor.cpu(self.infid_loss(val_thetas)).detach().numpy()
              print(f"Epoch: {epochs}")
              print(f"Total Loss: {test_loss}")
              print("-------------------")
        
        epoch_ratio = train_size // batch_size
        return np.array(losses),np.array(test_losses),np.array([test_losses[i * epoch_ratio -1] for i in range(iters)])

    def active_train(self,lr=5e-5,iters=10**5,train_size=500,test_size=4500,batch_size=100,c=1e-5):
        opt = Adam(self.model.parameters(), lr=lr)
        val_thetas=torch.linspace(0,2*pi,test_size).to(device)
        losses=[]
        test_losses = []
        for epochs in range(iters):
            
            infidelity = torch.vmap(func = lambda t: self.get_model_infid(t))
            temp_losses = infidelity(val_thetas)
            mean_loss = torch.mean(temp_losses)
            max_loss = torch.max(temp_losses)
            train_thetas = torch.rand(int(train_size*max_loss/mean_loss * 1.2)).to(device) * 2 * pi
            train_losses = infidelity(train_thetas)
            random_comp = torch.rand(len(train_thetas)).to(device) * max_loss
            l = train_losses-random_comp
            train_thetas = train_thetas[[l>=0]][:train_size]
            
            
            batched_thetas = DataLoader(train_thetas.clone().detach().float().to(device), batch_size=batch_size, shuffle=True)
            
            for x in batched_thetas:
              loss=self.infid_loss(x) + c*self.L1(x)
              opt.zero_grad()
              loss.backward()
              opt.step()
              losses.append(loss.item())
              test_loss = torch.Tensor.cpu(self.infid_loss(val_thetas)).detach().numpy()
              test_losses.append(test_loss)
            if(epochs % 100 ==0):
                test_loss = torch.Tensor.cpu(self.infid_loss(val_thetas)).detach().numpy()
                print(f"Epoch: {epochs}")
                print(f"Total Loss: {test_loss}")
                print(f"Train Size: {len(train_thetas)}")
                print("-------------------")
        
        epoch_ratio = train_size // batch_size
        return np.array(losses),np.array(test_losses),np.array([test_losses[i * epoch_ratio -1] for i in range(iters)])

    def initialize_model(self,path):
        self.model.load_state_dict(torch.load(path, map_location=device))
    
    
    ###Generate Callibrated Transfer pulses given a transfer matrix (assumes that the original model is good at these theta values)
    def callibrated_pulses(self,transfer_matrix,theta_list):
      return torch.vmap(lambda t: torch.einsum('ji,kj->ik',torch.inverse(transfer_matrix),self.get_controls(t)).T)(theta_list)


class FrozenNetwork1D(InterpolationNetwork1D):
  def __init__(self, N,layers,width,initial_model,dt,n_steps,operator,H_drives,transfer_matrix):
    self.N = N
    self.n_controls = len(H_drives)
    self.n_qubits = int(np.log2(H_drives.shape[-1]))
    self.dt = dt
    self.n_steps = n_steps
    self.operator = operator
    self.model = ControlNN(layers,width,self.n_controls,n_steps).to(device) ###Randomized network 
    self.H_drives=torch.einsum("ij,jkl->ikl", transfer_matrix.cfloat(), H_drives.cfloat())
    self.model.load_state_dict(initial_model)
    for i in self.model.layers[:-1]:
      i.requires_grad = False
  ##Make sure that pretraining data is stored on gpu
  def frozen_train(self,training_pulses, batch_size,lr = 1e-3,thresh=1e-5):

        opt = Adam(self.model.parameters(), lr=lr)
        loss_fn  = nn.MSELoss()

        losses=[]
        train_thetas = torch.linspace(0, 2*pi, batch_size).to(device) 

        batched_thetas = DataLoader(TensorDataset(train_thetas,training_pulses), batch_size=batch_size, shuffle=True)
        iters = 0
        stop = False 
        while(not stop):
            for (x,y) in batched_thetas:
                
                yhat = torch.vmap(func = lambda t: self.get_controls(t))(x)
                loss=loss_fn(yhat,y)
    
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
                if(iters % 100 == 0):
                    print(f"Iteration: {len(losses)} Loss: {loss.item()}")
                total_norm=0
                for p in self.model.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if(total_norm<thresh):
                    stop=True
                losses.append(loss.item())
            iters +=1 
            
        return np.array(losses)



    