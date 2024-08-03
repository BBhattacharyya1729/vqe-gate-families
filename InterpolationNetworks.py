"""Import Statements"""
import pandas as pd
import numpy as np
import torch
import copy
from torch import nn
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns 
from abc import ABC, abstractmethod

pi=np.pi

print(f"""Final Interpolation Networks Classes
----------------------------- 
File Includes the following Classes
*ControlNN
*QuantumSystem
*InterpolationNetwork (ABSTRACT)
*InterpNN1D
*FrozenNN1D
*InterpNN2D
*FrozenNN2D
----------------------------- """)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

"Pauli Matricies"
PAULIS = {
    "I" : np.array([[1,0],[0,1]],dtype='cfloat'),
    "X" : np.array([[0,1],[1,0]],dtype='cfloat'),
    "Y" : np.array([[0,1j],[-1j,0]],dtype='cfloat'),
    "Z" : np.array([[1,0],[0,-1]],dtype='cfloat'),
}

"Dense NN Class"
class ControlNN(nn.Module):
    """
    Initialize a dense NN.

    inputs (int): Number of gate parameters (1 or 2).
    layers (int): Number of hidden layers (after the first).
    width (int): Width of each hidden layer.
    n_controls (int): Number of controls.
    n_steps (int): Number of time steps.
    """
    def __init__(self,inputs,layers,width,n_controls,n_steps):

        super().__init__()

        # Create a ModuleList to store the linear layers
        self.layers = nn.ModuleList()

        # Add some linear layers to the ModuleList
        self.layers.append(nn.Linear(inputs,width))
        self.layers.append(nn.ReLU())

        
        for i in range(layers):
            self.layers.append(nn.Linear(width,width))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(width,n_controls*n_steps))

    """
    Forward pass through the neural network.
    
    x (Tensor): Input tensor.
    Returns: Output tensor
    """
    def forward(self, x):
        x=x.float()
        for layer in self.layers:
            x = layer(x)
        return x

"""Class To Contain all th dynamics"""
class QuantumSystem():
    """
    Initialize the Quantum Control System with specified parameters.

    dt (float): Time step.
    n_steps (int): Number of time steps.
    operator: Function representing the target parameterized operator
    H_drives (tensor): Array of Hamiltonians representing control drives.
    """
    def __init__(self,dt,n_steps,operator,H_drives):
        self.n_controls = len(H_drives)
        self.n_qubits = int(np.log2(H_drives.shape[-1]))
        self.dt = dt
        self.n_steps = n_steps
        self.operator = operator
        self.H_drives=H_drives
    """
    Euler integrate accelerations while mantaining boundary values

    dda (Tensor): Control accelerations
    """
    def euler(self,dda):
        da_init=-torch.sum(torch.cat((torch.tensor([[0 for i in range(self.n_controls)]]).to(device),torch.cumsum(dda[:-1]*self.dt,dim=0)))[:-1],dim=0)/(self.n_steps-1)
        da=torch.cat((torch.tensor([[0 for i in range(self.n_controls)]]).to(device),torch.cumsum(dda[:-1]*self.dt,dim=0)))+torch.Tensor.repeat(da_init,(self.n_steps,1))
        return torch.cat((torch.tensor([[0 for i in range(self.n_controls)]]).to(device),torch.cumsum(da[:-1]*self.dt,dim=0)))

    """
    Compute and multiply the propogators to get the time evolution

    controls (Tensor): Control values.
    """
    def get_rollout(self,controls):
        c_vals=controls.cfloat()
        G=torch.einsum('ij,jkl -> ikl',c_vals,self.H_drives.clone().cfloat().to(device))
        G=torch.linalg.matrix_exp(-1j*G*self.dt).cfloat()
        U = torch.tensor(np.eye(2**self.n_qubits)).cfloat().to(device)
        for j in range(0,self.n_steps-1):
             U = torch.einsum('ij,jk->ik',G[j],U)
        return U

    """
    Compute the infidelity between the target operator and the operator obtained from the control.

    control (Tensor): Control values.
    theta (Tensor): Parameters for the target operator.
    """
    def get_infid(self,control,theta):
        rollout_op = self.get_rollout(control)
        true_op = self.operator(theta).H.cfloat()
        p = torch.matmul(rollout_op,true_op)
        l = 1-abs(torch.trace(p))/2**self.n_qubits
        return torch.max(l,torch.tensor(1e-7))


"""Abstract lass for a General Network. Only used as parent"""
class InterpolationNetwork(ABC):
    """
    Initialize the InterpolationNetwork.

    N (int): Number of sample points
    model (ControlNN): The neural network model.
    system (QuantumSystem): The system being modeled.
    """
    def __init(N,model,system):
        self.N = N
        self.model = model
        self.system = system
    """
    Abstract method to get accelerations for the given theta.

    theta (Tensor): theta values.

    Returns: Accelerations.
    """
    @abstractmethod
    def get_accels(self,theta):
        pass

    """
    get controls for the given theta.

    theta (Tensor): theta values.

    Returns: controls.
    """
    def get_controls(self,theta):
        return self.system.euler(self.get_accels(theta))
    
    """
    get model infidelity for the given theta.

    theta (Tensor): theta values.

    Returns: infidelity.
    """
    def get_model_infid(self,theta):
        return(self.system.get_infid(self.get_controls(theta),theta))
    
    """
    get mean model infidelity for the given theta list.

    theta_list (Tensor): theta values.

    Returns: mean infidelity.
    """
    def infid_loss(self,theta_list): 
        infidelity = torch.vmap(func = lambda t: self.get_model_infid(t))
        return torch.mean(infidelity(theta_list))
    """
    get log10 model infidelity list for the given theta list.

    theta_list (Tensor): theta values.

    Returns: log10 infidelity tensor.
    """
    def infid_data(self,theta_list):
        infidelity = torch.vmap(func = lambda t: self.get_model_infid(t))
        return(torch.log10(infidelity(theta_list)))
    
    """
    get L1 norm for the given theta list.

    theta_list (Tensor): theta values.

    Returns: L1 norm.
    """
    def L1(self,theta_list):
        accels = torch.vmap(func = lambda t: self.get_accels(t))(theta_list)
        return torch.norm(accels,p=1)/(self.system.n_controls * self.system.n_steps*theta_list.shape[0])

    @abstractmethod
    def pretrain(self,pretraining_path,lr,iters,thresh):
        pass
    
    @abstractmethod
    def get_comparison_controls(self,pretraining_path):
        pass

    @abstractmethod
    def get_comparison_samples(self,pretraining_path):
        pass

    @abstractmethod
    def train(self,lr,iters,train_size,test_size,batch_size,c):
        pass

    @abstractmethod
    def active_train(self,lr,iters,train_size,test_size,batch_size,c):
        pass
    
    """
    Initialize the model.

    path: Path for model initialization.
    """
    def initialize_model(self,path):
        self.model.load_state_dict(torch.load(path, map_location=device))

    """
    Get callibrated pulses.
    
    transfer_matrix: Transfer matrix.
    theta_list: List of theta values.

    Returns: Callibrated pulses.
    """
    def callibrated_pulses(self,transfer_matrix,theta_list):
      return torch.vmap(lambda t: torch.einsum('ji,kj->ik',torch.inverse(transfer_matrix),self.get_controls(t)).T)(theta_list)
    

class InterpolationNetwork1D(InterpolationNetwork):
    """
    Initialize the InterpolationNetwork.

    N (int): Number of sample points
    layers (int): The number of hidden layers (after the first).
    width (int): The layer width
    system (QuantumSystem): The system being modeled.
    """
    def __init__(self,N,layers,width,system):
        self.N = N
        self.system=system
        self.model = ControlNN(1,layers,width,self.system.n_controls,self.system.n_steps).to(device) 

    """
    Get model accelerations

    theta (Tensor): Theta value

    Returns: accelerations
    """
    def get_accels(self,theta):
        x = self.model(theta.unsqueeze(0))
        return (x.reshape(self.system.n_steps,self.system.n_controls))

    """
    Pretrain the network

    pretraining_path (String): Pretraining path
    lr (Float): Learning rate
    iters (int): Iteration count
    thresh (Float): Threshold

    Returns: Loss data
    """
    def pretrain(self,pretraining_path,lr = 1e-3,iters=10**5,thresh=1e-5):

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.system.n_controls,self.system.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)
        vmap_euler = torch.vmap(func = lambda controls: self.system.euler(controls))
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
        return np.array(losses)

    """
    (fidelity) Train the network

    lr (Float): Learning rate
    iters (int): Iteration count
    train_size (int): Train size
    test_size (int): Test size
    batch_size (int): Batch size (preferably divisor of train_size)
    c (Float): L1 norm coefficient

    Returns: train losses every iteration, test losses every iteration, and test losses every epoch
    """
    def train(self,lr=5e-5,iters=10**5,train_size=500,test_size=4500,batch_size=100,c=1e-5):
        opt = Adam(self.model.parameters(), lr=5e-4)
        val_thetas = torch.linspace(0,2*pi,test_size).to(device)
        losses=[]
        test_losses = []
        model_history=[]
        for epochs in range(iters):
          train_thetas=torch.rand(train_size).to(device) * 2 * pi
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
              print("-------------------")
          state = self.model.state_dict()  
          for k, v in state.items():
            state[k] = v.cpu()
          model_history.append(state)
        epoch_ratio = train_size // batch_size
        return np.array(losses),np.array(test_losses),np.array([test_losses[i * epoch_ratio -1] for i in range(1,iters+1)]),model_history

    """
    (fidelity) Active train the network

    lr (Float): Learning rate
    iters (int): Iteration count
    train_size (int): Train size
    test_size (int): Test size
    batch_size (int): Batch size (preferably divisor of train_size)
    c (Float): L1 norm coefficient

    Returns: train losses every iteration, test losses every iteration, and test losses every epoch
    """
    def active_train(self,lr=5e-5,iters=10**5,train_size=500,test_size=4500,batch_size=100,c=1e-5):
        opt = Adam(self.model.parameters(), lr=lr)
        val_thetas=torch.linspace(0,2*pi,test_size).to(device)
        losses=[]
        test_losses = []
        model_history=[]
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
                print("-------------------")
            state = self.model.state_dict()  
            for k, v in state.items():
              state[k] = v.cpu()
            model_history.append(state)
        epoch_ratio = train_size // batch_size
        return np.array(losses),np.array(test_losses),np.array([test_losses[i * epoch_ratio -1] for i in range(1,iters+1)]),model_history

    """
    Compare model and pretraining control values

    pretraining_path (String): Pretraining path

    Returns: model controls, pretraining controls
    """
    def get_comparison_controls(self,pretraining_path):
        get_controls_list = torch.vmap(func = lambda t: self.get_controls(t))
        vmap_euler = torch.vmap(func = lambda controls: self.system.euler(controls))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.system.n_controls,self.system.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)

        c_vals=torch.Tensor.cpu(get_controls_list(torch.linspace(0,2*pi,self.N).to(device))).detach().numpy()
        c_vals1=torch.Tensor.cpu(vmap_euler(gpu_pretraining_data)).detach().numpy()

        return c_vals,c_vals1
        
    """
    Compare model and pretraining infidelity values

    pretraining_path (String): Pretraining path

    Returns: model infidelities, pretraining infidelities
    """
    def get_comparison_samples(self,pretraining_path):
        temp_f = torch.vmap(lambda t: self.get_model_infid(t))
        model_infid = temp_f(torch.linspace(0,2*pi,self.N).to(device))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.system.n_controls,self.system.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([i.T for i in pre_training_array])).to(device)

        pretrain_infid=[]
        for i,v in enumerate(np.linspace(0,2*pi,self.N)):
            pretrain_infid.append(self.system.get_infid(self.system.euler(gpu_pretraining_data[i]),v))

        return model_infid,torch.tensor(pretrain_infid)

class FrozenNetwork1D(InterpolationNetwork1D):
  def __init__(self, N,layers,width,initial_model,system,transfer_matrix):
    self.N = N
    self.system = system
    self.model = ControlNN(1,layers,width,self.n_controls,n_steps).to(device)
    self.system.H_drives=torch.einsum("ij,jkl->ikl", transfer_matrix.cfloat(), H_drives.cfloat())
    self.model.load_state_dict(initial_model)
    for i in self.model.layers[:-1]:
      i.requires_grad = False

    """
    Transfer learn

    training_pulses (Tensor): Pulses to callibrate to
    batch_size (int): Batch size
    lr (Float): Learning rate
    thresh (Float): Gradient Treshold for stopping

    Returns: Loss History
    """
    def frozen_train(self,training_pulses,lr = 1e-3,thresh=1e-5):
        batch_size = len(training_pulses)
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

class InterpolationNetwork2D(InterpolationNetwork):
    """
    Initialize the InterpolationNetwork.

    N (int): Number of sample points
    layers (int): The number of hidden layers (after the first).
    width (int): The layer width
    system (QuantumSystem): The system being modeled.
    """
    def __init__(self,N,layers,width,system):
        self.N = N
        self.system=system
        self.model = ControlNN(2,layers,width,self.system.n_controls,self.system.n_steps).to(device) 
    """
    Get model accelerations

    theta (Tensor): Theta value

    Returns: accelerations
    """
    def get_accels(self,theta):
        x = self.model(theta) 
        return (x.reshape(self.system.n_steps,self.system.n_controls))
    
    """
    Pretrain the network

    pretraining_path (String): Pretraining path
    lr (Float): Learning rate
    iters (int): Iteration count
    thresh (Float): Threshold

    Returns: Loss data
    """
    def pretrain(self,pretraining_path,lr = 1e-3,iters=10**5,thresh=1e-5):

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.N,self.system.n_controls,self.system.n_steps)
        gpu_pretraining_data = torch.vstack([t for t in torch.tensor(np.array([[pre_training_array[i][j].T for j in range(self.N)] for i in range(self.N)])).to(device)])
        vmap_euler = torch.vmap(func = lambda controls: self.system.euler(controls))
        gpu_pretraining_data=vmap_euler(gpu_pretraining_data).float()
        opt = Adam(self.model.parameters(), lr=lr)
        pretraining_loss_fn  = nn.MSELoss()

        losses=[]

        for epoch in range(iters):
            yhat = torch.vmap(func = lambda t: self.get_controls(t))(torch.cartesian_prod(torch.linspace(0,2*pi,self.N),torch.linspace(0,2*pi,self.N)).to(device))
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
        return np.array(losses)
    """
    (fidelity) Train the network

    lr (Float): Learning rate
    iters (int): Iteration count
    train_size (int): Train size
    test_size (int): Test size
    batch_size (int): Batch size (preferably divisor of train_size)
    c (Float): L1 norm coefficient

    Returns: train losses every iteration, test losses every iteration, and test losses every epoch
    """
    def train(self,lr=5e-5,iters=10**5,train_size=500,test_size=4500,batch_size=100,c=1e-5):
        opt = Adam(self.model.parameters(), lr=5e-4)
        d = int(np.ceil(np.sqrt(test_size)))
        val_thetas = torch.cartesian_prod(torch.linspace(0, 2*pi, d),torch.linspace(0, 2*pi, d))[torch.randperm(d**2)][:test_size].to(device)
        losses=[]
        test_losses = []
        model_history = []
        for epochs in range(iters):
          train_thetas=torch.rand(train_size,2).to(device) * 2 * pi
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
              print("-------------------")
          state = self.model.state_dict()  
          for k, v in state.items():
            state[k] = v.cpu()
          model_history.append(state)
        epoch_ratio = train_size // batch_size
        return np.array(losses),np.array(test_losses),np.array([test_losses[i * epoch_ratio -1] for i in range(1,iters+1)]),model_history

    """
    (fidelity) Active train the network

    lr (Float): Learning rate
    iters (int): Iteration count
    train_size (int): Train size
    test_size (int): Test size
    batch_size (int): Batch size (preferably divisor of train_size)
    c (Float): L1 norm coefficient

    Returns: train losses every iteration, test losses every iteration, and test losses every epoch
    """
    def active_train(self,lr=5e-5,iters=10**5,train_size=500,test_size=4500,batch_size=100,c=1e-5):
        opt = Adam(self.model.parameters(), lr=lr)
        losses=[]
        test_losses = []
        val_thetas=torch.rand(test_size,2).to(device) * 2* np.pi 
        model_history = []
        for epochs in range(iters):
            
            infidelity = torch.vmap(func = lambda t: self.get_model_infid(t))
            temp_losses = infidelity(val_thetas)
            mean_loss = torch.mean(temp_losses)
            max_loss = torch.max(temp_losses)
            train_thetas = torch.rand(int(train_size*max_loss/mean_loss * 1.2),2).to(device) * 2 * pi
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
                print("-------------------")
            state = self.model.state_dict()  
            for k, v in state.items():
              state[k] = v.cpu()
            model_history.append(state)
        epoch_ratio = train_size // batch_size
        return np.array(losses),np.array(test_losses),np.array([test_losses[i * epoch_ratio -1] for i in range(1,iters+1)]),model_history

    """
    Compare model and pretraining control values

    pretraining_path (String): Pretraining path

    Returns: model controls, pretraining controls
    """
    def get_comparison_controls(self,pretraining_path):
        get_controls_list = torch.vmap(torch.vmap(func = lambda t: self.get_controls(t)))
        vmap_euler = torch.vmap(torch.vmap(func = lambda controls: self.system.euler(controls)))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.N,self.system.n_controls,self.system.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([[pre_training_array[i][j].T for j in range(self.N)] for i in range(self.N)])).to(device)

        c_vals=torch.Tensor.cpu(get_controls_list(torch.cartesian_prod(torch.linspace(0,2*pi,self.N),torch.linspace(0,2*pi,self.N)).reshape(self.N,self.N,2).to(device))).detach().numpy()
        c_vals1=torch.Tensor.cpu(vmap_euler(gpu_pretraining_data)).detach().numpy()

        return c_vals,c_vals1

    """
    Compare model and pretraining infidelity values

    pretraining_path (String): Pretraining path

    Returns: model infidelities, pretraining infidelities
    """
    def get_comparison_samples(self,pretraining_path):
        temp_f = torch.vmap(torch.vmap(lambda t: self.get_model_infid(t)))
        model_infid = temp_f(torch.cartesian_prod(torch.linspace(0,2*pi,self.N),torch.linspace(0,2*pi,self.N)).reshape(self.N,self.N,2).to(device))

        pretraining_df = pd.read_csv(pretraining_path)
        pre_training_array = np.array(pretraining_df).T.reshape(self.N,self.N,self.system.n_controls,self.system.n_steps)
        gpu_pretraining_data = torch.tensor(np.array([[pre_training_array[i][j].T for j in range(self.N)] for i in range(self.N)])).to(device)

        pretrain_infid=[]
        for i,v in enumerate(np.linspace(0,2*pi,self.N)):
            for j,v1 in enumerate(np.linspace(0,2*pi,self.N)):
                pretrain_infid.append(self.system.get_infid(self.system.euler(gpu_pretraining_data[i][j]),torch.tensor([v,v1]).to(device)))

        return model_infid,torch.tensor(pretrain_infid)


class FrozenNetwork2D(InterpolationNetwork2D):
  def __init__(self, N,layers,width,initial_model,system,transfer_matrix):
    self.N = N
    self.system = system
    self.model = ControlNN(1,layers,width,self.n_controls,n_steps).to(device)
    self.system.H_drives=torch.einsum("ij,jkl->ikl", transfer_matrix.cfloat(), H_drives.cfloat())
    self.model.load_state_dict(initial_model)
    for i in self.model.layers[:-1]:
      i.requires_grad = False

    """
    Transfer learn

    training_pulses (Tensor): Pulses to callibrate to
    batch_size (int): Batch size
    lr (Float): Learning rate
    thresh (Float): Gradient Treshold for stopping
    train_thetas (Tensor): Training thetas
    
    Returns: Loss History
    """
      
    def frozen_train(self,training_pulses,train_thetas,lr = 1e-3,thresh=1e-5):
        batch_size = len(training_pulses)
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