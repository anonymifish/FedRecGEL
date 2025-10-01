import torch
import torch.nn as nn
from torch.autograd import Function
from utils.model_utils import PolyLinearWithPrecode, PolyLinear
import numpy as np
import copy

class GRL_(Function):

    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # need to return a gradient for each input parameter of the forward() function
        # for parameters that don't require a gradient, we have to return None
        # see https://stackoverflow.com/a/59053469
        return -ctx.grad_scaling * grad_output, None



class BaseAdv(nn.Module):
    def __init__(self, model, model_class, grad_scaling, h_out_dim):
        super().__init__()
        self.model = model
        self.model_class = model_class
        self.grl = GRL_.apply
        self.grad_scaling = grad_scaling
        self.h_laten_dim = 100
        self.h_layer_size = [self.h_laten_dim, h_out_dim]


    def eval_advloss(self, adv, attr):
        device = next(self.model.parameters()).device
        attr = attr.to(device)  
        loss_f = nn.CrossEntropyLoss()
        if hasattr(self.adv, 'precode'):
            adv_loss = loss_f(adv, attr) + self.adv.precode.kl_loss
        else:
            adv_loss = loss_f(adv, attr)
        return adv_loss

    def compute_adv_loss(self, user, attr, loss_list):
        z = self.model.get_user_embedding(user)
        h = self.grl(z, self.grad_scaling)
        adv_output = self.adv(h)
        adv_loss = self.eval_advloss(adv_output, attr)
        
        if 'adv_loss' not in loss_list:
            loss_list['adv_loss'] = []
        loss_list['adv_loss'].append(adv_loss.item())
        
        # Convert one-hot encoded attr to class indices
        attr_indices = torch.argmax(attr)
        
        # Calculate accuracy
        predicted = torch.argmax(adv_output)
        correct = (predicted == attr_indices)
        accuracy = correct.item()
        
        if 'adv_accuracy' not in loss_list:
            loss_list['adv_accuracy'] = []
        loss_list['adv_accuracy'].append(accuracy)

        if 'predict_num' not in loss_list:
            loss_list['predict_num'] = []
        if accuracy:
            loss_list['predict_num'].append(predicted.item())
        
        return adv_loss, accuracy


class BaseAdvClient(BaseAdv):
    def __init__(self, model, model_class, grad_scaling, h_out_dim):
        super().__init__(model, model_class, grad_scaling, h_out_dim)


    def recieve_server_modules(self, serve_modules):
        self.adv = copy.deepcopy(serve_modules[0])
        self.model.recieve_server_modules(serve_modules[1:])

    def get_server_modules(self):
        adv_serve_modules = torch.nn.ModuleList([self.adv])
        for serve_module in self.model.get_server_modules():
            adv_serve_modules.append(serve_module)
        return adv_serve_modules

    def del_server_modules(self):
        del self.adv
        self.model.del_server_modules()

    def fit_client_epochs(self, loss_list):
        clone_adv_model = copy.deepcopy(self.get_server_modules()[0])
        self.train()
        optimizer = torch.optim.Adam(self.get_server_modules().parameters(), lr=self.model.local_lr)
        for epoch in range(self.model.local_epochs):
            optimizer.zero_grad()
            if hasattr(self.model, 'client_optimizer'):
                self.model.client_optimizer.zero_grad()
            rec_loss = self.model.calculate_loss(loss_list)
            adv_loss, accuracy = self.compute_adv_loss(self.model.uid, self.model.attr, loss_list)
            loss = adv_loss + rec_loss
            if 'adv_total_loss' not in loss_list:
                loss_list['adv_total_loss'] = []
            loss_list['adv_total_loss'].append(loss.item())
            adv_loss.backward()
            
            if not accuracy:
                for user_embedding_grad in self.model.get_user_embedding_grad():
                    user_embedding_grad *= 0
            modules_grad_list = [p.grad.clone().detach() for p in self.get_server_modules()[0].parameters()]
            
            rec_loss.backward()
            optimizer.step()
            if hasattr(self.model, 'client_optimizer'):
                self.model.client_optimizer.step()
            if hasattr(self.model, 'update_count'):
                self.model.update_count += 1

        for param in self.get_server_modules()[1:].parameters():
            modules_grad_list.append(param.grad.clone().detach())
        self.adv_grad = [
            [param.grad.clone().detach().to('cpu') for param in self.get_server_modules()[0].parameters()],
            clone_adv_model.to('cpu'),
            self.model.attr.to('cpu'),
            self.model.get_user_embedding_weight().clone().detach().to('cpu')
            ]
        self.del_server_modules()
        return modules_grad_list

class BaseAdvServe(BaseAdv):
    def __init__(self, adv_lr, mu_std, precode, *args):
        super().__init__(*args)
        self.embeding_dim = self.model.get_embedding_dim()
        if precode == "on":
            self.adv = PolyLinearWithPrecode([self.embeding_dim] + self.h_layer_size, nn.ReLU(), mu_std=mu_std)
        else:
            self.adv = PolyLinear([self.embeding_dim] + self.h_layer_size, nn.ReLU())
        self.serve_optimizer = torch.optim.Adam(self.get_server_modules()[1:].parameters(), lr=self.model.global_lr)
        self.adv_optimizer = torch.optim.Adam(self.get_server_modules()[0].parameters(), lr=adv_lr)
        self.init_serve_weights()

    def init_serve_weights(self):
        self.model.init_serve_weights()
        for layer in self.adv.modules():
            if isinstance(layer, nn.Linear):
                # Xavier Initialization for weights
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)

                # Normal Initialization for Biases
                if layer.bias is not None:
                    layer.bias.data.normal_(0.0, 0.001)

    def get_server_modules(self):
        adv_serve_modules = torch.nn.ModuleList([self.adv])
        for serve_module in self.model.get_server_modules():
            adv_serve_modules.append(serve_module)
        return adv_serve_modules
    
    def recieve_server_modules_grad(self, modules_grad_list, select_user_num):
        for serve_modules, *client_modules in zip(self.get_server_modules().parameters(), *modules_grad_list):
            if serve_modules.grad is None:
                serve_modules.grad = torch.zeros_like(serve_modules)
            serve_modules.grad += sum(client_modules) / select_user_num

    def fit_serve_epoch(self):
        self.serve_optimizer.step()
        self.adv_optimizer.step()
        self.serve_optimizer.zero_grad()
        self.adv_optimizer.zero_grad()
