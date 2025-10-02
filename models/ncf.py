import copy
import random

import torch
import torch.nn as nn

from models.base_federated import BaseServerModel, BaseClientModel


class NCF(nn.Module):
    def __init__(self):
        super(NCF, self).__init__()

    def forward(self, user, item):
        embed_user_gmf = self.embed_user_gmf(user)
        embed_item_gmf = self.embed_item_gmf(item)
        output_gmf = embed_user_gmf * embed_item_gmf

        embed_user_mlp = self.embed_user_mlp(user)
        embed_item_mlp = self.embed_item_mlp(item)
        interaction = torch.cat((embed_user_mlp, embed_item_mlp), -1)
        output_mlp = self.mlp_layers(interaction)

        concat = torch.cat((output_gmf, output_mlp), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def get_embedding_dim(self):
        return self.embed_item_gmf.weight.shape[1]

    def get_user_embedding(self, users):
        return self.embed_user_gmf(users)

    def get_user_embedding_weight(self):
        return self.embed_user_gmf.weight

    def get_user_embedding_grad(self):
        return [self.get_user_embedding_weight().grad]

class NCFServer(NCF, BaseServerModel):
    def __init__(self, item_num, factor_num=32, num_layers=3, dropout=0.0, global_lr=0.001):
        NCF.__init__(self)
        BaseServerModel.__init__(self, global_lr)
        self.embed_item_gmf = nn.Embedding(item_num, factor_num)
        self.embed_item_mlp = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))

        self.dropout = dropout
        mlp_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self.serve_optimizer = torch.optim.Adam(self.get_server_modules().parameters(), lr=global_lr)
        self.init_serve_weights()

    def init_serve_weights(self):
        nn.init.normal_(self.embed_item_gmf.weight, std=0.01)
        nn.init.normal_(self.embed_item_mlp.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
            
    def get_server_modules(self):
        return torch.nn.ModuleList([self.embed_item_gmf, self.embed_item_mlp, self.mlp_layers, self.predict_layer])
    
    def recieve_server_modules_grad(self, modules_grad_list, select_user_num):
        for serve_modules, *client_modules in zip(self.get_server_modules().parameters(), *modules_grad_list):
            if serve_modules.grad is None:
                serve_modules.grad = torch.zeros_like(serve_modules)
            serve_modules.grad += sum(client_modules) / select_user_num

    def fit_serve_epoch(self):
        self.serve_optimizer.step()
        self.serve_optimizer.zero_grad()

class NCFClient(NCF, BaseClientModel):
    def __init__(self, user_interaction, local_epochs, local_lr, uid=torch.tensor(0), factor_num=32, num_layers=3, attr=None):
        NCF.__init__(self)
        BaseClientModel.__init__(self, local_epochs, local_lr)
        self.embed_user_gmf = nn.Embedding(1, factor_num)
        self.embed_user_mlp = nn.Embedding(1, factor_num * (2 ** (num_layers - 1)))
        self.client_optimizer = torch.optim.Adam(self.get_client_modules().parameters(), lr=self.local_lr)
        self.register_buffer("uid", uid)
        self.register_buffer("interaction", user_interaction)
        self.register_buffer("attr", attr)
        self.init_client_weight()

    def init_client_weight(self):
        nn.init.normal_(self.embed_user_gmf.weight, std=0.01)
        nn.init.normal_(self.embed_user_mlp.weight, std=0.01)

    def sample_user_pos_neg(self, num_negatives=4):
        device = self.interaction.device
        user_interaction = self.interaction.bool().squeeze()
        pos_items = torch.nonzero(user_interaction).flatten().tolist()
        neg_items = torch.nonzero(~user_interaction).flatten().tolist()

        random.shuffle(pos_items)

        users, items, labels = [], [], []
        for pos_item in pos_items:
            users.append(0)
            items.append(pos_item)
            labels.append(1)
            for neg_item in random.sample(neg_items, num_negatives):
                users.append(0)
                items.append(neg_item)
                labels.append(0)

        return (
            torch.tensor(users).to(device),
            torch.tensor(items).to(device),
            torch.tensor(labels).float().to(device)
        )

    def calculate_loss(self, loss_list):
        loss_fn = nn.BCEWithLogitsLoss()
        users, items, labels = self.sample_user_pos_neg()
        
        bs = 128
        n_samples = len(users)
        total_loss = 0
        
        for i in range(0, n_samples, bs):
            u_batch = users[i:i + bs]
            i_batch = items[i:i + bs]
            l_batch = labels[i:i + bs]
            pred = self.forward(u_batch, i_batch)
            loss = loss_fn(pred, l_batch.float())
            total_loss += loss

        if 'ncf_total_loss' not in loss_list:
            loss_list['ncf_total_loss'] = []
        loss_list['ncf_total_loss'].append(total_loss.item())

        return total_loss

    def fit_client_epochs(self, loss_list):
        self.train()
        serve_optimizer = torch.optim.Adam(self.get_server_modules().parameters(), lr=self.local_lr)
        for epoch in range(self.local_epochs):
            self.client_optimizer.zero_grad()
            serve_optimizer.zero_grad()
            loss = self.calculate_loss(loss_list)
            loss.backward()
            self.client_optimizer.step()
            serve_optimizer.step()
        modules_grad_list = [p.grad.clone().detach() for p in self.get_server_modules().parameters()]
        self.del_server_modules()
        return modules_grad_list
    
    def recieve_server_modules(self, serve_modules):
        self.embed_item_gmf = copy.deepcopy(serve_modules[0])
        self.embed_item_mlp = copy.deepcopy(serve_modules[1])
        self.mlp_layers = copy.deepcopy(serve_modules[2])
        self.predict_layer = copy.deepcopy(serve_modules[3])

    def get_server_modules(self):
        return torch.nn.ModuleList([self.embed_item_gmf, self.embed_item_mlp, self.mlp_layers, self.predict_layer])
 
    def get_client_modules(self):
        return torch.nn.ModuleList([self.embed_user_gmf, self.embed_user_mlp])

    def refactoring(self, serve_model: NCFServer, clients_list):
        # Get total number of users
        total_users = len(clients_list)
        # Get embedding dimensions from the first client
        factor_num = clients_list[0].embed_user_gmf.weight.shape[1]
        mlp_factor_num = clients_list[0].embed_user_mlp.weight.shape[1]
        
        # Create new embeddings for all users
        self.embed_user_gmf = nn.Embedding(total_users, factor_num)
        self.embed_user_mlp = nn.Embedding(total_users, mlp_factor_num)
        
        # Copy embeddings from each client
        for i, client in enumerate(clients_list):
            self.embed_user_gmf.weight.data[i] = client.embed_user_gmf.weight.data[0]
            self.embed_user_mlp.weight.data[i] = client.embed_user_mlp.weight.data[0]
            
        # Copy server modules
        self.embed_item_gmf = serve_model.embed_item_gmf
        self.embed_item_mlp = serve_model.embed_item_mlp
        self.mlp_layers = serve_model.mlp_layers
        self.predict_layer = serve_model.predict_layer

    
    def del_server_modules(self):
        del self.embed_item_gmf
        del self.embed_item_mlp
        del self.mlp_layers
        del self.predict_layer