import torch

from utils.evaluate import top100_metrics
from utils.prepare_models import prepare_rec_model


class ClientCluster:
    def __init__(self, clients_list):
        self.clients_list = torch.nn.ModuleList(clients_list)

    def receive_server_modules(self, server_modules, c_list):
        for client_index in c_list:
            client = self.clients_list[client_index]
            client.recieve_server_modules(server_modules)

    def send_serve_modules_grad(self, c_list, loss_list):
        grad_modules_list = []
        for client_index in c_list:
            client = self.clients_list[client_index]
            grad = client.fit_client_epochs(loss_list)
            grad_modules_list.append(grad)
        return grad_modules_list

    def evalue(self, model, serve_model, test_loader, user_num, item_num, device):
        refactor_model = prepare_rec_model(model, user_num, item_num)
        refactor_model.refactoring(serve_model, self.clients_list)
        hr_list, ndcg_list = top100_metrics(refactor_model.to(device), test_loader)
        return hr_list, ndcg_list, refactor_model
