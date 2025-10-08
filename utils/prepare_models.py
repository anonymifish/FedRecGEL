from models.fedrap import FedRAPClient, FedRAPServer
from models.ncf import NCFClient, NCFServer


def prepare_rec_model(model, num_user, num_item, user_interaction=None,
                      train_dataset=None, dataset_path=None, local_epochs=1,
                      local_lr=0.001, global_lr=0.001, model_type="client",
                      attr=None):
    if model == 'ncf':
        if model_type == "client":
            return NCFClient(user_interaction, local_epochs, local_lr, attr=attr)
        else:
            return NCFServer(item_num=num_item, global_lr=0.001)
    elif model == 'fedrap':
        if model_type == "client":
            return FedRAPClient(user_interaction, local_epochs, local_lr, num_item, attr=attr)
        else:
            return FedRAPServer(num_items=num_item, global_lr=global_lr, local_lr=local_lr)
    else:
        raise ValueError(f'model {model} not supported')
