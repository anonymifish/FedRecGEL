from models.fedrecgel import FedRecGELClient, FedRecGELServer


def prepare_rec_model(model, num_user, num_item, user_interaction=None,
                      train_dataset=None, dataset_path=None, local_epochs=1,
                      local_lr=0.001, global_lr=0.001, model_type="client",
                      attr=None):
    if model == 'fedrecgel':
        if model_type == "client":
            return FedRecGELClient(user_interaction, local_epochs, local_lr, attr=attr)
        else:
            return FedRecGELServer(item_num=num_item, global_lr=0.001)
    else:
        raise ValueError(f'model {model} not supported')
