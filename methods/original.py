import os
import pickle

import torch
from tqdm import tqdm

from data.data_utils import prepare_data
from modules.ClientCluster import ClientCluster
from modules.Server import Server
from utils.prepare_models import prepare_rec_model
from utils.save_utils import construct_weight_path


def original(args):
    device = torch.device(args.device)
    train_loader, test_loader, user_num, item_num, train_mat = prepare_data(args)
    clients_list = []
    for i in tqdm(range(user_num), desc="Preparing client models"):
        clients_list.append(prepare_rec_model(args.model, 1, item_num, train_mat[i:i + 1, :],
                                              local_epochs=args.local_epochs, local_lr=args.local_lr, attr=args.attr))
    client_cluster = ClientCluster(clients_list)
    serve_model = prepare_rec_model(args.model, user_num, item_num, global_lr=args.global_lr,
                                    model_type="serve")
    serve = Server(serve_model, client_cluster, fraction=args.fraction)
    best_hr, best_ndcg = 0, 0
    hr_trace_list = []
    total_loss_dict = {}
    for epoch in range(args.global_epochs):
        loss_list = {}
        serve.train_model(device=device, loss_list=loss_list)
        mean_loss_dict = {k: sum(v) / len(v) for k, v in loss_list.items()}
        for k, v in mean_loss_dict.items():
            if k not in total_loss_dict:
                total_loss_dict[k] = []
            total_loss_dict[k].append(v)
        if epoch % 1 == 0:
            hr_list, ndcg_list, model = client_cluster.evalue(args.model, serve_model, test_loader, \
                                                              user_num, item_num, device)
            hrAT10, ndcgAT10 = hr_list[1], ndcg_list[1]
            hr_trace_list.append(hrAT10)
            args.logger.info(f"epoch {epoch:03d}:")
            for k, v in mean_loss_dict.items():
                args.logger.info(f"{k}: {v:.4f}")
            for idx, topk in zip([0, 1, 2, 3], [5, 10, 15, 20]):
                args.logger.info(f"HR@{topk}: {hr_list[idx]:.4f}\tNDCG@{topk}: {ndcg_list[idx]:.4f}")

            if hrAT10 > best_hr:
                best_hr, best_ndcg, best_epoch = hrAT10, ndcgAT10, epoch
                save_path = construct_weight_path(args)
                torch.save(model, os.path.join(save_path, f'epoch{epoch}.pth'))

        if (epoch + 1) % 50 == 0:
            trace_dict = {"hr_trace": hr_trace_list}
            trace_dict.update({k: v for k, v in total_loss_dict.items()})
            pickle.dump(trace_dict, open(os.path.join(save_path, "trace.pkl"), "wb"))
    args.logger.info(f"End. Best epoch {best_epoch:03d}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}")
