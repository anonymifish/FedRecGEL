import os

import torch
import torch.nn as nn
import numpy as np

from data.data_utils import prepare_data
from utils.save_utils import construct_target_model_path, construct_weight_path
from utils.evaluate import top100_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from attack import main_attack_mutlti, sample_attack_sample
from attack_ensemble import train, EnsembleClassifier
from models.ncf import NCFClient
from models.multvae import MultVAEClient
from models.fedrap import FedRAPClient

def get_label_dict(args, attr):
    file_path = os.path.join(args.dataset_path, f'{args.dataset}/preprocess/user_{attr}.npy')
    label_dict = np.load(file_path, allow_pickle=True).item()
    return label_dict


def add_dp_noise(args, model):
    if args.model == "ncf":
        user_embedding = model.get_user_embedding_weight()
        with torch.no_grad():
            user_embedding += torch.normal(
                mean=0.0,
                std=3,
                size=user_embedding.shape,
                device=user_embedding.device
            )

    elif args.model == "multvae":
        with torch.no_grad():
            for layer in model.q_layers.layers:
                if isinstance(layer, nn.Linear):
                    with torch.no_grad():
                        noise = torch.normal(
                            mean=0.0,
                            std=0.12,
                            size=layer.weight.shape,
                            device=layer.weight.device
                            )
                        layer.weight += noise

    elif args.model == "fedrap":
        for i, emb in enumerate(model.client_embeddings):
            with torch.no_grad():
                weight = emb.weight

                noise = torch.normal(
                    mean=0.0,
                    std=0.1,
                    size=weight.shape,
                    device=weight.device
                    )
                
                emb.weight.add_(noise)


def dp(args):
    device = torch.device(args.device)
    target_model_path = f'./exp_results/{args.model}/{args.dataset}/original/'
    train_loader, test_loader, user_num, item_num, train_mat = prepare_data(args)
    for epoch in range(1):
        model = torch.load(construct_target_model_path(args, target_model_path, args.target_model), weights_only=False, map_location=device)
        model.to(device)
        add_dp_noise(args, model)

        hr_list, ndcg_list = top100_metrics(model, test_loader)
        args.logger.info(f"epoch {epoch:03d}:")
        for idx, topk in zip([0, 1, 2, 3], [5, 10, 15, 20]):
            args.logger.info(f"HR@{topk}: {hr_list[idx]:.4f}\tNDCG@{topk}: {ndcg_list[idx]:.4f}")

        if args.model== 'ncf' or args.model == 'multvae':
            user_mat = model.get_user_embedding_weight().to("cpu").detach().numpy()
            for attr in ['gender', 'age']:
                args.logger.info('loading label user dict...')
                label_dict = get_label_dict(args, attr)
                label_user_dict = sample_attack_sample(label_dict, user_mat.shape[0], 4, args.logger)

                main_attack_mutlti(user_mat, label_user_dict, args)

        elif args.model == 'fedrap':
            user_mat = model.get_user_embedding_matrix()
            for attr in ['gender', 'age']:
                args.logger.info('loading label user dict...')
                label_dict = get_label_dict(args, attr)
                label_user_dict = sample_attack_sample(label_dict, len(user_mat), 4, args.logger)

                all_indices = []
                all_labels = []
                for label, users in label_user_dict.items():
                    all_indices.extend(users)
                    all_labels.extend([label] * len(users))  # label 应该是编号后的

                all_labels = torch.tensor(all_labels)
                all_indices = torch.tensor(all_indices)

                num_rounds = 3
                f1_list = []
                bacc_list = []

                for round_idx in range(num_rounds):
                    args.logger.info(f"\n=== Round {round_idx + 1} ===")
                    train_indices, test_indices, train_labels, test_labels = train_test_split(
                        all_indices, all_labels, test_size=0.8, stratify=all_labels)

                    num_classes = len(torch.unique(all_labels))
                    train_labels = train_labels.to(device)
                    test_labels = test_labels.to(device)

                    train_user_mat = torch.stack([user_mat[i].weight for i in train_indices]).to(device)
                    test_user_mat = torch.stack([user_mat[i].weight for i in test_indices]).to(device)

                    embed_dim = user_mat[0].weight.shape[1]
                    num_items = user_mat[0].weight.shape[0]
                    model = EnsembleClassifier(embed_dim=embed_dim, num_items=num_items, num_classes=num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    loss_fn = nn.CrossEntropyLoss()

                    train(model, train_user_mat, train_labels, optimizer, loss_fn, test_user_mat, test_labels, epochs=80, logger=args.logger)

                    # 评估本轮的Bacc和F1
                    model.eval()
                    with torch.no_grad():
                        outputs = model(test_user_mat)
                        preds = torch.argmax(outputs, dim=1)

                        bacc = balanced_accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
                        f1 = f1_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                        args.logger.info(f"[Round {round_idx+1}] Bacc: {bacc:.4f}, F1: {f1:.4f}")
                        bacc_list.append(bacc)
                        f1_list.append(f1)
                    model.train()

                avg_bacc = sum(bacc_list) / num_rounds
                avg_f1 = sum(f1_list) / num_rounds
                args.logger.info(f"\n=== {num_rounds}轮平均结果 ===")
                args.logger.info(f"平均Bacc: {avg_bacc:.4f}, 平均F1: {avg_f1:.4f}")