import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_utils import get_real_label_one_hot
from utils.save_utils import construct_logger, construct_weight_path_wocheck, construct_weight_path


def initialize_settings(path):
    class Args:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

        def add_attribute(self, k, v):
            setattr(self, k, v)

        def record_config(self, logger):
            for attr in self.__dict__:
                logger.info(f'{attr}: {getattr(self, attr)}')

    with open(path, 'r') as f:
        configuration = json.load(f)
    return Args(configuration)



# After optimization, compare predicted label with real label
def calculate_result(args, model, grad_list, y_dummy, user_label, x_dummy, user_idx, logger):
    
    real_label_class = torch.argmax(user_label, dim=0)
    
    def get_h_prime(module, input, output):
        global h_prime
        h_prime = output

    hook = model.layers[-2].register_forward_hook(get_h_prime)
    logits = model(x_dummy)
    y_dummy_pred = F.softmax(logits, dim=1)

    # if args.task_name == 'vae' or args.task_name == 'dsvae':
    delta_Wi = grad_list[0][real_label_class]
    # else:
    #     delta_Wi = grad_list[-2][real_label_class]
    yi_dummy_pred = y_dummy_pred[0][real_label_class]
    Delta_i = torch.matmul(h_prime, delta_Wi) / (torch.norm(h_prime, p=2) ** 2)
    hook.remove()
    
    y_pred_prob = F.softmax(y_dummy, dim=1)
    y_pred_class = torch.argmax(y_pred_prob, dim=1)
    yi_dummy = y_dummy[0][real_label_class]

    is_correct = (y_pred_class == real_label_class).item()


    logger.info(f"User {user_idx}: yi_dummy_pred = {yi_dummy_pred.item():.4f}, "
                f"Delta_i = {Delta_i.item():.4f}, "
                f"yi_dummy = {yi_dummy.item():.4f}, "
                f"Correct = {is_correct}")

    return is_correct, yi_dummy_pred.item(), Delta_i.item(), yi_dummy.item()



if __name__ == '__main__':
    config_path = 'configs/attack_grad_config.json'
    args = initialize_settings(config_path)

    save_path = f'./exp_results/{args.model}/{args.dataset}/{args.method}/'

    if not os.path.exists(save_path):
        print('attack model not exist')
    else:

        args.add_attribute("save_path", save_path)
        weight_folder_path = construct_weight_path(args)
        logger = construct_logger(args)
        args.record_config(logger)
        args.add_attribute("logger", logger)
        model_path = construct_weight_path_wocheck(args)
        if not os.path.exists(model_path):
            args.logger.info(f'Target model path {model_path} does not exist')
        else:
            args.add_attribute("target_model_path", model_path)

        args.logger.info('loading user grad...')
        grad_and_model = torch.load(args.target_model_path, weights_only=False)
        for i in range(len(grad_and_model)):
            try:
                embedding_dim, laten_dim, out_dim = grad_and_model[i][1].layer_config
                break
            except:
                continue
        device = args.device if hasattr(args, 'device') else 'cpu'

        total_correct = 0
        total_yi_dummy_pred = 0
        total_Delta_i = 0
        total_yi_dummy = 0
        total_embedding_diff = 0
        total_users = 0
        
        for user_idx, (grad_list, model, user_label, user_embedding) in enumerate(grad_and_model):
            # Move model to the specified device
            # Check if the first gradient tensor is all zeros
            if grad_list == 0:
                logger.info(f"User {user_idx}: Skipped due to zero gradients")
                continue

            total_users += 1
            model = model.to(device)
            
            # Create dummy data on the specified device
            x_dummy = torch.randn((1, embedding_dim), requires_grad=True, device=device)
            y_dummy = torch.randn((1, out_dim), requires_grad=True, device=device)
            user_label = user_label.to(device)

            # Optimizer for updating x and y
            optimizer = torch.optim.Adam([x_dummy, y_dummy], lr=0.1)

            for step in range(300):
                optimizer.zero_grad()
                logits = model(x_dummy)
                loss_f = nn.CrossEntropyLoss()
                loss = loss_f(logits, y_dummy)

                dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                grad_list = [g.to(device) if g.device != device else g for g in grad_list]
                
                grad_diff = 0
                for g_fake, g_real in zip(dummy_grads[:], grad_list[:]):
                    grad_diff += ((g_fake - g_real)**2).sum()

                grad_diff.backward()
                optimizer.step()

            # Evaluate prediction accuracy and get embedding difference
            is_correct, yi_dummy_pred, Delta_i, yi_dummy = calculate_result(args, model, grad_list, y_dummy, user_label, x_dummy, user_idx, args.logger)
            
            if math.isnan(Delta_i):
                total_users -= 1
                continue

            total_correct += is_correct
            total_yi_dummy_pred += yi_dummy_pred
            total_Delta_i += Delta_i
            total_yi_dummy += yi_dummy
            
            # Log accuracy and average embedding difference every 50 users
            if (user_idx + 1) % 50 == 0:
                accuracy_so_far = total_correct / (user_idx + 1)
                avg_yi_dummy_pred = total_yi_dummy_pred / (user_idx + 1)
                avg_Delta_i = total_Delta_i / (user_idx + 1)
                avg_yi_dummy = total_yi_dummy / (user_idx + 1)
                logger.info(f"After {user_idx + 1} users: Accuracy = {accuracy_so_far:.4f}, "
                           f"Avg yi_dummy_pred = {avg_yi_dummy_pred:.4f}, "
                           f"Avg Delta_i = {avg_Delta_i:.4f}, "
                           f"Avg yi_dummy = {avg_yi_dummy:.4f}")

        # Calculate and log the final accuracy and average embedding difference
        accuracy = total_correct / total_users
        avg_embedding_diff_final = total_embedding_diff / total_users
        avg_yi_dummy_pred_final = total_yi_dummy_pred / total_users
        avg_Delta_i_final = total_Delta_i / total_users
        avg_yi_dummy_final = total_yi_dummy / total_users
        
        args.logger.info(f"Final results: Accuracy = {accuracy:.4f} ({total_correct}/{total_users}), "
                        f"Avg Embedding Diff = {avg_embedding_diff_final:.4f}, "
                        f"Avg yi_dummy_pred = {avg_yi_dummy_pred_final:.4f}, "
                        f"Avg Delta_i = {avg_Delta_i_final:.4f}, "
                        f"Avg yi_dummy = {avg_yi_dummy_final:.4f}")


