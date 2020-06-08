import numpy as np
import torch
import configs.dann_config as dann_config


def _loss_DADA_splitted(
        lambda_,
        epoch,
        class_logits_on_src,
        class_logits_on_trg,
        true_labels_on_src,
        true_labels_on_trg,
        unk_value=dann_config.UNK_VALUE,
        device=torch.device('cpu')):

    true_labels_on_src = torch.as_tensor(true_labels_on_src).long()

    if epoch > dann_config.NUM_EPOCH_PRETRAIN:
        probs_all_src = torch.nn.Softmax(-1)(class_logits_on_src)
        loss_source = - torch.mean(
            (torch.ones(len(class_logits_on_src), dtype=torch.long, device=device) - probs_all_src[:, -1]) * \
            torch.log(probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src] + 10e-6) +
            probs_all_src[:, -1] *
            torch.log(
                (torch.ones(len(class_logits_on_src), dtype=torch.long, device=device) -
                 probs_all_src[torch.arange(probs_all_src.size(0)), true_labels_on_src.flatten()]) + 10e-6)
        )

    # loss target
        probs_all_trg = torch.nn.Softmax(-1)(class_logits_on_trg)
        probs_real_trg = torch.div(
            probs_all_trg, torch.ones((len(probs_all_trg), 1), dtype=torch.long, device=device) -
                           probs_all_trg[:, -1].reshape(-1,1)
        )
        probs_real_trg[:, -1] = 0

        probs_trg_hat = (probs_all_trg / (probs_all_trg + probs_all_trg[:, -1].reshape(-1, 1) + 10e-6))[:, :-1]
        loss_trg_classifier = - torch.sum(torch.log(probs_trg_hat + 10e-6) * probs_real_trg[:, :-1], dim=-1).mean()
        loss_trg_generator = torch.sum(
            probs_real_trg[:, :-1] * torch.log(torch.ones_like(probs_trg_hat) - probs_trg_hat + 10e-6), dim=-1).mean()

        entropy_loss_on_trg = torch.sum(torch.log(probs_real_trg + 10e-6) * probs_real_trg, dim=-1).mean()

    else:
        crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value, reduction='mean')
        loss_source = crossentropy(class_logits_on_src, true_labels_on_src)

        loss_trg_classifier = torch.zeros([1], dtype=torch.long, device=device)
        loss_trg_generator = torch.zeros([1], dtype=torch.long, device=device)
        entropy_loss_on_trg = torch.zeros([1], dtype=torch.long, device=device)

    loss_min = lambda_ * (loss_source + loss_trg_classifier) + entropy_loss_on_trg
    loss_max = lambda_ * (loss_source + loss_trg_generator) - entropy_loss_on_trg

    return [loss_min, loss_max], {
            "classifier_loss_on_src": loss_source.data.cpu().item(),
            "loss_trg_classifier": loss_trg_classifier.data.cpu().item(),
            "loss_trg_generator": loss_trg_generator.data.cpu().item(),
            "entropy_loss_on_trg": - entropy_loss_on_trg.data.cpu().item(),
            "loss_min": loss_min.data.cpu().item(),
            "loss_max": loss_max.data.cpu().item(),
            "lambda": lambda_
    }


def calc_lambda(current_iteration,
                        total_iterations,
                        gamma=dann_config.LOSS_GAMMA):

    if current_iteration > dann_config.NUM_EPOCH_PRETRAIN:
        progress = (current_iteration - dann_config.NUM_EPOCH_PRETRAIN) / \
                   (total_iterations - dann_config.NUM_EPOCH_PRETRAIN)
        lambda_p = 2 / (1 + np.exp(-gamma * progress)) - 1
        return lambda_p
    return 1


def loss_DADA(model,
              batch,
              epoch,
              n_epochs,
              device=torch.device('cpu')):

    lambda_ = calc_lambda(epoch, n_epochs)
    
    model_output = model.forward(batch['src_images'].to(device))
    class_logits_on_src = model_output['class']

    model_output = model.forward(batch['trg_images'].to(device))
    class_logits_on_trg = model_output['class']

    return _loss_DADA_splitted(
        lambda_,
        epoch,
        class_logits_on_src,
        class_logits_on_trg,
        true_labels_on_src=batch['src_classes'],
        true_labels_on_trg=batch['trg_classes'],
        device=device)
