import numpy as np
import torch
import configs.dann_config as dann_config


def _loss_DANNCA_splitted(
        class_logits_on_src,
        class_logits_on_trg,
        true_labels_on_src,
        true_labels_on_trg,
        domain_loss_weight,
        unk_value=dann_config.UNK_VALUE,
        device=torch.device('cpu'),
):
    # TARGET_DOMAIN_IDX is 1
    source_len = len(class_logits_on_src)
    target_len = len(class_logits_on_trg)
    true_labels_on_src = torch.as_tensor(true_labels_on_src).long()
    if dann_config.IS_UNSUPERVISED:
        true_labels_on_trg = unk_value * torch.ones(target_len, dtype=torch.long, device=device)
    else:
        true_labels_on_trg = torch.as_tensor(true_labels_on_trg).long()

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value, reduction='mean')

    classifier_loss_on_src = crossentropy(class_logits_on_src, true_labels_on_src)
    classifier_loss_on_trg = - torch.mean(torch.log_softmax(class_logits_on_trg, dim=-1)[:, -1])

    if dann_config.ENTROPY_REG:
        probs_on_trg = torch.nn.Softmax(-1)(class_logits_on_trg[:, : -1])
        entropy_loss_on_trg = - dann_config.ENTROPY_REG_COEF * \
                              torch.sum(torch.log(probs_on_trg) * probs_on_trg, dim=1).mean()
        classifier_loss_on_trg += entropy_loss_on_trg

    classifier_loss = classifier_loss_on_src + classifier_loss_on_trg

    feature_loss_on_src = crossentropy(class_logits_on_src[:, : -1], true_labels_on_src)
    feature_loss_on_trg = - torch.mean(torch.log(torch.ones_like(true_labels_on_trg) -
                                                 torch.softmax(class_logits_on_trg, dim=-1)[:, -1]))
    if dann_config.ENTROPY_REG:
        feature_loss_on_trg -= dann_config.ENTROPY_REG_COEF * entropy_loss_on_trg
    feature_loss = domain_loss_weight * (feature_loss_on_src +  feature_loss_on_trg)

    return [classifier_loss, feature_loss], {
            "classifier_loss_on_src": classifier_loss_on_src.data.cpu().item(),
            "classifier_loss_on_trg": classifier_loss_on_trg.data.cpu().item(),
            "feature_loss_on_src": feature_loss_on_src.data.cpu().item(),
            "feature_loss_on_trg": feature_loss_on_trg.data.cpu().item(),
            "classifier_loss": classifier_loss.data.cpu().item(),
            "feature_loss": feature_loss.data.cpu().item()
        }


def calc_domain_loss_weight(current_iteration,
                        total_iterations,
                        gamma=dann_config.LOSS_GAMMA):
    progress = current_iteration / total_iterations
    lambda_p = 2 / (1 + np.exp(-gamma * progress)) - 1
    return lambda_p

def loss_DANNCA(model,
              batch,
              epoch,
              n_epochs,
              device=torch.device('cpu')):

    model_output = model.forward(batch['src_images'].to(device))
    class_logits_on_src = model_output['class']

    model_output = model.forward(batch['trg_images'].to(device))
    class_logits_on_trg = model_output['class']

    domain_loss_weight = calc_domain_loss_weight(epoch, n_epochs)

    return _loss_DANNCA_splitted(
        class_logits_on_src,
        class_logits_on_trg,
        true_labels_on_src=batch['src_classes'],
        true_labels_on_trg=batch['trg_classes'],
        domain_loss_weight=domain_loss_weight,
        device=device)
