import numpy as np
import torch
import configs.dann_config as dann_config


def class_prediction_loss(model, batch, device=torch.device('cpu'),
                          unk_value=dann_config.UNK_VALUE, **kwargs):
    """
    :param model: model.forward(images) should return dict with keys
        'class' : Tensor, shape = (batch_size, n_classes)  logits  of classes (raw, not logsoftmax)
    :param batch: dict with keys
        'src_images':
        'src_classes':np.Array, shape = (batch_size,)
    if true_class is unknown, then class should be dann_config.UNK_VALUE
    :return:
        loss: torch.Tensor,
        dict: {}
    """
    class_logits_on_src = model.forward(batch['src_images'].to(device))['class']
    true_labels_on_src = batch['src_classes']

    crossentropy = torch.nn.CrossEntropyLoss(ignore_index=unk_value, reduction='mean')
    prediction_loss = crossentropy(class_logits_on_src, true_labels_on_src)
    return prediction_loss, {}
