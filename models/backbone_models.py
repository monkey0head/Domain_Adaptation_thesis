import torch.nn as nn

import configs.dann_config as dann_config


def split_layers(model, layer_ids):
    """
    Args:
        model (nn.Module) - nn.Sequential model
        layer_ids (list of int) - required model outputs
    Return:
        layers (list of nn.Module)- model splitted in parts

    Function for splitting nn.Sequential model in separate
    layers, which indexed by numbers from the list layer_ids.
    It will be required to obtain activations of intermediate model layers
    and calculate loss function between intermediate layers
    """
    separate_layers = list(model)
    layers = []
    for i in range(len(layer_ids)):
        if i == 0:
            start_layer = 0
        else:
            start_layer = layer_ids[i - 1] + 1
        layers.append(nn.Sequential(*separate_layers[start_layer:layer_ids[i] + 1]))
    if len(separate_layers) > layer_ids[-1] + 1:
        layers.append(nn.Sequential(*separate_layers[layer_ids[-1] + 1:]))        
    return layers


def get_backbone_model():
    """
    Return:
        features (nn.Module) - convolutional model part for feature extracting
        pooling (nn.Module) - model pooling layers
        classifier (nn.Module) - model fully connected layers
        domain_input_len (int) - size of domain classifier input
        classifier_before_domain_cnt (int) - amount of layers to skip in classifier

    Returns three parts of model — the convolutional part to extract features,
    part with pooling and part with fully connected model layers.
    Can return these parts with pre-trained weights for standard architecture.
    """
    if dann_config.MODEL_BACKBONE == "alexnet":
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_alexnet()
    elif dann_config.MODEL_BACKBONE == "resnet50":
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_resnet50()
    elif dann_config.MODEL_BACKBONE == "resnet50_rich":
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_resnet50_rich_classifier()
    elif dann_config.MODEL_BACKBONE == 'DANN-CA_simple':
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_resnet50_for_DANN_CA()
    elif dann_config.MODEL_BACKBONE == 'DANN-CA_rich':
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_resnet50_for_DANN_CA_rich()
    elif dann_config.MODEL_BACKBONE == 'vanilla_dann' and not dann_config.BACKBONE_PRETRAINED:
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_vanilla_dann()
    elif dann_config.MODEL_BACKBONE == 'mnist_dann' and not dann_config.BACKBONE_PRETRAINED:
        features, pooling, classifier, \
            classifier_layer_ids, domain_input_len, classifier_before_domain_cnt = get_mnist_dann()  
    else:
        raise RuntimeError("model %s with pretrained = %s, does not exist" \
            % (dann_config.MODEL_BACKBONE, dann_config.BACKBONE_PRETRAINED))

    classifier = nn.ModuleList(split_layers(classifier, classifier_layer_ids))
    if dann_config.FREEZE_BACKBONE_FEATURES and dann_config.FREEZE_LEVEL is not None:
        if dann_config.FREEZE_LEVEL > 10 and dann_config.MODEL_BACKBONE == "alexnet":
            raise ValueError('FREEZE_LEVEL greater than number of layers')
        for i, param in enumerate(features.parameters()):
            # possible for AlexNet: 0, 2, 4, 6, 8, 10
            # possible for ResNet: 0, 1, 3, 33, 72, 129, 141, 159
            if i < dann_config.FREEZE_LEVEL:
                param.requires_grad = False
    return features, pooling, classifier, domain_input_len, classifier_before_domain_cnt


def get_alexnet():
    from torchvision.models import alexnet
    model = alexnet(pretrained=dann_config.BACKBONE_PRETRAINED)
    features, pooling, classifier = model.features, model.avgpool, model.classifier
    classifier[-2] = nn.ReLU()
    classifier[-1] = nn.Linear(4096, dann_config.CLASSES_CNT)
    classifier_layer_ids = [1, 4, 6]
    pooling_ftrs = 256
    pooling_output_side = 6
    return features, pooling, classifier, classifier_layer_ids, 4096, 2


def get_resnet50():
    from torchvision.models import resnet50
    model = resnet50(pretrained=dann_config.BACKBONE_PRETRAINED)
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    pooling = model.avgpool
    classifier = nn.Sequential(nn.Linear(2048, dann_config.CLASSES_CNT))
    classifier_layer_ids = [0]
    pooling_ftrs = 2048
    pooling_output_side = 1
    return features, pooling, classifier, classifier_layer_ids, 2048, 0


def get_resnet50_for_DANN_CA():
    from torchvision.models import resnet50
    model = resnet50(pretrained=dann_config.BACKBONE_PRETRAINED)
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    pooling = model.avgpool
    classifier = nn.Sequential(
        nn.Linear(2048, 256),
        # nn.Dropout2d(),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, dann_config.CLASSES_CNT + 1),
        )
    domain_input_len = 256
    classifier_layer_ids = [0, 2, 4]
    pooling_ftrs = 2048
    pooling_output_side = 1
    return features, pooling, classifier, classifier_layer_ids, domain_input_len, 2


def get_resnet50_for_DANN_CA_rich():
    from torchvision.models import resnet50
    model = resnet50(pretrained=dann_config.BACKBONE_PRETRAINED)
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    pooling = model.avgpool
    domain_input_len = 512
    if dann_config.LONG_CLS and dann_config.NEED_ADAPTATION_BLOCK_AV:
        domain_input_len = dann_config.BOTTLENECK_SIZE
        classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout2d(dann_config.DROPOUT),
            nn.ReLU(),
            nn.Linear(2048, domain_input_len),
            nn.BatchNorm1d(domain_input_len),
            nn.Dropout2d(0.05),
            nn.ReLU(),
            nn.Linear(domain_input_len, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout2d(dann_config.DROPOUT),
            nn.ReLU(),
            nn.Linear(2048, dann_config.CLASSES_CNT + 1),
        )
        classifier_layer_ids = [0, 4, 8, 12]

    elif dann_config.LONG_CLS:
        classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout2d(dann_config.DROPOUT),
            nn.ReLU(),
            nn.Linear(2048, domain_input_len * 2),
            nn.BatchNorm1d(domain_input_len * 2),
            nn.Dropout2d(dann_config.DROPOUT),
            nn.ReLU(),
            nn.Linear(domain_input_len * 2, domain_input_len),
            nn.BatchNorm1d(domain_input_len),
            nn.Dropout2d(dann_config.DROPOUT / 2),
            nn.ReLU(),
            nn.Linear(domain_input_len, dann_config.CLASSES_CNT + 1),
        )
        classifier_layer_ids = [0, 4, 8, 12]
    else:
        classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout2d(dann_config.DROPOUT),
            nn.ReLU(),
            nn.Linear(2048, domain_input_len),
            nn.BatchNorm1d(domain_input_len),
            nn.Dropout2d(dann_config.DROPOUT),
            nn.ReLU(),
            nn.Linear(domain_input_len, dann_config.CLASSES_CNT + 1),
        )
        classifier_layer_ids = [0, 4, 8]
    pooling_ftrs = 2048
    pooling_output_side = 1
    return features, pooling, classifier, classifier_layer_ids, domain_input_len, 2


def get_resnet50_rich_classifier():
    from torchvision.models import resnet50
    model = resnet50(pretrained=dann_config.BACKBONE_PRETRAINED)
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    pooling = model.avgpool
    if dann_config.NEED_ADAPTATION_BLOCK_AV:
        domain_input_len = dann_config.BOTTLENECK_SIZE
        classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Linear(2048, domain_input_len),
            nn.BatchNorm1d(domain_input_len),
            nn.ReLU(),
            nn.Linear(domain_input_len, 2048),
            nn.ReLU(),
            nn.Linear(2048, dann_config.CLASSES_CNT),
            )
        classifier_layer_ids = [0, 4, 7, 9]
    else:
        domain_input_len = 1024
        classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Linear(2048, domain_input_len),
            nn.BatchNorm1d(domain_input_len),
            nn.ReLU(),
            nn.Linear(domain_input_len, dann_config.CLASSES_CNT),
            )
        classifier_layer_ids = [0, 4, 7]
    pooling_ftrs = 2048
    pooling_output_side = 1
    return features, pooling, classifier, classifier_layer_ids, domain_input_len, 2


def get_vanilla_dann():
    hidden_size = 64
    pooling_output_side = 6

    features = nn.Sequential(
        nn.Conv2d(3, hidden_size, kernel_size=5),
        nn.BatchNorm2d(hidden_size),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(hidden_size, hidden_size, kernel_size=5),
        nn.BatchNorm2d(hidden_size),
        nn.Dropout2d(),
        nn.MaxPool2d(2),
        nn.ReLU(),
    )
    pooling = nn.AdaptiveAvgPool2d((pooling_output_side, pooling_output_side))
    classifier = nn.Sequential(
        nn.Linear(hidden_size * pooling_output_side * pooling_output_side, hidden_size * 2),
        nn.BatchNorm1d(hidden_size * 2),
        nn.Dropout2d(),
        nn.ReLU(),
        nn.Linear(hidden_size * 2, hidden_size * 2),
        nn.BatchNorm1d(hidden_size * 2),
        nn.ReLU(),
        nn.Linear(hidden_size * 2, dann_config.CLASSES_CNT),
    )
    classifier_layer_ids = [0, 4, 7]
    pooling_ftrs = hidden_size
    return features, pooling, classifier, classifier_layer_ids, hidden_size * 2, 2


def get_mnist_dann():
    pooling_output_side = 4
    
    features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),  # [b, 32, 24, 24]
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2, stride=2),  # [b, 32, 12, 12]
        nn.ReLU(),
        nn.Conv2d(32, 48, kernel_size=5),  # [b, 48, 8, 8]
        nn.BatchNorm2d(48),
        nn.Dropout2d(),
        nn.MaxPool2d(2, stride=2),  # [b, 48, 4, 4]
        nn.ReLU(),
    )
    pooling = nn.AdaptiveAvgPool2d((pooling_output_side, pooling_output_side))
    classifier = nn.Sequential(
        nn.Linear(48 * pooling_output_side * pooling_output_side, 100),
        nn.BatchNorm1d(100),
        nn.Dropout2d(),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, dann_config.CLASSES_CNT),
    )
    classifier_layer_ids = [0, 4, 7]
    return features, pooling, classifier, classifier_layer_ids, 48 * pooling_output_side * pooling_output_side, 0
