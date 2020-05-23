import torch
import torch.nn as nn

import configs.dann_config as dann_config
import models.backbone_models as backbone_models
import models.domain_heads as domain_heads
import models.blocks as blocks


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass


class DANNModel(BaseModel):
    def __init__(self):
        super(DANNModel, self).__init__()
        self.features, self.pooling, self.class_classifier, \
            domain_input_len, self.classifier_before_domain_cnt = backbone_models.get_backbone_model()

        if dann_config.NEED_ADAPTATION_BLOCK:
            self.adaptation_block = nn.Sequential(
                nn.ReLU(),
                nn.Linear(domain_input_len, 2048),
                nn.ReLU(inplace=True),
            )
            domain_input_len = 2048
            classifier_start_output_len = self.class_classifier[self.classifier_before_domain_cnt][-1].out_features
            self.class_classifier[self.classifier_before_domain_cnt][-1] = nn.Linear(2048, classifier_start_output_len)

        self.domain_classifier = domain_heads.get_domain_head(domain_input_len)

    def forward(self, input_data, rev_grad_alpha=dann_config.GRADIENT_REVERSAL_LAYER_ALPHA):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (map of tensors) - map with model output tensors
        """
        features = self.features(input_data)
        features = self.pooling(features)
        features = torch.flatten(features, 1)

        output_classifier = features
        classifier_layers_outputs = []
        for i in range(self.classifier_before_domain_cnt):
            output_classifier = self.class_classifier[i](output_classifier)
            classifier_layers_outputs.append(output_classifier)

        if dann_config.NEED_ADAPTATION_BLOCK:
            output_classifier = self.adaptation_block(output_classifier)

        reversed_features = blocks.GradientReversalLayer.apply(output_classifier, rev_grad_alpha)

        for i in range(self.classifier_before_domain_cnt, len(self.class_classifier)):
            output_classifier = self.class_classifier[i](output_classifier)
            classifier_layers_outputs.append(output_classifier)
        
        output_domain = self.domain_classifier(reversed_features)

        output = {
            "class": output_classifier,
            "domain": output_domain,
            "features": features
        }

        if dann_config.LOSS_NEED_INTERMEDIATE_LAYERS:
            output["classifier_layers"] = classifier_layers_outputs

        return output

    def predict(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        return self.forward(input_data)["class"]

    def get_features(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        output = self.forward(input_data)
        if not dann_config.LOSS_NEED_INTERMEDIATE_LAYERS and dann_config.FEATURES == 'before_class':
            raise RuntimeError('intermediate layers outputs are not saved')
        if dann_config.FEATURES == 'before_class':
            return output["classifier_layers"][-2]
        if not dann_config.LOSS_NEED_INTERMEDIATE_LAYERS and dann_config.FEATURES == 'before_bottleneck':
            raise RuntimeError('intermediate layers outputs are not saved')
        if dann_config.FEATURES == 'before_bottleneck':
            return output["classifier_layers"][-4]
        if dann_config.FEATURES == 'after_conv':
            return output["features"]
        raise RuntimeError(str(dann_config.FEATURES + ' is not implemented'))


class DANNCA_Model(BaseModel):
    def __init__(self):
        super(DANNCA_Model, self).__init__()
        self.features, self.pooling, self.class_classifier, \
        _, _ = backbone_models.get_backbone_model()

    def forward(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (map of tensors) - map with model output tensors
        """
        features = self.features(input_data)
        features = self.pooling(features)
        features = torch.flatten(features, 1)

        output_classifier = features
        classifier_layers_outputs = []
        for i in range(len(self.class_classifier)):
            output_classifier = self.class_classifier[i](output_classifier)
            classifier_layers_outputs.append(output_classifier)

        output = {
            "class": output_classifier,
            "features": features
        }

        if dann_config.LOSS_NEED_INTERMEDIATE_LAYERS:
            output["classifier_layers"] = classifier_layers_outputs

        return output

    def predict(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        return self.forward(input_data)["class"][:, :-1]

    def get_features(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        output = self.forward(input_data)
        if not dann_config.LOSS_NEED_INTERMEDIATE_LAYERS and dann_config.FEATURES == 'before_class':
            raise RuntimeError('intermediate layers outputs are not saved')
        if dann_config.FEATURES == 'before_class':
            return output["classifier_layers"][-2]
        if not dann_config.LOSS_NEED_INTERMEDIATE_LAYERS and dann_config.FEATURES == 'before_bottleneck':
            raise RuntimeError('intermediate layers outputs are not saved')
        if dann_config.FEATURES == 'before_bottleneck':
            return output["classifier_layers"][-4]
        if dann_config.FEATURES == 'after_conv':
            return output["features"]
        raise RuntimeError(str(dann_config.FEATURES + ' is not implemented'))


class OneDomainModel(BaseModel):
    def __init__(self):
        super(OneDomainModel, self).__init__()
        self.features, self.pooling, self.class_classifier, *_ = backbone_models.get_backbone_model()

    def forward(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (map of tensors) - map with model output tensors
        """
        features = self.features(input_data)
        features = self.pooling(features)
        features = torch.flatten(features, 1)
        
        output_classifier = features
        for block in self.class_classifier:
            output_classifier = block(output_classifier)

        output = {
            "class": output_classifier,
        }
        
        return output

    def predict(self, input_data):
        """
        Args:
            input_data (torch.tensor) - batch of input images
        Return:
            output (tensor) - model predictions

        Function for testing process when need to solve only
        target task.
        """
        return self.forward(input_data)["class"]     
