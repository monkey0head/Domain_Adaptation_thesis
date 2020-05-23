import torch
from trainer.logger import AvgLossLogger
import configs.dann_config as dann_config

class Trainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.epoch = 0
        self.device = next(self.model.parameters()).device
        self.loss_logger = AvgLossLogger()

    def calc_loss(self, src_batch, trg_batch):
        batch = self._merge_batches(src_batch, trg_batch)
        metadata = {'epoch': self.epoch, 'n_epochs': self.n_epochs}
        loss, loss_info = self.loss(self.model, batch, device=self.device, **metadata)
        return loss, loss_info
    
    def train_on_batch(self, src_batch, trg_batch, opt):
        self.model.train()
        loss, loss_info = self.calc_loss(src_batch, trg_batch)


        if dann_config.DANN_CA:
            classifier_loss, feature_loss = loss
            self.loss_logger.store(loss=classifier_loss.data.cpu().item() + feature_loss.data.cpu().item(), **loss_info)
            opt.zero_grad()
            classifier_loss.backward(retain_graph=True)
            temp_grad = []
            for param in self.model.parameters():
                if param.requires_grad:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append(None)
            grad_for_classifier = temp_grad

            opt.zero_grad()
            feature_loss.backward()
            temp_grad = []
            for param in self.model.parameters():
                if param.requires_grad:
                    temp_grad.append(param.grad.data.clone())
                else:
                    temp_grad.append(None)
            grad_for_feature_extractor = temp_grad

            # update parameters
            count = 0
            for param in self.model.parameters():
                # print(count, param.shape)
                if param.requires_grad:
                    temp_grad = param.grad.data.clone()
                    temp_grad.zero_()
                    if dann_config.ALTERNATING_UPDATE:
                        if count < dann_config.FEATURES_END and self.epoch % 2 == 0:
                            temp_grad = grad_for_feature_extractor[count]
                        elif count >= dann_config.FEATURES_END and self.epoch % 2 == 1:
                            temp_grad = grad_for_classifier[count]
                    else:
                        if count < dann_config.FEATURES_END:
                            temp_grad = grad_for_feature_extractor[count]
                        elif count >= dann_config.FEATURES_END:
                            temp_grad = grad_for_classifier[count]
                    param.grad.data = temp_grad
                count = count + 1

        else:
            self.loss_logger.store(loss=loss.data.cpu().item(), **loss_info)
            opt.zero_grad()
            loss.backward()

        opt.step()
        
    def _merge_batches(self, src_batch, trg_batch):
        src_images, src_classes = src_batch
        trg_images, trg_classes = trg_batch
        batch = dict()
        batch['src_images'] = src_images
        batch['trg_images'] = trg_images
        batch['src_classes'] = src_classes
        trg_classes = torch.zeros_like(trg_classes)
        batch['trg_classes'] = trg_classes
        return batch

    def fit(self, src_data, trg_data, n_epochs=1000, steps_per_epoch=100, val_freq=1,
            opt='adam', opt_kwargs=None, validation_data=None, metrics=None,
            lr_scheduler=None, callbacks=None):

        self.n_epochs = n_epochs

        if opt_kwargs is None:
            opt_kwargs = dict()

        if opt == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), **opt_kwargs)
        elif opt == 'sgd':
            parameters = self.model.parameters()
            if dann_config.NEED_ADAPTATION_BLOCK and hasattr(self.model, "adaptation_block"):
                parameters = [{ "params": self.model.features.parameters(), "lr": 0.1 * opt_kwargs["lr"] },
                              { "params": self.model.class_classifier[:-1].parameters(), "lr": 0.1 * opt_kwargs["lr"] },
                              { "params": self.model.class_classifier[-1].parameters() },
                              { "params": self.model.domain_classifier.parameters() },
                              { "params": self.model.adaptation_block.parameters() },
                ]
            opt = torch.optim.SGD(parameters, **opt_kwargs)
        else:
            raise NotImplementedError

        if validation_data is not None:
            src_val_data, trg_val_data = validation_data

        for self.epoch in range(self.epoch, n_epochs):
            self.loss_logger.reset_history()
            for step, (src_batch, trg_batch) in enumerate(zip(src_data, trg_data)):
                if step == steps_per_epoch:
                    break
                self.train_on_batch(src_batch, trg_batch, opt)
            
            # validation
            src_metrics = None
            trg_metrics = None
            if self.epoch % val_freq == 0 and validation_data is not None:
                self.model.eval()

                # calculating metrics on validation
                if metrics is not None:
                    if src_val_data is not None:
                        src_metrics = self.score(src_val_data, metrics)
                    if trg_val_data is not None:
                        trg_metrics = self.score(trg_val_data, metrics)
                
                # calculating loss on validation
                if src_val_data is not None and trg_val_data is not None:
                    for val_step, (src_batch, trg_batch) in enumerate(zip(src_val_data, trg_val_data)):
                        loss, loss_info = self.calc_loss(src_batch, trg_batch)
                        if dann_config.DANN_CA:
                            classifier_loss, feature_loss = loss
                            self.loss_logger.store(prefix="val",
                                loss=classifier_loss.data.cpu().item() + feature_loss.data.cpu().item(), **loss_info)
                        else:
                            self.loss_logger.store(prefix="val", loss=loss.data.cpu().item(), **loss_info)

            if callbacks is not None:
                epoch_log = dict(**self.loss_logger.get_info())
                if src_metrics is not None:
                    epoch_log['src_metrics'] = src_metrics
                if trg_metrics is not None:
                    epoch_log['trg_metrics'] = trg_metrics
                for callback in callbacks:
                    callback(self.model, epoch_log, self.epoch, n_epochs)

            if lr_scheduler:
                lr_scheduler.step(opt, self.epoch, n_epochs)

    def score(self, data, metrics):
        for metric in metrics:
            metric.reset()

        data.reload_iterator()
        for images, true_classes in data:
            pred_classes = self.model.predict(images)
            for metric in metrics:
                metric(true_classes, pred_classes)
        data.reload_iterator()
        return {metric.name: metric.score for metric in metrics}

    def predict(self, data):
        predictions = []
        for batch in data:
            predictions.append(self.model.predict(batch))
        return torch.cat(predictions)
