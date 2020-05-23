import wandb
import os
import configs.dann_config as dann_config


def simple_callback(model, epoch_log, current_epoch, total_epoch):
    train_loss = epoch_log['loss']
    val_loss = epoch_log['val_loss']
    trg_metrics = epoch_log['trg_metrics']
    src_metrics = epoch_log['src_metrics']
    message_head = f'Epoch {current_epoch+1}/{total_epoch}\n'
    message_loss = 'loss: {:<10}\t val_loss: {:<10}\t'.format(train_loss, val_loss)
    message_src_metrics = ' '.join(['val_src_{}: {:<10}\t'.format(k, v) for k, v in src_metrics.items()])
    message_trg_metrics = ' '.join(['val_trg_{}: {:<10}\t'.format(k, v) for k, v in trg_metrics.items()])
    print(message_head + message_loss + message_src_metrics + message_trg_metrics)


class print_callback:
    def __init__(self, watch=None):
        """
        Callback which prints everything from log (by default)
        or items specified by watch list
        """
        if watch is not None:
            self.watch = set(watch)
        else:
            self.watch = None
        
    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        print('Epoch {}/{}'.format(current_epoch+1, total_epoch))
        for key, value in epoch_log.items():
            if (self.watch is None) or (key in self.watch):
                if not key.endswith('metrics'):
                    print('{}: {:.5f}'.format(key, value))
                elif key == 'trg_metrics':
                    print(' '.join(['val_trg_{}: {:<10}\t'.format(k, v) for k, v in epoch_log['trg_metrics'].items()]))
                elif key == 'src_metrics':
                    print(' '.join(['val_src_{}: {:<10}\t'.format(k, v) for k, v in epoch_log['src_metrics'].items()]))
        print()


class ModelSaver:
    def __init__(self, model_type, save_freq=1, path="checkpoints", save_by_schedule=True,
                 save_best=False, eval_metric=None):
        self.model_type = model_type
        self.path = path
        self.save_by_schedule = save_by_schedule
        self.save_freq = save_freq
        self.save_best = save_best
        self.eval_metric = eval_metric
        self.best_metric = 0
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, model_type)):
            os.makedirs(os.path.join(path, model_type))

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        import torch
        if self.save_best and self.eval_metric is not None:
            if current_epoch == 0:
                self.best_metric = epoch_log['trg_metrics'][self.eval_metric]
            if epoch_log['trg_metrics'][self.eval_metric] > self.best_metric:
                filename = os.path.join(self.path, self.model_type, "best_metric_on_trg.pt")
                torch.save(model.state_dict(), filename)
                self.best_metric = epoch_log['trg_metrics'][self.eval_metric]

        if self.save_by_schedule and current_epoch != 0 and current_epoch % self.save_freq == 0:
            filename = os.path.join(self.path, self.model_type, "epoch_{}.pt".format(current_epoch))
            torch.save(model.state_dict(), filename)


class HistorySaver:
    import json
    from collections import defaultdict
    json = json
    defaultdict = defaultdict

    def __init__(self, log_name, val_freq=1, path="_log", plot=True, extra_losses=None):
        self.is_plotting = plot
        self.val_freq = val_freq
        self.loss_history = self.defaultdict(list)
        self.src_metrics_history = self.defaultdict(list)
        self.trg_metrics_history = self.defaultdict(list)
        
        self.extra_losses = extra_losses
        if extra_losses is not None:
            self.extra_losses_history = self.defaultdict(list)

        if plot:
            import matplotlib.pyplot as plt
            self.plt = plt
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, log_name)):
            os.makedirs(os.path.join(path, log_name))

        self.path = os.path.join(path, log_name)

    def _plot(self, data, name, current_epoch, total_epoch):
        self.plt.figure(figsize=(6, 4))

        for key in data:
            if key != 'loss':
                self.plt.plot(list(range(0, current_epoch + 1, self.val_freq)), data[key], label=key)
            else:
                self.plt.plot(data[key], label=key)

        self.plt.grid()
        self.plt.legend()
        self.plt.title('{} history for {} epochs of {}'.format(name, current_epoch + 1, total_epoch))
        self.plt.savefig(os.path.join(self.path, name + '_plot'))

    def plot_all(self, current_epoch, total_epoch):
        self._plot(self.loss_history, 'loss', current_epoch, total_epoch)
        self._plot(self.src_metrics_history, 'src_metrics', current_epoch, total_epoch)
        self._plot(self.trg_metrics_history, 'trg_metrics', current_epoch, total_epoch)
        
        if self.extra_losses is not None:
            for pic_name, loss_names in self.extra_losses.items():
                self._plot({k: v for k, v in self.extra_losses_history.items() if k in loss_names},
                           pic_name, current_epoch, total_epoch)
        self.plt.close('all')

    def _save_to_json(self, data, name=None):
        filename = os.path.join(self.path, name)
        with open(filename, 'w') as f:
            self.json.dump(data, f)

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        if current_epoch % self.val_freq == 0:
            self.loss_history['val_loss'].append(epoch_log['val_loss'])

            for metric in epoch_log['trg_metrics']:
                self.trg_metrics_history[metric].append(epoch_log['trg_metrics'][metric])

            for metric in epoch_log['src_metrics']:
                self.src_metrics_history[metric].append(epoch_log['src_metrics'][metric])
            
            if self.extra_losses is not None:
                for loss_names in self.extra_losses.values():
                    for loss_name in loss_names:
                        self.extra_losses_history[loss_name].append(epoch_log[loss_name])

        self.loss_history['loss'].append(epoch_log['loss'])

        self._save_to_json(self.loss_history, name='loss_history')
        self._save_to_json(self.src_metrics_history, name='src_metrics')
        self._save_to_json(self.trg_metrics_history, name='trg_metrics')
        
        if self.extra_losses is not None:
            self._save_to_json(self.extra_losses_history, name='extra_losses')

        if self.is_plotting:
            self.plot_all(current_epoch, total_epoch)


def dict_from_module(module):
    context = {}
    for setting in dir(module):
        if not setting.startswith("__"):
            context[setting] = getattr(module, setting)

    return context


class WandbCallback:
    def __init__(self, *args, config=dann_config, **kwargs):
        """
        Callback that logs everything to wandb
        """
        wandb.init(*args, **kwargs, project="domain_adaptation", reinit=True)
        # wandb.init(*args, **kwargs, project="DomainAdaptation", entity='arqwer', reinit=True)
        wandb.config.update(dict_from_module(dann_config))

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        logged = {
            'epoch': current_epoch
        }
        logged.update(epoch_log)
        wandb.log(logged)
