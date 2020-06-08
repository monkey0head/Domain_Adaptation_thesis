"""
Code to train the model 3 times and plot results localy and to wandb. Config the process in configs/dann_config
"""
import torch
import os
import wandb

from trainer import Trainer
from loss import loss_DANN, class_prediction_loss, loss_DANNCA, loss_DADA
from models import DANNModel, OneDomainModel, DANNCA_Model, DADA_Model
from dataloader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback, print_callback, ModelSaver, HistorySaver, WandbCallback
from utils.schedulers import LRSchedulerSGD, DANNCASchedulerSGD
import configs.dann_config as dann_config

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not torch.cuda.is_available():
    raise RuntimeError()

if __name__ == '__main__':
    print('source_domain is {}, target_domain is {}'.format(dann_config.SOURCE_DOMAIN, dann_config.TARGET_DOMAIN))

    train_gen_s, val_gen_s, _ = create_data_generators(dann_config.DATASET,
                                                       dann_config.SOURCE_DOMAIN,
                                                       batch_size=dann_config.BATCH_SIZE,
                                                       infinite_train=True,
                                                       image_size=dann_config.IMAGE_SIZE,
                                                       split_ratios=[0.95, 0.5, 0],
                                                       num_workers=dann_config.NUM_WORKERS,
                                                       device=device,
                                                       random_seed=None)

    train_gen_t, _, _ = create_data_generators(dann_config.DATASET,
                                               dann_config.TARGET_DOMAIN,
                                               batch_size=dann_config.BATCH_SIZE,
                                               infinite_train=True,
                                               split_ratios=[1, 0, 0],
                                               image_size=dann_config.IMAGE_SIZE,
                                               num_workers=dann_config.NUM_WORKERS,
                                               device=device,
                                               random_seed=None)

    val_gen_t, _, _ = create_data_generators(dann_config.DATASET,
                                             dann_config.TARGET_DOMAIN,
                                             batch_size=dann_config.BATCH_SIZE,
                                             infinite_train=False,
                                             split_ratios=[0.1, 0, 0],
                                             image_size=dann_config.IMAGE_SIZE,
                                             num_workers=dann_config.NUM_WORKERS,
                                             device=device,
                                             random_seed=None)

    experiment_name = 'DANN_rich_129_a_w_entropy_1'
    details_name = ''

    for i in range(3):
        # select model type: OneDomainModel, DANNModel, DANNCA_Model, DADA_Model
        model = DANNModel().to(device)
        acc = AccuracyScoreFromLogits()
        # select custom lr scheduler from utils/schedulers
        scheduler = LRSchedulerSGD()
        # select loss function for selected model from class_prediction_loss, loss_DANN, loss_DANNCA, loss_DADA
        tr = Trainer(model, loss_DANN)

        print(experiment_name, details_name)
        tr.fit(train_gen_s, train_gen_t,
               n_epochs=dann_config.N_EPOCHS,
               validation_data=[val_gen_s, val_gen_t],
               metrics=[acc],
               steps_per_epoch=dann_config.STEPS_PER_EPOCH,
               val_freq=dann_config.VAL_FREQ,
               opt='sgd',
               opt_kwargs={'lr': dann_config.LR, 'momentum': 0.9},
               lr_scheduler=scheduler,
               callbacks=[
                   print_callback(watch=["loss", "domain_loss", "val_loss",
                                         "val_domain_loss", 'trg_metrics', 'src_metrics']),

                   ModelSaver(str(experiment_name + '_' + dann_config.SOURCE_DOMAIN + '_' +
                                  dann_config.TARGET_DOMAIN + '_' + details_name),
                              dann_config.SAVE_MODEL_FREQ,
                              save_by_schedule=True,
                              save_best=False,
                              eval_metric='accuracy'),
                   #WandbCallback(config=dann_config,
                    #             name=str(dann_config.SOURCE_DOMAIN + "_" + dann_config.TARGET_DOMAIN +
                    #                      "_" + details_name),
                    #             group=experiment_name),
                   HistorySaver(str(experiment_name + '_' + dann_config.SOURCE_DOMAIN + '_' +
                                    dann_config.TARGET_DOMAIN + "_" + details_name),
                                dann_config.VAL_FREQ, path=str('_log/' + experiment_name + "_" + details_name),
                                )
               ])
