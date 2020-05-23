import argparse
import torch
import os

from trainer import Trainer
from models import DANNModel, OneDomainModel, DANNCA_Model
from dataloader import create_data_generators
from metrics import AccuracyScoreFromLogits
import configs.dann_config as dann_config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_gen_t, _, _ = create_data_generators(dann_config.DATASET,
                                              dann_config.TARGET_DOMAIN,
                                              batch_size=dann_config.BATCH_SIZE,
                                              infinite_train=False,
                                              image_size=dann_config.IMAGE_SIZE,
                                              split_ratios=[1, 0, 0],
                                              num_workers=dann_config.NUM_WORKERS,
                                              device=device)
    # model = DANNModel().to(device)
    model = DANNCA_Model().to(device)
    # model = OneDomainModel().to(device)
    # print(model)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    acc = AccuracyScoreFromLogits()
    tr = Trainer(model, None)
    scores = tr.score(test_gen_t, [acc])
    scores_string = '   '.join(['{}: {:.5f}\t'.format(k, float(v)) for k, v in scores.items()])
    print(f"scores on dataset \"{dann_config.DATASET}\", domain \"{dann_config.TARGET_DOMAIN}\":")
    print(scores_string)
