import os
import torch
import argparse
import numpy as np

from models import DANNModel, DANNCA_Model
from dataloader import create_data_generators_my
from metrics import AccuracyScoreFromLogits
import configs.dann_config as dann_config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_classes_features(model, data):
    features = []
    classes = []
    for images, true_classes in data:
        features.append(model.get_features(images).data.cpu())
        classes.append(true_classes)
    return torch.cat(features).data.cpu().numpy(), torch.cat(classes).data.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    args = parser.parse_args()

    for domain, name in [('amazon', 'A'), ('webcam', 'W'), ('dslr', 'D')]:
        dann_config.TARGET_DOMAIN = domain
        gen_t, _, _ = create_data_generators_my(dann_config.DATASET,
                                                   dann_config.TARGET_DOMAIN,
                                                   batch_size=dann_config.BATCH_SIZE,
                                                   infinite_train=False,
                                                   split_ratios=[1, 0, 0],
                                                   image_size=dann_config.IMAGE_SIZE,
                                                   num_workers=dann_config.NUM_WORKERS,
                                                   device=device)

        model = DANNCA_Model().to(device)
        # model = DANNModel().to(device)
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()

        features, classes = get_classes_features(model, gen_t)
        classes = classes.astype('int')
        features /= np.linalg.norm(features,  axis=-1, keepdims=True)
        embeddings_name = 'dann-ca_141_after_conv'

        path = './embeddings/' + embeddings_name
        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt(str(path + '/' + name +'f.txt'), features, delimiter=',', fmt='%.7f')

        with open(str(path + '/' + name +'l.txt'), 'w') as f:
            for idx in range(len(classes) - 1):
                f.write(str(classes[idx].item()))
                f.write(',')
            f.write(str(classes[-1].item()))
            f.write('\n')
