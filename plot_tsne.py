import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import configs.dann_config as dann_config

N_CLASSES = 10

def get_features(domain, path):
    labels = np.loadtxt(str(path + '/' + domain + 'l.txt'), delimiter=',')
    features = np.loadtxt(str(path + '/' + domain + 'f.txt'), delimiter=',')
    return labels[labels < N_CLASSES].astype('int'), features[labels < N_CLASSES, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model embeddings in embeddings folder')
    parser.add_argument('--name', type=str, required=False, help='plot name')
    args = parser.parse_args()

    colors = ['darkcyan', 'royalblue', 'yellowgreen', 'indianred', 'plum', 'orange', 'gray', 'deepskyblue', 'navy', 'purple']
    path = str('./embeddings/' + args.checkpoint)

    # plt.scatter(np.arange(10), np.arange(10),
    #             color=colors,
    #             marker='*',
    #             alpha=0.7,
    #             label='amazon')

    plt.figure(figsize=(10, 10))
    labels_a, features_a = get_features('A', path)
    labels_w, features_w = get_features('W', path)
    pca = PCA(128)
    features_pca = pca.fit_transform(np.vstack([features_a, features_w]))
    tsne = TSNE(verbose=10, n_jobs=4)
    tsne_output = tsne.fit_transform(features_pca)
    features_a_tsne = tsne_output[: features_a.shape[0]]
    features_w_tsne = tsne_output[features_a.shape[0]:]
    plt.scatter(features_a_tsne[:, 0], features_a_tsne[:, 1],
                color=[colors[idx] for idx in labels_a],
                marker='*',
                alpha=0.7,
                label='amazon')
    plt.scatter(features_w_tsne[:, 0], features_w_tsne[:, 1],
                color=[colors[idx] for idx in labels_w],
                marker='H',
                alpha=1,
                label='webcam')
    plt.legend()
    plt.title('TSNE plot')
    name = args.name if args.name else args.checkpoint
    plt.savefig(str('TSNE_plot_' + name))
    plt.show()