import numpy as np
import torch

from torchvision.datasets import ImageFolder, MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import configs.dann_config as dann_config

from dataloader.mnistm_dataset import MNISTM


class Office31Dataset(ImageFolder):
    """
        Office31 Dataset class.
        More info about the dataset: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/
        Data link: https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
    """
    def __init__(self, root, transform, image_size, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        super().__init__(root, transform=transform, **kwargs)

    def get_semi_supervised_indexes_for_subset(self, labeled_ratio, subset_indices):
        subset_images_with_classes = np.array(self.imgs)[subset_indices]
        class_name_to_id = self.class_to_idx
        data_classes = [int(x[1]) for x in subset_images_with_classes]
        unique_classes, classes_counts = np.unique(data_classes, return_counts=True)
        labeled_indexes = []
        unlabeled_indexes = []
        last_included = False

        for class_id in unique_classes:
            all_class_indexes = np.where(np.array(data_classes) == class_id)[0]
            labeled_num = labeled_ratio*classes_counts[class_id]
            if (labeled_num % 1 > 0):
                labeled_num = int(labeled_num + last_included)
                last_included = not last_included
            else:
                labeled_num = int(labeled_num)
            labeled_indexes.extend(all_class_indexes[:labeled_num])
            unlabeled_indexes.extend(all_class_indexes[labeled_num:])

        return labeled_indexes, unlabeled_indexes
    
    
class MnistDataset(MNIST):
    """
        MNIST Dataset class. Inherits from `torchvision.datasets.MNIST`.
    """
    def __init__(self, root, transform, image_size, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        super().__init__(root, transform=transform, **kwargs)
    
    def get_semi_supervised_indexes_for_subset(self):
        raise NotImplementedError
        

class MnistMDataset(MNISTM):
    """
        MNIST-M Dataset class.
    """
    def __init__(self, root, mnist_root, transform, image_size, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        super().__init__(root, mnist_root, transform=transform, **kwargs)
    
    def get_semi_supervised_indexes_for_subset(self):
        raise NotImplementedError


class DataGenerator(DataLoader):
    def __init__(self, is_infinite=False, device=torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.is_infinite = is_infinite
        self.reload_iterator()

    def reload_iterator(self):
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            if self.is_infinite:
                self.reload_iterator()
            batch = next(self.dataset_iterator)
        batch = [elem.to(self.device) for elem in batch]
        return batch

    def get_classes_to_idx(self):
        return self.dataset.dataset.class_to_idx

    def get_classes(self):
        return self.dataset.dataset.classes
    
    
class SemiSupervisedDataGenerator:
    def __init__(self, dataset, is_infinite, labeled_ratio, batch_size,
                 unk_value=dann_config.UNK_VALUE, *args, **kwargs):
        assert labeled_ratio is not None, "labeled ratio argument should be provided for semi-supervised dataset"
        # need dataset.dataset as dataset - dataset subset created because of splitting to train, val, test
        labeled_indexes, unlabeled_indexes = dataset.dataset.get_semi_supervised_indexes_for_subset(labeled_ratio,
                                                                                                    dataset.indices)
        self.batch_size = batch_size
        self.labeled_batch_size = int(labeled_ratio*batch_size)
        self.unlabeled_batch_size = self.batch_size - self.labeled_batch_size
        self.labeled_generator = DataGenerator(is_infinite, batch_size=self.labeled_batch_size,
                                               dataset=torch.utils.data.Subset(dataset, labeled_indexes),
                                               *args, **kwargs)
        self.unlabeled_generator = DataGenerator(is_infinite, batch_size=self.unlabeled_batch_size,
                                                 dataset=torch.utils.data.Subset(dataset, unlabeled_indexes),
                                                 *args, **kwargs)
        self.unk_class = unk_value

    def __next__(self):
        labeled_batch = next(self.labeled_generator)
        unlabeled_batch = next(self.unlabeled_generator)
        return (torch.cat([labeled_batch[0], unlabeled_batch[0]]), torch.cat([labeled_batch[1],
                                                                             -1*torch.ones_like(unlabeled_batch[1])]))

    def __iter__(self):
        return self

    def __len__(self):
        return min(len(self.unlabeled_generator), len(self.labeled_generator))

    def get_classes_to_idx(self):
        labeled_classes_to_idx = self.labeled_generator.get_classes_to_idx()
        labeled_classes_to_idx[self.unk_class] = -1
        return labeled_classes_to_idx

    def get_classes(self):
        return self.labeled_generator.dataset.dataset.classes


def create_data_generators(dataset_name, domain, data_path="data", batch_size=16,
                           transform=None, num_workers=1, split_ratios=[0.8, 0.1, 0.1],
                           image_size=500, infinite_train=False, device=torch.device('cpu'),
                           semi_supervised=False, semi_supervised_labeled_ratio=None,
                           random_seed=42):
    """
    Args:
        dataset_name (string)
        domain (string)
            - valid domain of the dataset dataset_name
        data_path (string)
            - valid path, which contains dataset_name folder
        batch_size (int)
        transform (callable)
            - optional transform applied on image sample
        num_workers (int)
            - multi-process data loading
        split_ratios (list of ints, len(split_ratios) = 3)
            - ratios of train, validation and test parts,
              in case of MNIST dataset two first values are scaled and used, 
              as test dataset is fixed

    Return:
        3 data generators  - for train, validation and test data
    """
        
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    train_dataset, val_dataset, test_dataset = create_dataset(dataset_name, domain, data_path, split_ratios, 
                                                              transform, image_size, device)
    
    if semi_supervised:
        train_dataloader = SemiSupervisedDataGenerator(is_infinite=infinite_train,
                                                       labeled_ratio=semi_supervised_labeled_ratio,
                                                       device=device, dataset=train_dataset,
                                                       batch_size=batch_size, shuffle=True,
                                                       num_workers=num_workers, drop_last=True)
    else:
        train_dataloader = DataGenerator(is_infinite=infinite_train, device=device, dataset=train_dataset,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataGenerator(is_infinite=False, device=device, dataset=val_dataset,
                                   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataGenerator(is_infinite=False, device=device, dataset=test_dataset,
                                    batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def create_dataset(dataset_name, domain, data_path, split_ratios, transform, image_size, device):
    """
    Args:
        dataset_name (string)
        domain (string)
            - valid domain of the dataset dataset_name
        data_path (string)
            - valid path, which contains dataset_name folder
        transformations (callable)
            - optional transform to be applied on an image sample

    Return:
        torchvision.dataset object
    """

    assert dataset_name in ["office-31", "mnist"], f"Dataset {dataset_name} is not implemented"

    if dataset_name == "office-31":
        dataset_domains = ["amazon", "dslr", "webcam"]

        assert domain in dataset_domains, f"Incorrect domain {domain}: " + \
            f"dataset {dataset_name} domains: {dataset_domains}"

        dataset = Office31Dataset(f"{data_path}/{dataset_name}/{domain}/images", transform=transform, image_size=image_size)
        
        len_dataset = len(dataset)
        train_size = int(len_dataset * split_ratios[0])
        val_size = int(len_dataset * split_ratios[1])
        test_size = len_dataset - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    elif dataset_name == "mnist":
        
        dataset_domains = ["mnist", "mnist-m"]
        
        assert domain in dataset_domains, f"Incorrect domain {domain}: " + \
            f"dataset {dataset_name} domains: {dataset_domains}"
        
        if domain == "mnist":
            train_dataset = MnistDataset(f"{data_path}/{dataset_name}/{domain}",
                                         train=True, transform=transform, image_size=image_size, download=True)
            test_dataset = MnistDataset(f"{data_path}/{dataset_name}/{domain}",
                                        train=False, transform=transform, image_size=image_size, download=True)
        elif domain == "mnist-m":
            train_dataset = MnistMDataset(f"{data_path}/{dataset_name}/{domain}",
                                          f"{data_path}/{dataset_name}/mnist",
                                          train=True, transform=transform, image_size=image_size, download=True)
            test_dataset = MnistMDataset(f"{data_path}/{dataset_name}/{domain}",
                                         f"{data_path}/{dataset_name}/mnist",
                                         train=False, transform=transform, image_size=image_size, download=True)
        
        split_ratios = [split_ratios[0] / sum(split_ratios[0:2]),
                        split_ratios[1] / sum(split_ratios[0:2])]  # test_size is fixed in MNIST
        
        len_dataset = len(train_dataset)
        train_size = int(len_dataset * split_ratios[0])
        val_size = len_dataset - train_size
        
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset
