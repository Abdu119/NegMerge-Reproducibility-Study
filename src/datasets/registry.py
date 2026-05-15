import sys
import inspect
import random
import torch
import copy
import os
import warnings

from torch.utils.data.dataset import random_split

# Use HuggingFace-based Cars loader (original Stanford URLs are dead)
try:
    from src.datasets.cars_hf import Cars
except ImportError:
    from src.datasets.cars import Cars  # Fallback to original if HF not available

from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397

# ImageNet with fallback handling
def get_imagenet_class():
    """Get the appropriate ImageNet class based on availability."""
    # Check if USE_DUMMY_IMAGENET is set (for pipeline testing)
    if os.environ.get('USE_DUMMY_IMAGENET', '').lower() in ('1', 'true', 'yes'):
        from src.datasets.imagenet_hf import ImageNetDummy
        warnings.warn("Using DUMMY ImageNet - results will NOT be meaningful!")
        return ImageNetDummy
    
    # Try original ImageNet (requires local download)
    try:
        from src.datasets.imagenet import ImageNet as OriginalImageNet
        # We'll let it fail at runtime if files don't exist
        return OriginalImageNet
    except ImportError:
        pass
    
    # Try HuggingFace ImageNet
    try:
        from src.datasets.imagenet_hf import ImageNetValOnly
        return ImageNetValOnly
    except ImportError:
        pass
    
    # Return original and let it fail with a helpful error
    from src.datasets.imagenet import ImageNet
    return ImageNet

# Lazy ImageNet class that checks availability at instantiation time
class ImageNet:
    """
    ImageNet wrapper that tries multiple loading strategies.
    """
    _actual_class = None
    
    def __new__(cls, preprocess, location=None, batch_size=128, num_workers=4):
        # Check for local ImageNet first
        local_train = os.path.join(location or "dataset", "imagenet", "train")
        local_val = os.path.join(location or "dataset", "imagenet", "val")
        
        if os.path.exists(local_train) and os.path.exists(local_val):
            # Use original ImageNet loader
            from src.datasets.imagenet import ImageNet as OriginalImageNet
            return OriginalImageNet(preprocess, location, batch_size, num_workers)
        
        # Check for val-only local
        if os.path.exists(local_val):
            print("Found ImageNet validation set only, using val-only mode")
            from src.datasets.imagenet_hf import ImageNetValOnly
            return ImageNetValOnly(preprocess, location, batch_size, num_workers)
        
        # Check if user wants dummy data
        if os.environ.get('USE_DUMMY_IMAGENET', '').lower() in ('1', 'true', 'yes'):
            from src.datasets.imagenet_hf import ImageNetDummy
            return ImageNetDummy(preprocess, location, batch_size, num_workers)
        
        # Try HuggingFace
        try:
            from src.datasets.imagenet_hf import ImageNetValOnly
            return ImageNetValOnly(preprocess, location, batch_size, num_workers)
        except Exception as e:
            # Provide helpful error message
            raise FileNotFoundError(
                f"ImageNet dataset not found at {local_train}\n\n"
                "Options to resolve this:\n"
                "1. Download ImageNet to dataset/imagenet/ (train/ and val/ subdirs)\n"
                "2. Accept HuggingFace terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
                "   Then set HF_TOKEN environment variable\n"
                "3. For pipeline testing only, set USE_DUMMY_IMAGENET=1 (results won't be valid)\n"
                "4. Skip ImageNet evaluation by modifying eval_datasets in the notebook\n"
                f"\nOriginal error: {e}"
            )


registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}

# Ensure ImageNet is in registry (it's defined in this module now)
registry['ImageNet'] = ImageNet


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=16, val_fraction=0.1, max_val_samples=5000):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    return dataset
