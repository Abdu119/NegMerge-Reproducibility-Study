# ImageNet dataset loader using HuggingFace
# This provides a lighter-weight alternative to the full ImageNet download
# NOTE: Requires accepting the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
# And setting HF_TOKEN environment variable

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings

class ImageNetHF:
    """
    ImageNet dataset loaded from HuggingFace.
    
    This requires:
    1. A HuggingFace account
    2. Accepting the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    3. Setting HF_TOKEN environment variable or logging in via `huggingface-cli login`
    
    For reproducibility studies where full ImageNet isn't available,
    consider using the validation set only or a proxy dataset.
    """
    
    def __init__(
        self,
        preprocess,
        location=None,  # Not used, kept for API compatibility
        batch_size=128,
        num_workers=4,
    ):
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Try to load from HuggingFace
        try:
            from datasets import load_dataset
            print("Loading ImageNet from Hugging Face...")
            print("NOTE: This requires accepting terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            
            # Load only validation split to save space/time
            self.hf_dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split="validation",
                trust_remote_code=True
            )
            
            self.train_dataset = ImageNetHFDataset(self.hf_dataset, self.preprocess, is_train=True)
            self.test_dataset = ImageNetHFDataset(self.hf_dataset, self.preprocess, is_train=False)
            
            # For validation, we use the same data (ImageNet val set)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            print(f"Loaded {len(self.hf_dataset)} ImageNet validation images")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ImageNet from HuggingFace: {e}\n"
                "Please ensure you have:\n"
                "1. A HuggingFace account\n"
                "2. Accepted the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
                "3. Set HF_TOKEN environment variable or run `huggingface-cli login`"
            )
        
        # ImageNet class names (subset for reference)
        self.classnames = [f"class_{i}" for i in range(1000)]  # Placeholder
    
    def name(self):
        return "imagenet"


class ImageNetHFDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace ImageNet."""
    
    def __init__(self, hf_dataset, preprocess, is_train=False):
        self.hf_dataset = hf_dataset
        self.preprocess = preprocess
        self.is_train = is_train
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing
        if self.preprocess is not None:
            image = self.preprocess(image)
        
        return image, label


class ImageNetValOnly:
    """
    Minimal ImageNet validation set loader.
    Uses only the validation split for both train and test to save space.
    Suitable for evaluation-only scenarios in reproducibility studies.
    """
    
    def __init__(
        self,
        preprocess,
        location=None,
        batch_size=128,
        num_workers=4,
    ):
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Check if local ImageNet exists first
        local_val_path = os.path.join(location or "dataset", "imagenet", "val") if location else None
        
        if local_val_path and os.path.exists(local_val_path):
            print(f"Loading ImageNet validation from local path: {local_val_path}")
            from torchvision.datasets import ImageFolder
            val_dataset = ImageFolder(local_val_path, transform=preprocess)
            self.train_dataset = val_dataset
            self.test_dataset = val_dataset
        else:
            # Fall back to HuggingFace
            print("Local ImageNet not found, attempting HuggingFace download...")
            self._load_from_hf()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.classnames = [f"class_{i}" for i in range(1000)]
    
    def _load_from_hf(self):
        try:
            from datasets import load_dataset
            print("Loading ImageNet validation from Hugging Face...")
            hf_dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split="validation",
                trust_remote_code=True
            )
            self.train_dataset = ImageNetHFDataset(hf_dataset, self.preprocess, is_train=True)
            self.test_dataset = ImageNetHFDataset(hf_dataset, self.preprocess, is_train=False)
            print(f"Loaded {len(hf_dataset)} ImageNet validation images")
        except Exception as e:
            raise RuntimeError(
                f"ImageNet not available locally or via HuggingFace: {e}\n"
                "Options:\n"
                "1. Download ImageNet manually to dataset/imagenet/\n"
                "2. Accept HuggingFace terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
                "3. Skip ImageNet evaluation (modify eval_datasets in notebook)"
            )
    
    def name(self):
        return "imagenet"


# Dummy/Placeholder ImageNet for when dataset isn't available
class ImageNetDummy:
    """
    Dummy ImageNet dataset that returns random data.
    Use this ONLY for testing pipeline functionality, not for actual evaluation.
    """
    
    def __init__(
        self,
        preprocess,
        location=None,
        batch_size=128,
        num_workers=4,
    ):
        warnings.warn(
            "Using DUMMY ImageNet dataset! Results will NOT be meaningful. "
            "This is only for testing the pipeline."
        )
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = DummyImageNetDataset(1000, preprocess)
        self.test_dataset = DummyImageNetDataset(1000, preprocess)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.classnames = [f"class_{i}" for i in range(1000)]
    
    def name(self):
        return "imagenet"


class DummyImageNetDataset(Dataset):
    """Generates random images for pipeline testing."""
    
    def __init__(self, size, preprocess):
        self.size = size
        self.preprocess = preprocess
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create a random image
        image = Image.new('RGB', (224, 224), color=(idx % 256, (idx * 2) % 256, (idx * 3) % 256))
        if self.preprocess is not None:
            image = self.preprocess(image)
        label = idx % 1000
        return image, label
