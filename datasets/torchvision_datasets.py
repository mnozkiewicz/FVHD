from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from functools import partial

from typing import Type, Tuple, Optional
import torch
from torchvision.transforms import v2
from torchvision.datasets import (
    MNIST,
    FashionMNIST,
    QMNIST,
    KMNIST,
    VisionDataset,
    EMNIST
)

def load_split(
        dataset: type[VisionDataset],
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

    torch_dataset = dataset(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    images, labels = zip(*torch_dataset)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.int)

    return images, labels


def get_dataset_class(dataset_name: str) -> Type[VisionDataset]:
    match dataset_name:
        case "mnist":
            return MNIST
        case "fmnist":
            return FashionMNIST
        case "qmnist":
            return QMNIST
        case "kmnist":
            return KMNIST
        case "emnist_letters":
            return partial(EMNIST, split="letters")
        case "emnist_byclass":
            return partial(EMNIST, split="byclass")
        case "emnist_bymerge":
            return partial(EMNIST, split="bymerge")
        case "emnist_balanced":
            return partial(EMNIST, split="balanced")
        case _:
            raise ValueError(f"No dataset named {dataset_name}")

class TorchvisionDataset:

    def __init__(self, dataset_name: str):
        self.dataset_class = get_dataset_class(dataset_name)

        self._train_images: Optional[torch.Tensor] = None
        self._train_labels: Optional[torch.Tensor] = None

        self._test_images: Optional[torch.Tensor] = None
        self._test_labels: Optional[torch.Tensor] = None


    def train_images(self) -> torch.Tensor:
        if self._train_images is None:
            self._train_images, self._train_labels = load_split(
                self.dataset_class, 
                train=True
            )

        return self._train_images.clone().reshape(self._train_images.shape[0], -1)
    
    def train_labels(self) -> torch.Tensor:
        if self._train_labels is None:
            self._train_images, self._train_labels = load_split(
                self.dataset_class, 
                train=True
            )

        return self._train_labels.clone()
    
    def test_images(self) -> torch.Tensor:
        if self._test_images is None:
            self._test_images, self._test_labels = load_split(
                self.dataset_class, 
                train=False
            )

        return self._test_images.clone().reshape(self._test_images.shape[0], -1)
    
    def test_labels(self) -> torch.Tensor:
        if self._test_labels is None:
            self._test_images, self._test_labels = load_split(
                self.dataset_class, 
                train=False
            )

        return self._test_labels.clone()
    

    def show_examples(self, number_of_images: int, revert_colors: bool = False):
        self.test_images()
        random_indices = torch.randperm(len(self._test_images))[:number_of_images]
        sample_images = self._test_images[random_indices]
        
        grid = make_grid(sample_images, nrow=4).permute(1, 2, 0)

        if revert_colors:
            grid = 1 - grid

        plt.imshow(grid)
        plt.axis("off")
        plt.show()


    def unique_labels(self) -> torch.Tensor:
        labels = self.train_labels()
        return torch.sort(torch.unique(labels)).values
