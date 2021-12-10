import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, CenterCrop, ConvertImageDtype, ColorJitter, Pad, \
    Grayscale, RandomAffine, RandomResizedCrop, RandomInvert, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.utils import make_grid

from Continuum.ContinuumBenchmarks.PYTorchPlayground import Config

if __name__ == '__main__':
    print(torch.__version__)
    # Since v0.8.0 all random transformations are using torch default random generator to sample random parameters.
    # It is a backward compatibility breaking change and user should set the random state.
    # https://pytorch.org/vision/stable/transforms.html
    # https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py

    import torch

    torch.manual_seed(17)

    # -------------------------------------------------------------------------------------------------#
    # Composition of Transformation
    compose_trfm = transforms.Compose([
        ToTensor(),
        CenterCrop(32),
        RandomAffine(degrees=(0, 90)),
        Pad(padding=10),
        ConvertImageDtype(torch.float),
        RandomResizedCrop(size=(32, 32)),
        RandomInvert(p=0.2),
        RandomVerticalFlip(p=0.5),
        RandomHorizontalFlip(p=0.5)
    ])
    cifar_dataset = CIFAR10(root=Config.data_dir, train=True, download=True, transform=compose_trfm,
                            target_transform=None)
    train_loader = DataLoader(dataset=cifar_dataset, shuffle=False, num_workers=1, batch_size=16)
    imgs, _ = iter(train_loader).next()

    grid_img = make_grid(tensor=imgs, nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0), interpolation="nearest")
    plt.title("Compositions of Tranformations")
    plt.show()

    # -------------------------------------------------------------------------------------------------#
    # Scriptable Transformation (Works with Tensors only)
    cifar_dataset = CIFAR10(root=Config.data_dir, train=True, transform=ToTensor(), download=True,
                            target_transform=None)
    train_loader = DataLoader(dataset=cifar_dataset, shuffle=False, num_workers=1, batch_size=16)
    imgs, _ = iter(train_loader).next()
    script_trfm = nn.Sequential(
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        CenterCrop(32*2),
        # ColorJitter(brightness=10),
        # Grayscale(num_output_channels=1),
        # RandomVerticalFlip(p=0.5),
        # RandomHorizontalFlip(p=0.5)
    )
    imgs = nn.Upsample(scale_factor=1.5)(imgs)
    script_trfm = torch.jit.script(script_trfm)
    grid_img = make_grid(tensor=script_trfm(imgs), nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0), interpolation="nearest")
    plt.title("Scriptable Transforms")
    plt.show()
    # -------------------------------------------------------------------------------------------------#
