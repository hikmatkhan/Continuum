import torch
from torch import nn
# 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
# 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

if __name__ == '__main__':

    rand_img = torch.rand((1, 1, 28, 28))

    # https://pytorch.org/docs/stable/nn.html#containers
    # -------------------------------------------------------------------------------------------------#
    # Flatten
    rand_img = torch.rand((1, 1, 28,28))
    flatten = nn.Flatten()
    dropout = nn.Dropout(p=0.9)
    print("Before:", rand_img.size())
    print("After:", flatten(rand_img).size())

    flat_img = flatten(rand_img)
    print("Flat Image:", flat_img.size())
    unflatten = nn.Unflatten(unflattened_size=(1 ,28 ,28), dim=1)
    print("Unflat Image:", unflatten(flat_img).size())
    print("-" * 100)
    # -------------------------------------------------------------------------------------------------#
    # Linear
    linear = nn.Linear(in_features=flatten(rand_img).size()[1], out_features=10)
    print(linear(flatten(rand_img)))
    print("-" * 100)

    # -------------------------------------------------------------------------------------------------#
    # Convolution
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3))
    print("Before:", rand_img.size())
    print("After:", conv(rand_img).size())
    print("-" * 100)

    # -------------------------------------------------------------------------------------------------#
    # Pooling
    pool = nn.MaxPool2d(kernel_size=(2, 2))
    print("Before:", rand_img.size())
    print("After:", pool(rand_img).size())
    print("-" * 100)

    # -------------------------------------------------------------------------------------------------#
    # Non-linear Activation
    relu = nn.ReLU(inplace=True)
    print(relu(linear(flatten(rand_img))))
    print("-" * 100)

    # -------------------------------------------------------------------------------------------------#
    # Softmax
    relu = nn.Softmax()
    print(relu(linear(flatten(rand_img))))
    print("-" * 100)

    # -------------------------------------------------------------------------------------------------#
    # Distance function
    # https://pytorch.org/docs/stable/nn.html#distance-functions
    cosine_distance = nn.CosineSimilarity()
    x_1 = torch.rand((1, 2))
    x_2 = torch.rand((1, 2))
    print("Norm of X1:", torch.norm(x_1))
    print("Norm of X2:", torch.norm(x_2))
    print("X1:", x_1, " X2:", x_2)
    print("Cosine Distance:", cosine_distance(x_1, x_2))

    m = nn.Upsample(scale_factor=2)
    print(m(rand_img).size())

    # -------------------------------------------------------------------------------------------------#
    # EOF