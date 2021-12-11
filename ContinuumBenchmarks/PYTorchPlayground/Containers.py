from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ModuleList, ParameterList, ParameterDict

if __name__ == '__main__':
    print("Welcome to containers.")


    # --------------------------------------------------------------------------------------- #
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            print("Model: __init__()")

        def forward(self, x):
            return F.relu(x)


    m = Model()
    x = torch.rand((2, 2))
    print("Before:", x)
    x = m(x)
    print("After:", x)
    print("Children:", len(list(m.children())))
    print(m.eval())
    print("-" * 100)
    # --------------------------------------------------------------------------------------- #
    l = nn.Linear(in_features=2, out_features=1)
    print("-" * 100)
    print("Weight:", l.weight, " Bias:", l.bias)
    print("-" * 100)
    with torch.no_grad():
        l.weight.fill_(0.25)
        l.bias.fill_(100)
    print("-" * 100)
    print("Weight:", l.weight, " Bias:", l.bias)
    print("-" * 100)
    l.eval()
    print(list(l.parameters()))
    print("-" * 100)
    # --------------------------------------------------------------------------------------- #
    sequential_1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3)),
        nn.Linear(in_features=10, out_features=1)
    )

    sequential_2 = nn.Sequential(OrderedDict([
        ("Conv", nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3))),
        ("Linear", nn.Linear(in_features=10, out_features=1))
    ]))
    print(sequential_1)
    print(sequential_2)
    print("-" * 100)
    # --------------------------------------------------------------------------------------- #
    m1 = nn.Linear(in_features=2, out_features=2)
    print("M1:", m1(torch.rand((1, 2))))
    m2 = nn.Linear(in_features=2, out_features=1)
    module_list = ModuleList(modules=[m1, m2])
    # print("Module_List:", module_list(torch.rand((1, 2))))
    print(module_list)
    print("*" * 10)
    print(list(module_list[0].parameters()))
    print("*" * 10)
    print(list(module_list[1].parameters()))
    print("-" * 100)


    # --------------------------------------------------------------------------------------- #
    class MultiHead(nn.Module):

        def __init__(self):
            super().__init__()
            h1 = nn.Linear(in_features=4, out_features=1)
            h2 = nn.Linear(in_features=4, out_features=2)
            h3 = nn.Linear(in_features=4, out_features=3)
            self.multihead = nn.ModuleDict({
                "h1": h1,
                "h2": h2,
                "h3": h3
            })

        def forward(self, x, head_id):
            return self.multihead[head_id](x)


    x = torch.randn((1, 4))
    multihead = MultiHead()
    print("Head 1:", multihead.forward(x=x, head_id="h1"))
    print("Head 2:", multihead.forward(x=x, head_id="h2"))
    print("Head 3:", multihead.forward(x=x, head_id="h3"))
    print("-" * 100)
    # --------------------------------------------------------------------------------------- #
    param_list = ParameterList([
        nn.Parameter(torch.rand((1, 5)), requires_grad=True),
        nn.Parameter(torch.rand((5, 5)), requires_grad=True)
    ])
    print(param_list)
    print("*" * 10)
    print(param_list[0])
    print("*" * 10)
    print((param_list[1]))
    print("-" * 100)
    # --------------------------------------------------------------------------------------- #
    param_dict = ParameterDict({
        "param_1": nn.Parameter(torch.rand((1, 1))),
        "param_2": nn.Parameter(torch.rand((1, 2))),
        "param_3": nn.Parameter(torch.rand((1, 3))),
        "param_4": nn.Parameter(torch.rand((1, 4))),
        "param_5": nn.Parameter(torch.rand((1, 5)))

    })
    print(param_dict["param_1"])
    print(param_dict["param_2"])
    print(param_dict["param_3"])
    print(param_dict["param_4"])
    print(param_dict["param_5"])

    # --------------------------------------------------------------------------------------- #
    print("-" * 100)


    # Register Module Full Backward Hook
    def backward_hook(module, grad_input, grad_output):
        print("Module:", module)
        print("Grad_input:", grad_input.size())
        print("Grad_output:", grad_output.size())

    def forward_hook(module, input, output):
        print("^" * 10)
        print("Module:", module)
        print("input:", input)
        print("output:", output)
        print("^" * 10)

    # def backward_hook():
    m = nn.Linear(in_features=10, out_features=1)
    sequence = nn.Sequential(nn.Linear(in_features=10, out_features=10),
                             nn.Linear(in_features=10, out_features=10),
                             nn.Linear(in_features=10, out_features=1))
    # sequence.register_module_full_backward_hook(backward_hook)
    # sequence.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)
    x = torch.rand((1, 10))
    print("X:", x)
    print(sequence(x))
    print("*" * 10)
    print("-" * 100)
    # --------------------------------------------------------------------------------------- #
    # Batch Normalization
    b_norm = nn.BatchNorm1d(num_features=10)
    b_norm(torch.rand((1, 10, 10)))
    b_norm(torch.rand((1, 10, 10)))
    b_norm(torch.rand((1, 10, 10)))
    print(b_norm.weight)
    print(b_norm.bias)
    print(b_norm)
    print("-" * 100)


    # --------------------------------------------------------------------------------------- #


