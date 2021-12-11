import torch
from tensorboardX import SummaryWriter

writer = SummaryWriter()

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

import time
def train_model(iter):
    for epoch in range(iter):
        print("E:", epoch)
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.flush()
        time.sleep(1)



train_model(1000)
writer.flush()
writer.close()