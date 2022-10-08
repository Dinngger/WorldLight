
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt


def genMeshGrid(h, w):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    return torch.stack(torch.meshgrid(x, y, indexing='xy'))


class AttentionPooling(nn.Module):
    def __init__(self, d_in, d_out, n_out):
        super(AttentionPooling, self).__init__()
        self.d_in = d_in
        self.inducing_points = nn.Parameter(torch.Tensor(n_out, d_out, d_in))
        nn.init.xavier_uniform_(self.inducing_points)
    
    def forward(self, x):
        routing = torch.matmul(self.inducing_points.expand(x.shape[0], 1, 1), x.permute(0, 2, 1))
        routing = F.softmax(routing / np.sqrt(self.d_in), -1)
        return torch.matmul(routing, x)


class AttentionConv(nn.Module):
    def __init__(self, in_channels):
        super(AttentionConv, self).__init__()
        self.enc = nn.Conv2d(in_channels, 2 * in_channels, (1, 1))

    def forward(self, x):
        B, Ci, H, W = x.shape
        C = Ci * 2
        x = self.enc(x)
        x = F.unfold(x, (3, 3), padding=1).view(B, C, 3*3, -1) # (B, C, 9, N)
        x = x.permute(0, 3, 2, 1) # (B, N, 9, C)
        mid = x[:, :, 4:5, :]

        # [B, N, 1, C] x [B, N, C, 9] = [B, N, 1, 9]
        routing = torch.matmul(mid, x.permute(0, 1, 3, 2))
        routing = F.softmax(routing / np.sqrt(C), -1)
        # [B, N, 1, 9] x [B, N, 9, C] = [B, N, 1, C]
        res = torch.matmul(routing, x)
        return res.squeeze(2).permute(0, 2, 1).view(B, C, H, W)


class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        self.coords = None
        self.cnn = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.ap1 = AttentionPooling(8, 8, 10)
        self.feat2coords = nn.Conv1d(8, 2, 1)

        # self.acnn = AttentionConv(4)        # (B, 8, 14, 14)
        # self.acnn = nn.Conv2d(4, 8, 3, 1, 1)    # compare to acnn
        self.pointNet = nn.Conv1d(10, 64, 1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.cnn(x)
        top = self.pool(x)
        seeds = self.ap1(top)

        B, C, H, W = x.shape
        if self.coords is None:
            self.coords = genMeshGrid(H, W).unsqueeze(0).to(x.device) #(1, 2, H, W)
        x = torch.cat((self.coords.expand(B, 2, H, W), x), 1)
        points = x.view(B, 10, H * W)
        points = self.pointNet(points)
        feature = torch.max(points, 2)[0]
        return self.fc(feature)

mnist_train = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data',
    train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data',
    train=False, download=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = TCNN().to(device)

if False:
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print("Parameters: ", param_count)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), "tcnn.pt")
else:
    # model.load_state_dict(torch.load("tcnn.pt"))
    for X, y in train_loader:
        x = X.to(device)
        # x = F.conv2d(x, torch.ones((1, 1, 3, 3), device=x.device))
        # B, C, H, W = x.shape
        # x = F.softmax(x.view(B, -1), -1).view(B, C, H, W)
        # x = model.cnn(X.to(device)).relu()
        img = torchvision.utils.make_grid(x[:64, ...], normalize=True)
        img = np.squeeze(img.detach().cpu().numpy())
        img = np.transpose(img, [1,2,0])
        plt.imshow(img)
        plt.show()
        break
