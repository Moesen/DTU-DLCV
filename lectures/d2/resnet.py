import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
import json
from clogger import init_logger


log_folder = Path("log")

logger = init_logger(__name__, testing_mode=True, log_folder=log_folder)

if torch.cuda.is_available():
    logger.info("The code will run on GPU.")
else:
    logger.warning(
        "The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator"
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fun(output, target):
    return F.nll_loss(F.log_softmax(output, dim=1), target)


# We define the training as a function so we can easily re-use it.
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
):
    out_dict = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for _ in tqdm(range(num_epochs), unit="epoch"):
        model.train()
        # For each epoch
        train_correct = 0
        train_loss = []
        for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = loss_fun(output, target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
        # Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target == predicted).sum().cpu().item()
        out_dict["train_acc"].append(train_correct / len(train_loader.dataset))  # type: ignore
        out_dict["test_acc"].append(test_correct / len(test_loader.dataset))  # type: ignore
        out_dict["train_loss"].append(np.mean(train_loss))
        out_dict["test_loss"].append(np.mean(test_loss))
        logger.info(
            f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t\nAccuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%",
        )
    return out_dict


class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding="same"),
	    nn.Dropout(p=.2),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, 3, padding="same"),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        wout = self.weight_layer(x)
        fxpx = wout + x
        out = self.relu(fxpx)
        return out


class ResNet(nn.Module):
    def __init__(self, n_in, n_features, num_res_blocks=3):
        super(ResNet, self).__init__()
        # First conv layers needs to output the desired number of features.
        conv_layers = [
            nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
	    nn.Dropout(p=.2),
            nn.ReLU(),
        ]
        for i in range(num_res_blocks):
            conv_layers.append(ResNetBlock(n_features))
        self.res_blocks = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * n_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.res_blocks(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    logger.info("loading dataset")
    batch_size = 64
    trainset = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(20)])
    )
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    logger.info("train dataset loaded")
    testset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    logger.info("test dataset loaded")

    model = ResNet(3, 8)
    model.to(device)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 10
    logger.info("model created")

    logger.info("beginning training")
    out_dict = train(model, optimizer, train_loader, test_loader, num_epochs=num_epochs)
    logger.info("Training done")

    logger.info("creating report")
    _, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(out_dict['train_acc'])
    ax1.plot(out_dict['test_acc'])

    ax2.plot(out_dict["train_loss"])
    ax2.plot(out_dict["test_loss"])


    ax1.legend(('Train acc','Test acc'))
    ax2.legend(('Train error','Test eror'))
    ax1.set_xlabel('Epoch number')
    ax2.set_xlabel('Epoch number')
    plt.ylabel('Accuracy')

    plt.savefig(log_folder / "exercise1-4.png")
    with open(log_folder / "exercise1-4_results.json", "w") as f:
        f.write(json.dumps(out_dict, indent=2))
