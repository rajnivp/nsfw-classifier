import torch
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import ctypes
from PIL import ImageFile

# This will deal with file size errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# For running on gpu
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# Define batch size
# 20 percent train data will be used as validation data
batch_size = 64
valid_size = 0.2
num_workers = 0

# Path of training images folder
TRAIN_DATA_PATH = "category/train/"

# Apply transformations
transform = transforms.Compose(
    [transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform)

# Make training and validation samples
num_train = len(train_data)
indice = list(range(num_train))
np.random.shuffle(indice)
split = int(np.floor(num_train * valid_size))
train_idx, valid_idx = indice[split:], indice[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Load data in defined batch sizes
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers)

dataiter = iter(train_loader)
images, labels = dataiter.next()

# Set device to cuda to run on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Will be using pre trained resnet 50 model for classification
model = models.resnet50(pretrained=True)

# Set gradients to false
for param in model.parameters():
    param.requires_grad = False

# Define classifier based inputs sizes, output sizes and number of classes
classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512, 5),
                           nn.LogSoftmax(dim=1))

# Add classifier to model
model.fc = classifier

# Set loss function, optimizer and learning rate
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
model.to(device)

# Define number of iterations on training data for model training
epochs = 30
steps = 0
v = 0

# Set initial valid loss to infinity so model get saved after first iteration
valid_loss_min = np.inf

# Create list to keep track of losses and accuracies to plot them on graph later
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(epochs):
    # Set initial losses and accuracies to zero
    train_loss = 0
    valid_loss = 0
    train_accuracy = 0
    valid_accuracy = 0

    # Create list of labels and predictions
    # this will keep track of input labels and model predictions to plot confusion matrix
    all_labels = []
    all_preds = []

    for images, labels in train_loader:
        steps += 1
        print(steps)

        # Move images and labels to gpu
        images, labels = images.to(device), labels.to(device)

        # Remove gradient
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)

        # Back propagation and optimizer steps
        loss.backward()
        optimizer.step()

        # Get predictions for images
        preds = torch.argmax(logps, dim=1)

        # Compare predictions to true labels
        res = torch.eq(labels, preds)

        # Calculate loss and accuracy
        train_accur = (int(torch.sum(res)) / len(labels)) * 100
        train_loss += loss.item()
        train_accuracy += train_accur
        all_labels.append(labels)
        all_preds.append(preds)

    for images, labels in valid_loader:
        v += 1
        print(v)
        # Move images and labels to gpu
        images, labels = images.to(device), labels.to(device)

        # This is validation loop so we will not do back propagation and optimizer steps
        logps = model(images)
        loss = criterion(logps, labels)

        # Get predictions for images
        preds = torch.argmax(logps, dim=1)

        # Compare predictions to true labels
        res = torch.eq(labels, preds)

        # Calculate loss and accuracy
        valid_accur = (int(torch.sum(res)) / len(labels)) * 100
        valid_loss += loss.item()
        valid_accuracy += valid_accur
        all_labels.append(labels)
        all_preds.append(preds)

    # Calculate overall loss and accuracy
    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)

    valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(valid_loss)

    train_accuracy = train_accuracy / len(train_loader)
    train_accuracies.append(train_accuracy)

    valid_accuracy = valid_accuracy / len(valid_loader)
    valid_accuracies.append(valid_accuracy)

    # Convert list of tensors into single tensor
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    print("epoch:{} \ttrainloss {}".format(epoch + 1, train_loss))
    print("epoch:{} \tvalidloss {}".format(epoch + 1, valid_loss))
    print("epoch:{} \ttrainaccuracy {}".format(epoch + 1, train_accuracy))
    print("epoch:{} \tvalidaccuracy {}".format(epoch + 1, valid_accuracy))

    # Save model if validation loss decreases
    if valid_loss <= valid_loss_min:
        print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), "model.pt")
        valid_loss_min = valid_loss

# Move all labels and predictions to cpu and flatten them to create confusion matrix
all_labels, all_preds = all_labels.cpu(), all_preds.cpu()
cm = confusion_matrix(all_labels.view(-1), all_preds.view(-1))


def save_plot():
    """
    plot charts to compare training and validation losses and accuracies
    and save them as png
    """
    plt.plot(train_losses, label="train loss")
    plt.plot(valid_losses, label="valid loss")
    plt.title("loss comparison")
    plt.legend(loc="upper left")
    plt.savefig("charts/losses.png")
    plt.close()

    plt.plot(train_accuracies, label="train accuracy")
    plt.plot(valid_accuracies, label="valid accuracy")
    plt.legend(loc="upper left")
    plt.title("accuracy")
    plt.savefig("charts/accuracy.png")
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """
    plot chart to show confusion matrix
    it will show number of right and wrong predictions for each class
    and save it as png
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("charts/confusion_matrix.png", bbox_inches="tight")
    plt.close()


save_plot()
plot_confusion_matrix(cm, train_data.classes)
