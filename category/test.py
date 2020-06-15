from category.helper import *
import torchvision.models as models
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
from PIL import Image
import sys

batch_size = 20
num_workers = 0

# Folder for test images
TEST_DATA_PATH = "category/test/"

# Set device to cuda to run on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def procces_image(image_path):
    """
    Apply transformations to image and returned transformed image
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


def load_model():
    """
    Load trained model and put saved weights to our model architecture
    return model with updated weights
    """
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(2048, 512),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(512, 5),
                               nn.LogSoftmax(dim=1))

    model.fc = classifier
    model.to(device)

    # Load saved model and set it to evaluation mode
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    return model


def test_loader(model, mapping):
    """
    Build test loader for loading images in batches
    iterate over test loader to test results
    Run this function to plot all images in one batch with predicted label
    """
    test_data = datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # Test model on batch of images
    output = model(images)
    _, preds = torch.max(output, 1)

    # Move images and preds to cpu to plot them
    images, preds = images.cpu(), preds.cpu()

    fig = plt.figure(figsize=(25, 4))

    for idx in range(batch_size):
        ax = fig.add_subplot(batch_size // 10, batch_size // (batch_size // 10), idx + 1, xticks=[], yticks=[])
        imshow(images[idx], ax, title=mapping[preds[idx].item()])


def predict(image_path, model, topk=5):
    """
    Returns a label ,classes and probabilities predicted by model for given image
    """
    image = procces_image(image_path)

    output = model(image.to(device))
    _, preds = torch.max(output, 1)
    ps = F.softmax(output, dim=1)
    topk = ps.cpu().topk(topk)
    image, preds = image.cpu(), preds.cpu()
    return preds.item(), (e.data.numpy().squeeze().tolist() for e in topk)


def main():
    image_path = sys.argv[1]
    model = load_model()
    pred, (probs, classes) = predict(image_path, model)
    mapper = {0: "drawing-SFW", 1: "hentai-NSFW", 2: "neutral-SFW", 3: "porn-NSFW", 4: "sexy-SFW"}
    view_classify(image_path, probs, classes, pred, mapper)

    #test_loader(model,mapper)

    plt.show()

if __name__ == '__main__':
    main()

