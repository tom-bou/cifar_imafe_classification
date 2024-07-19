import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from process import create_dataloader, classes
from models.modelCNN import CNN
from models.modelMLP import MLP
from sklearn.metrics import confusion_matrix
import yaml

with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if config['model']=='CNN': model = CNN(config['layer_1'], config['layer_2'])
elif config['model']=='MLP': model = MLP(config['input_size'], config['hidden_sizes'], config['output_size'])
model.load_state_dict(torch.load('model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader, test_dataloader = create_dataloader()

all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_dataloader)
images, labels = next(dataiter)

outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(14, 14))
for idx in np.arange(16):
    ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(f"True: {classes[labels[idx]]}\nPred: {classes[predicted[idx]]}", 
                 color=("green" if predicted[idx] == labels[idx] else "red"))
plt.savefig("correct_incorrect_preds.png")

