import torch
import torch.optim as optim
import torch.nn as nn
from models.modelCNN import CNN
from models.modelMLP import MLP
from process import create_dataloader
import yaml

import wandb


with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Initialize a new W&B run
wandb.init(project='cifar10-pytorch', 
           config={
            "architecture": config['model'],
            "dataset": "CIFAR-100",
            "epochs": config['num_epochs'],
            }
           , entity='tomboustedt')
cfg = wandb.config

if config['model'] == 'CNN': model = CNN(config['layer_1'], config['layer_2'])
elif config['model'] == 'MLP': model = MLP(config['input_size'], config['hidden_sizes'], config['output_size'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader, test_loader = create_dataloader()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

for epoch in range(config['num_epochs']):
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            wandb.log({"loss": running_loss / 100})
            running_loss = 0.0
            
print('Finished Training')

# Save the model checkpoint
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

# Evaluate on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
wandb.log({"accuracy": accuracy})

# Finish the W&B run
wandb.finish()