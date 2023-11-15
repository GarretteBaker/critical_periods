# Models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from transformers import AutoModelForImageClassification
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from devinterp.slt import estimate_learning_coeff_with_summary, plot_learning_coeff_trace
from devinterp.optim import SGLD
import time

class AllCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AllCNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, 3, 2, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.Conv2d(192, 10, 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)

class FCNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FCNet, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(32*32, 2500),
            nn.BatchNorm1d(2500),
            nn.ReLU(),

            nn.Linear(2500, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),

            nn.Linear(2000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),

            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),

            nn.Linear(500, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)

# importing data

# Define a function to downsample and then upsample the images
def down_up_sample(x):
    try:
        # Ensure input is a single image tensor
        if x.ndim != 3:
            raise ValueError(f"Expected input tensor with 3 dimensions, got {x.ndim}")
        
        # Downsample to 8x8
        x_down = F.interpolate(x.unsqueeze(0), size=(8, 8), mode='area').squeeze(0)
        # Upsample back to 32x32 using bilinear interpolation
        x_up = F.interpolate(x_down.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
        
        return x_up
    except Exception as e:
        print(f"Error in down_up_sample: {e}")
        print(f"Input shape: {x.shape}")
        raise e


# Define the new transformation with down_up_sample
transform_train_blur = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    # Apply the down_up_sample function
    transforms.Lambda(lambda x: down_up_sample(x)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Original transformations for the test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Apply the new transform to the training dataset
train_dataset_blur = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_blur)
train_loader_blur = DataLoader(train_dataset_blur, batch_size=256, shuffle=True, num_workers=2)

# Original training and test datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Original DataLoaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# TODO: Add train loss, validation loss, and validation accuracy at blur removal epoch
def train_model(remove_blur_after, num_epochs, train_loader_blur, train_loader, test_loader, train_dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the model, criterion, optimizer, and scheduler
    model = resnet18(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    learning_coeff_at_blur_removal = None
    train_loss_at_blur_removal = None
    val_loss_at_blur_removal = None
    val_accuracy_at_blur_removal = None
    t0 = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Time elapsed: {time.time() - t0} seconds")
        t0 = time.time()
        # Select the correct data loader based on the current epoch
        current_loader = train_loader_blur if epoch <= remove_blur_after else train_loader

        # Train the model
        train_loss = train(model, current_loader, criterion, optimizer, device)
        
        # Scheduler step
        scheduler.step()

        if epoch == remove_blur_after:
            learning_coeff_stats = estimate_learning_coeff_with_summary(
                model,
                loader=train_loader_blur,
                criterion=criterion,
                sampling_method=SGLD,
                optimizer_kwargs=dict(lr=1e-5, elasticity=1.0, num_samples=len(train_dataset)),
                num_chains=10,
                num_draws=100,
                num_burnin_steps=0,
                num_steps_bw_draws=1,
                device=device
            )
            learning_coeff_at_blur_removal = learning_coeff_stats['mean']
            val_loss_at_blur_removal, val_accuracy_at_blur_removal = validate(model, test_loader, criterion, device)
            train_loss_at_blur_removal = train_loss
        
    # After the final epoch, perform validation and learning coefficient calculation
    val_loss, val_accuracy = validate(model, test_loader, criterion, device)
    learning_coeff_stats = estimate_learning_coeff_with_summary(
        model,
        loader=train_loader,  # Assuming we want the learning coefficient from the non-blurred dataset
        criterion=criterion,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=1e-5, elasticity=1.0, num_samples=len(train_dataset)),
        num_chains=10,
        num_draws=100,
        num_burnin_steps=0,
        num_steps_bw_draws=1,
        device=device
    )
    lambdahat = learning_coeff_stats['mean']

    # Return the final training loss, validation loss, validation accuracy, and lambdahat mean
    return (train_loss, val_loss, val_accuracy, lambdahat, learning_coeff_at_blur_removal,
            train_loss_at_blur_removal, val_loss_at_blur_removal, val_accuracy_at_blur_removal)
    
num_epochs = 240
# num_epochs=30 # for testing
step = num_epochs//14

# Looping over the function with different values for remove_blur_effect
results = []
for remove_blur_effect in tqdm(range(0, num_epochs+1, step)):
    final_train_loss, final_val_loss, final_val_accuracy, final_lambdahat, learning_coeff_at_blur_removal, train_loss_at_blur_removal, val_loss_at_blur_removal, val_accuracy_at_blur_removal = train_model(
        remove_blur_effect, num_epochs, train_loader_blur, train_loader, test_loader, train_dataset
    )
    results.append({
        'remove_blur_effect': remove_blur_effect,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_accuracy,
        'final_lambdahat': final_lambdahat,
        'learning_coeff_at_blur_removal': learning_coeff_at_blur_removal, 
        'train_loss_at_blur_removal': train_loss_at_blur_removal,
        'val_loss_at_blur_removal': val_loss_at_blur_removal,
        'val_accuracy_at_blur_removal': val_accuracy_at_blur_removal
    })
    
# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('results_128batch.csv', index=False)    

# loading results
results = pd.read_csv('results_128batch.csv')
# moving to dictionary
results = results.to_dict('records')

# Extracting the data for plotting
remove_blur_effects = [result['remove_blur_effect'] for result in results][:-1]
final_train_losses = [result['final_train_loss'] for result in results][:-1]
final_val_losses = [result['final_val_loss'] for result in results][:-1]
final_val_accuracy = [result['final_val_accuracy'] for result in results][:-1]
final_lambdahats = [result['final_lambdahat'] for result in results][:-1]
# learning_coeffs_at_blur_removal = [result['learning_coeff_at_blur_removal_'] for result in results][:-1]
# train_losses_at_blur_removal = [result['train_loss_at_blur_removal'] for result in results][:-1]
val_losses_at_blur_removal = [result['val_loss_at_blur_removal'] for result in results][:-1]
val_accuracies_at_blur_removal = [result['val_accuracy_at_blur_removal'] for result in results][:-1]

# Creating the Final Training Losses and LambdaHats vs. Blur Effect Removal Epoch plot
fig, ax1 = plt.subplots()

# Plotting train and validation losses on the left y-axis
ax1.set_xlabel('Remove Blur Effect (epoch)')
ax1.set_ylabel('Accuracy %', color='tab:blue')
# l1, = ax1.plot(remove_blur_effects, train_losses_at_blur_removal, 'b-', label='Train Loss')
l2, = ax1.plot(remove_blur_effects, final_val_accuracy, 'b-.', label='Final validation accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Instantiating a second y-axis to plot lambdahats
# ax2 = ax1.twinx()
# ax2.set_ylabel('Lambdahat', color='tab:red')
# #l3, = ax2.plot(remove_blur_effects, final_lambdahats, 'r-', label='Final LambdaHat')
# # l4, = ax2.plot(remove_blur_effects, learning_coeffs_at_blur_removal, 'r--', label='LambdaHat at Blur Removal')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# Adding a legend
lns = [l2]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1))

# Final touches
plt.title('Final validation accuracy vs. Blur Effect Removal Epoch')
fig.tight_layout()

# Save the figure
plt.savefig('accuracy_128batch.png')
plt.close()

# # Creating the Final & Blur Removal Validation Accuracies and LambdaHats vs. Blur Effect Removal Epoch plot
# fig, ax1 = plt.subplots()

# # Plotting final and blur removal validation accuracies on the left y-axis
# ax1.set_xlabel('Remove Blur Effect (epoch)')
# ax1.set_ylabel('Validation Accuracy', color='tab:blue')
# l1, = ax1.plot(remove_blur_effects, final_val_accuracy, 'b-', label='Final Validation Accuracy')
# # l2, = ax1.plot(remove_blur_effects, val_accuracies_at_blur_removal, 'b--', label='Validation Accuracy at Blur Removal')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Instantiating a second y-axis to plot lambdahats
# ax2 = ax1.twinx()
# ax2.set_ylabel('Loss', color='tab:red')
# l3, = ax2.plot(remove_blur_effects, learning_coeffs_at_blur_removal, 'r-', label='LambdaHat at Blur Removal')
# #l4, = ax2.plot(remove_blur_effects, val_losses_at_blur_removal, 'r--', label='Validation Loss at Blur Removal')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# # Adding a legend
# lns = [l1, l3]
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1))

# # Final touches
# plt.title('Validation Accuracies and LambdaHats vs. Blur Effect Removal Epoch')
# fig.tight_layout()

# # Save the figure
# plt.savefig('accuracies_and_lambdahats.png')
# plt.close()

# # Creating the Blur Removal Training Losses and LambdaHats vs. Blur Effect Removal Epoch plot
# fig, ax1 = plt.subplots()

# # Plotting train and validation losses on the left y-axis
# ax1.set_xlabel('Remove Blur Effect (epoch)')
# ax1.set_ylabel('Loss', color='tab:blue')
# l1, = ax1.plot(remove_blur_effects, train_losses_at_blur_removal, 'b-', label='Train Loss at Blur Removal')
# l2, = ax1.plot(remove_blur_effects, val_losses_at_blur_removal, 'b--', label='Validation Loss at Blur Removal')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Instantiating a second y-axis to plot lambdahats
# ax2 = ax1.twinx()
# ax2.set_ylabel('Lambdahat', color='tab:red')
# l3, = ax2.plot(remove_blur_effects, learning_coeffs_at_blur_removal, 'r-', label='LambdaHat at Blur Removal')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# # Adding a legend
# lns = [l1, l2, l3]
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1))

# # Final touches
# plt.title('Blur Removal Training Losses and LambdaHats vs. Blur Effect Removal Epoch')
# fig.tight_layout()

# # Save the figure
# plt.savefig('blur_removal_losses_and_lambdahats.png')
# plt.close()
