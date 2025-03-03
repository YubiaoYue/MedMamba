import os
import matplotlib.pyplot as plt
import sys
# import numpy as np

from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from MedMamba import VSSM as medmamba  # import model

DEFAULT_ROOT = './MedMamba/data/'
DATASET_PATH = os.path.join(DEFAULT_ROOT, "raw")

dataset_mean, dataset_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
normalization_transform = transforms.Normalize(mean=dataset_mean, std=dataset_std)

# Augmentation transforms (only applied to minority classes)
augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    normalization_transform
])

# Unaugmented data transform
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalization_transform
])

class LBCDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", val_split=0.2, seed=42, augment=False):
        """
        Custom Dataset for Mendeley LBC classification with train-validation splitting.
        Includes augmentation for minority classes.
        """
        assert split in ["train", "val"], "split must be 'train' or 'val'"

        self.root_dir = root_dir
        self.transform = transform if transform else data_transform
        self.split = split
        self.augment = augment

        self.class_to_idx = {
            "Negative for Intraepithelial malignancy": 0,  # NILM
            "Low squamous intra-epithelial lesion": 1,     # LSIL
            "High squamous intra-epithelial lesion": 2,    # HSIL
            "Squamous cell carcinoma": 3                   # SCC
        }

        image_paths, labels = [], []
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                                if img.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
                image_paths.extend(class_images)
                labels.extend([class_idx] * len(class_images))
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=val_split, random_state=seed, stratify=labels
        )

        self.image_paths = train_paths if split == "train" else val_paths
        self.labels = train_labels if split == "train" else val_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = augmentation_transforms(image) if self.augment and label in [1, 2, 3] else self.transform(image)
        return image, label

def save_plots(train_accuracies, val_accuracies, train_losses, val_losses):
    os.makedirs('./results', exist_ok=True)
    epochs_completed = len(train_accuracies)
    plt.figure()
    plt.plot(range(epochs_completed), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs_completed), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.savefig('./results/train_val_accuracy.png')
    
    plt.figure()
    plt.plot(range(epochs_completed), train_losses, label='Train Loss')
    plt.plot(range(epochs_completed), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.savefig('./results/train_val_loss.png')



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    train_dataset = LBCDataset(DATASET_PATH, split="train", augment=True)
    val_dataset = LBCDataset(DATASET_PATH, split="val", augment=False)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net = medmamba(num_classes=len(train_dataset.class_to_idx)).to(device)
        

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './MedMambaNet-LBC.pth'
    os.makedirs('./results', exist_ok=True)
    
    train_accuracies, val_accuracies, train_losses, val_losses = [], [], [], []

    for epoch in range(epochs):
        net.train()
        running_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, file=sys.stdout):
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)
        train_losses.append(avg_train_loss)
        
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader, file=sys.stdout):
                val_images, val_labels = val_images.to(device), val_labels.to(device).long()
                outputs = net(val_images)
                all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
        
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_accuracy)
        avg_val_loss = sum(train_losses) / len(train_losses)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.3f} | Train Acc: {train_accuracy:.3f} | Val Loss: {avg_val_loss:.3f} | Val Acc: {val_accuracy:.3f}")
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
            print(f"Best model updated at epoch {epoch+1} (Accuracy: {best_acc:.3f})")
        
        save_plots(train_accuracies, val_accuracies, train_losses, val_losses)

if __name__ == '__main__':
    main()
