import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from datasets import Dataset, DatasetDict
import os
import random
import cv2
from tqdm import tqdm

# Configuration
real_folder = "/home/adithya/chamber_of_secrets/DeepFakeDetection/monotonic_curriculum/dataset/real"
fake_folder = "/home/adithya/chamber_of_secrets/DeepFakeDetection/monotonic_curriculum/dataset/fake"
image_extensions = ['.jpg', '.jpeg', '.png']

# Create dataset from folders
def create_dataset(folder, label):
    samples = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    samples.append({
                        'image': img,
                        'label': label,
                        'file_path': img_path  # Optional but useful for debugging
                    })
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    return samples

# Create real and fake datasets
real_samples = create_dataset(real_folder, label=0)
fake_samples = create_dataset(fake_folder, label=1)

# Combine into Hugging Face dataset format
full_dataset = real_samples + fake_samples

# Create train/validation splits (80/20 split)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset = Dataset.from_list(full_dataset[:train_size])
val_dataset = Dataset.from_list(full_dataset[train_size:])

# Create DatasetDict similar to original FF++ structure
ffpp_dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

real_samples = [s for s in ffpp_dataset['train'] if s['label'] == 0]
o_fake_samples = [s for s in ffpp_dataset['train'] if s['label'] == 1]

def curriculum_schedule(epoch, total_epochs):
    epsilon = 2 * total_epochs / np.pi
    return np.sin(epoch / epsilon)

def blending_augmentation(real_img):
    img1 = np.array(real_img)
    img2 = np.array(random.choice(real_samples)['image'])
    
    # Resize both images to a fixed size (e.g., 224x224)
    img1_resized = cv2.resize(img1, (224, 224))
    img2_resized = cv2.resize(img2, (224, 224))
    
    # Blend the resized images
    alpha = 0.3 + 0.4 * random.random()  # Random blend ratio (30%-70%)
    blended = (alpha*img1_resized + (1-alpha)*img2_resized).astype(np.uint8)
    
    return Image.fromarray(blended)

class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, real_samples, o_fake_samples, total_epochs):
        self.real = real_samples
        self.o_fake = o_fake_samples
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def __len__(self):
        return len(self.real) + len(self.o_fake)
        
    def __getitem__(self, idx):
        # Calculate current curriculum ratio
        q_t = curriculum_schedule(self.current_epoch, self.total_epochs)
        
        # 50% real, 50% mixed fake
        if idx % 2 == 0:
            real_sample = random.choice(self.real)
            return {'image': real_sample['image'], 'label': 0}
        else:
            # Decide o-fake vs p-fake
            if random.random() > q_t:
                fake_sample = random.choice(self.o_fake)
                return {'image': fake_sample['image'], 'label': 1}
            else:
                real_sample = random.choice(self.real)
                p_fake_image = blending_augmentation(real_sample['image'])
                return {'image': p_fake_image, 'label': 0.5}  # Label for blended image

curriculum_ds = CurriculumDataset(real_samples, o_fake_samples, total_epochs=50)

# Access real samples
print(len(curriculum_ds.real))

# Access original fake samples
print(len(curriculum_ds.o_fake))

# Set and get the current epoch
epochs = 10;
curriculum_ds.set_epoch(10)
print(curriculum_ds.current_epoch)

def count_p_fakes_per_epoch(dataset, total_epochs):
    p_fake_counts = []

    for epoch in range(1, total_epochs + 1):
        dataset.set_epoch(epoch)
        count = 0

        for idx in range(len(dataset)):
            sample = dataset[idx]
            # A p-fake is a fake with a blended image
            if sample['label'] == 0.5:
                count += 1

        print(f"Epoch {epoch}: {count} p-fakes out of {len(dataset)} samples")
        p_fake_counts.append(count)

    return p_fake_counts

# Call it like this
p_fake_stats = count_p_fakes_per_epoch(curriculum_ds, total_epochs=10)
