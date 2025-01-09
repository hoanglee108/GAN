from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

# Định nghĩa transform để chuẩn hóa và resize ảnh
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CustomDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.real_images = os.listdir(real_dir)
        self.fake_images = os.listdir(fake_dir)
        self.transform = transform

    def __len__(self):
        return len(self.real_images) + len(self.fake_images)

    def __getitem__(self, idx):
        if idx < len(self.real_images):
            img_name = os.path.join(self.real_dir, self.real_images[idx])
            label = 1  # Hình ảnh thật
        else:
            img_name = os.path.join(self.fake_dir, self.fake_images[idx - len(self.real_images)])
            label = 0  # Hình ảnh giả
        
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
