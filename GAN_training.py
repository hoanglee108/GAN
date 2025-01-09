import torch.nn as nn
import torch.optim as optim
from GAN import Generator, Discriminator
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Data_loader import CustomDataset, transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Hàm hiển thị 9 hình ảnh đầu ra từ Generator
def show_generated_images(epoch, generator, device):
    z = torch.randn(9, 100).to(device)  # Tạo 9 latent vectors
    fake_images = generator(z).cpu().detach()  # Sinh ra 9 hình ảnh
    fake_images = fake_images.permute(0, 2, 3, 1).numpy()  # Chuyển đổi dạng tensor (N, C, H, W) thành (N, H, W, C)

    # Chuyển giá trị từ [-1, 1] về [0, 1]
    fake_images = (fake_images + 1) / 2.0

    # Tạo grid 3x3 để hiển thị
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i])  # Hiển thị hình ảnh
        ax.axis('off')  # Tắt hiển thị trục
    plt.suptitle(f'Generated Images - Epoch {epoch}')
    plt.show()

# Ví dụ về quá trình huấn luyện và hiển thị
def train_gan(num_epochs, dataloader, generator, discriminator, optimizer_G, optimizer_D, criterion, device):
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Tạo nhãn cho hình ảnh thật
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Huấn luyện Discriminator
            optimizer_D.zero_grad()

            # Đánh giá hình ảnh thật
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            # Tạo hình ảnh giả
            z = torch.randn(batch_size, 100).to(device)  # Vector latent
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Huấn luyện Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)  # Generator cố gắng đánh lừa Discriminator
            g_loss.backward()
            optimizer_G.step()

        print(f'Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # Hiển thị 9 hình ảnh đầu ra mỗi epoch
        # if epoch % 10 == 0:  # Hiển thị sau mỗi 10 epoch
        #     show_generated_images(epoch, generator, device)

        
# Tạo DataLoader cho cả dữ liệu thực và giả
real_dir = './GAN_data/rem_preprocessed_512/'
fake_dir = './GAN_data/rem_fakes/'
dataset = CustomDataset(real_dir, fake_dir, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
num_epochs = 100
train_gan(num_epochs, dataloader, generator, discriminator, optimizer_G, optimizer_D, criterion, device)

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
