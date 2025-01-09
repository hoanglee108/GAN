import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from GAN import Generator

# Giả sử bạn đã định nghĩa và huấn luyện các mô hình Generator và Discriminator ở đây.
# Hãy chắc chắn rằng mô hình đã được tải vào từ file .pth nếu bạn đã lưu chúng trước đó.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải mô hình đã huấn luyện
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc hình ảnh từ camera.")
        break

    # Hiển thị hình ảnh từ webcam
    cv2.imshow('Camera', frame)

    # Chuyển đổi ảnh từ OpenCV (BGR) sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    frame_tensor = frame_tensor * 2 - 1  # Chuẩn hóa ảnh trong khoảng [-1, 1]

    # Sinh ảnh giả từ Generator
    z = torch.randn(1, 100).to(device)  # Latent vector
    fake_image = generator(z).cpu().detach()

    # Chuyển fake_image về dạng ảnh và hiển thị
    fake_image = fake_image.squeeze().permute(1, 2, 0).numpy()  # Chuyển đổi về HWC
    fake_image = (fake_image + 1) / 2  # Đưa giá trị về khoảng [0, 1]

    # Hiển thị ảnh giả
    plt.imshow(fake_image)
    plt.axis('off')
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
