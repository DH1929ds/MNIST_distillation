import torch
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def save_grid_images_with_info(cache_dir, output_path, num_samples=100):
    # 캐시된 이미지, 라벨, timestep 파일 불러오기
    img_cache_path = os.path.join(cache_dir, "mnist_images.pt")
    class_cache_path = os.path.join(cache_dir, "mnist_labels.pt")
    t_cache_path = os.path.join(cache_dir, "mnist_t.pt")
    
    img_cache = torch.load(img_cache_path)
    class_cache = torch.load(class_cache_path)
    t_cache = torch.load(t_cache_path)
    
    print(img_cache[0])
    # 이미지, 라벨, timestep의 전체 길이 확인
    total_samples = img_cache.size(0)
    
    # 랜덤 인덱스 100개 선택
    random_indices = random.sample(range(total_samples), num_samples)
    
    # 선택된 인덱스의 이미지, 라벨, timestep 가져오기
    selected_imgs = img_cache[random_indices]
    selected_classes = class_cache[random_indices]
    selected_tsteps = t_cache[random_indices]
    
    # 이미지 그리드를 만들어 matplotlib으로 시각화
    grid_img = vutils.make_grid(selected_imgs.cpu(), nrow=10, padding=2, normalize=True)
    
    # 그리드 이미지 저장을 위해 PIL 변환
    grid_np = grid_img.numpy().transpose(1, 2, 0) * 255  # [C, H, W] -> [H, W, C] 및 픽셀 값 범위를 [0, 255]로 변환
    grid_np = grid_np.astype(np.uint8)  # NumPy 배열로 변환
    grid_pil = Image.fromarray(grid_np)  # PIL 이미지로 변환
    draw = ImageDraw.Draw(grid_pil)
    
    # MNIST 이미지가 작은 점을 고려해 글씨 크기 조정
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)  # 적절한 크기의 폰트 설정
    
    # 각 이미지에 해당하는 클래스와 timestep 정보 추가
    img_size = selected_imgs.shape[2]  # MNIST 이미지 크기 (28x28)
    for i, idx in enumerate(random_indices):
        row = i // 10
        col = i % 10
        x = col * (img_size + 2)  # 이미지 간격에 맞춰서 위치 계산
        y = row * (img_size + 2)
        
        # 클래스와 timestep 정보 추가 (작은 글씨 크기로 조정)
        class_label = f"C:{selected_classes[i].item()}"
        t_label = f"T:{int(selected_tsteps[i].item())}"
        
        # 이미지 위에 텍스트 그리기
        draw.text((x + 1, y + 1), f"{class_label}, {t_label}", font=font, fill=(255, 255, 255))

    # 출력 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 최종 그리드 이미지를 저장
    grid_pil.save(output_path)
    print(f"Saved grid image with class and t info to {output_path}")

# 사용 예시
cache_dir = "./cache"  # 캐시 파일이 저장된 디렉토리 경로
output_path = "./output_images/mnist_grid_image_with_info.png"
save_grid_images_with_info(cache_dir, output_path)
