import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

from funcs import visualize_t_cache_distribution

import os, random, math

# Custom Dataset class using only cached data
class MNISTDataset_x0(Dataset):
    def __init__(self, cached_images, cached_labels):
        self.cached_images = cached_images
        self.cached_labels = cached_labels

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        label = self.cached_labels[idx]

        return img, label
    
class MNISTDataset(Dataset):
    def __init__(self, img_cache, t_cache, class_cache, n_T, cond_sharing, cache_dir, lambda_param=5):
        self.img_cache = img_cache
        self.t_cache = t_cache
        self.class_cache = class_cache
        self.n_T = n_T
        self.cond_sharing = cond_sharing
        self.total_class = [0,1,2,3,4,5,6,7,8,9]
        self.cache_dir = cache_dir
        self.lambda_param = lambda_param
        
        # 제외할 클래스 정의
        classes_to_exclude = [1, 3, 5, 7, 9]

        # class_cache가 있는 디바이스를 확인하고 동일하게 맞추기
        device = self.class_cache.device
        classes_to_exclude_tensor = torch.tensor(classes_to_exclude, device=device)

        # 제외할 클래스가 없는 데이터 선택
        mask = ~torch.isin(self.class_cache, classes_to_exclude_tensor)

        # Filter the class_cache, img_cache, and t_cache using the mask
        self.class_cache = self.class_cache[mask]
        self.img_cache = self.img_cache[mask]
        self.t_cache = self.t_cache[mask]
        
        # 전체 길이에서 10%에 해당하는 수를 계산
        num_elements = len(self.class_cache)
        num_to_modify = int(0.1 * num_elements)  # 10%의 수

        # 랜덤으로 num_to_modify 개의 인덱스를 선택
        indices_to_modify = torch.randperm(num_elements)[:num_to_modify]

        # 해당 인덱스의 값을 10으로 변경
        self.class_cache[indices_to_modify] = 10
        
    def __len__(self):
        return len(self.img_cache)

    def __getitem__(self, idx):
        img = self.img_cache[idx]
        t = self.t_cache[idx]
        label = self.class_cache[idx]

        if self.cond_sharing:
            t_value = t.item()  # t는 텐서이므로 스칼라 값으로 변환
            p = math.exp(-self.lambda_param * (1 - t_value / self.n_T))
            if torch.rand(1).item() < p:
                new_label = random.choice(self.total_class)
                label = torch.tensor(new_label, dtype=label.dtype, device=self.class_cache.device)

        return img, t, label, idx
    
    def update_data(self, indices, new_imgs):
        # 이 부분을 고쳐서 `indices`를 정수 배열 형태로 바꿔 인덱싱
        indices = indices.view(-1).long()  # ensure indices are a flat, long tensor
        
        device = self.img_cache.device
        indices = indices.to(device)
        new_imgs = new_imgs.to(device)
        
        # print('self.img_cache device:', self.img_cache.device)
        # print('indices device:', indices.device)
        # print('new_imgs device:', new_imgs.device)
        
        # 인덱스가 맞지 않는 경우 torch.index_select로 인덱스를 처리
        self.img_cache.index_copy_(0, indices, new_imgs)  # indices에 맞는 부분만 교체
        self.t_cache.index_copy_(0, indices, self.t_cache[indices] - 1)
        
        # t_cache 값이 0 미만인 인덱스 처리
        negative_indices = (self.t_cache[indices] == 0).nonzero(as_tuple=True)[0]
        
        # 실제 zero_indices를 전체 t_cache 기준으로 변환
        zero_indices = indices[negative_indices]
        num_zero_indices = zero_indices.size(0)

        if num_zero_indices > 0:
            # 0인 인덱스를 T-1로 초기화
            self.t_cache.index_fill_(0, zero_indices, self.n_T)
            self.img_cache.index_copy_(0, zero_indices, torch.randn(
                num_zero_indices, 
                new_imgs.shape[1],  # channels
                new_imgs.shape[2],  # height
                new_imgs.shape[3],  # width
                device=device 
            ))
            

    def check_cache(self, step):
        visualize_t_cache_distribution(self.t_cache, 100)
        # 선택할 타임스텝 목록
        specific_t_values = [1, 101, 201, 301, 400]
        selected_images_list = []

        for t_value in specific_t_values:
            # 해당 타임스텝 값이 있는 인덱스를 추출
            t_specific_indices = torch.nonzero(self.t_cache == t_value, as_tuple=True)[0]

            # 해당 타임스텝에 맞는 샘플이 20개보다 적으면 그 수만큼 선택
            num_samples = min(20, len(t_specific_indices))

            # 무작위로 20개의 인덱스를 선택
            chosen_indices = torch.randperm(len(t_specific_indices))[:num_samples]

            # 선택된 인덱스를 기반으로 이미지를 추출
            selected_images = self.img_cache[t_specific_indices[chosen_indices]].cpu()  # CPU로 이동

            # 선택된 이미지를 리스트에 추가
            selected_images_list.append(selected_images)

        # 모든 타임스텝의 이미지를 하나의 텐서로 합침
        all_selected_images = torch.cat(selected_images_list, dim=0)

        # 이미지 그리드를 생성 (한 그리드에 50x10 형태로 저장, 각 타임스텝에서 20개씩 선택했으므로 총 100개의 이미지)
        grid_img = make_grid(all_selected_images, nrow=10, padding=2, normalize=True)

        # 디렉터리 존재 여부 확인 후 생성
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # 하나의 이미지 파일로 저장
        check_cache_path = os.path.join(self.cache_dir, f"check_cache_step_{step}.png")
        save_image(grid_img, check_cache_path)

        print(f"Image saved successfully as '{check_cache_path}'")