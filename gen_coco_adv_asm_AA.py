# yolov8生成对抗敏感度图和对抗样本 use APGD
import os
import torch
import numpy as np
from ultralytics import YOLO
from torchattacks import APGD  # 直接使用 APGD 核心，更稳定，ASM 效果更好
from tqdm import tqdm
import cv2

# --- 配置 ---
MODEL_PATH = '/root/wangchao/FlowPure/yolov8s.pt'
SOURCE_DIR = '/root/wangchao/FlowPure/datasets/coco'
SAVE_ADV_DIR = '/root/wangchao/FlowPure/datasets/coco_aa_output/adv_images'
SAVE_ASM_DIR = '/root/wangchao/FlowPure/datasets/coco_aa_output/asm_npy_maps'
DEVICE = torch.device('cuda:0') 
EPS = 8/255

os.makedirs(SAVE_ADV_DIR, exist_ok=True)
os.makedirs(SAVE_ASM_DIR, exist_ok=True)

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        yolo_instance = YOLO(model_path, task='detect')
        self.model = yolo_instance.model.to(DEVICE)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 尺寸对齐
        h, w = x.shape[2:]
        new_h, new_w = (h // 32) * 32, (w // 32) * 32
        if h != new_h or w != new_w:
            x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

        results = self.model(x)
        pred = results[0] if isinstance(results, (list, tuple)) else results
        
        # 提取类别概率 [B, 80, 8400]
        cls_scores = pred[:, 4:, :] 
        
        # 强力攻击逻辑：取所有候选点的最大置信度
        # 加入 1e-6 防止 log(0) 或梯度消失
        max_logits = cls_scores.reshape(x.shape[0], -1).max(dim=1, keepdim=True)[0] + 1e-6

        # 显式构造 [Batch, 2] 维度的逻辑值
        # 这样 torchattacks 永远不会找错 index
        out = torch.cat([max_logits, 1.0 - max_logits], dim=1)
        return out

def generate_aa_data():
    torch.cuda.empty_cache()
    wrapper = YOLOWrapper(MODEL_PATH)
    
    # 策略调整：使用 APGD (AutoAttack 的核心引擎)
    # 相比完整的 AutoAttack，它不会因为检测任务的奇怪 Logits 报 index 错误
    # 且能够保证生成高质量的对抗样本和 ASM
    atk = APGD(wrapper, eps=EPS, steps=100, eot_iter=1, n_restarts=1, loss='ce')
    
    img_paths = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png'))]
    img_paths.sort()
    
    print(f"[*] V6 工业级稳定版启动。")

    for path in tqdm(img_paths):
        img_id = os.path.splitext(os.path.basename(path))[0]
        adv_path = os.path.join(SAVE_ADV_DIR, f"{img_id}.png")
        asm_path = os.path.join(SAVE_ASM_DIR, f"{img_id}.npy")

        if os.path.exists(adv_path) and os.path.exists(asm_path):
            continue

        try:
            img_bgr = cv2.imread(path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            # labels=0 引导攻击者压低第一列（max_logits）
            labels = torch.zeros(1).long().to(DEVICE)

            # 执行攻击 (APGD 是 AutoAttack 里面最稳的)
            adv_images = atk(img_tensor, labels)

            # 生成 ASM
            diff = torch.abs(adv_images - img_tensor)
            asm_map = torch.mean(diff, dim=1).squeeze().detach().cpu().numpy()
            
            if asm_map.max() > 0:
                asm_map = (asm_map - asm_map.min()) / (asm_map.max() - asm_map.min())

            # 保存
            adv_img_np = (adv_images.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(adv_path, cv2.cvtColor(adv_img_np, cv2.COLOR_RGB2BGR))
            np.save(asm_path, asm_map)

        except Exception as e:
            print(f"\n[!] 处理失败: {e}")
            continue

if __name__ == "__main__":
    generate_aa_data()

# nohup /root/miniconda3/envs/FlowPure/bin/python -u /root/wangchao/FlowPure/asm_genetrate/gen_coco_adv_asm_AA.py > /root/wangchao/FlowPure/aa_attack_log.txt 2>&1 &