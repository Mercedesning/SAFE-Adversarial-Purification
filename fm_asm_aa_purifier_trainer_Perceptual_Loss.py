# 此文件加入了感知损失来提升图像质量
import math
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchvision.utils import save_image
import os
import math
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchvision.utils import save_image
import numpy as np

# ==========================================
# 1. 新增：感知损失 (Perceptual Loss) 类
# ==========================================
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        # 使用预训练的 VGG19 提取特征
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4]).to(device).eval()   # relu1_2
        self.slice2 = nn.Sequential(*vgg[4:9]).to(device).eval()  # relu2_2
        self.slice3 = nn.Sequential(*vgg[9:18]).to(device).eval() # relu3_4
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # VGG 期望的输入是 ImageNet 归一化格式，这里做简单的缩放
        x_f1 = self.slice1(x)
        y_f1 = self.slice1(y)
        x_f2 = self.slice2(x_f1)
        y_f2 = self.slice2(y_f1)
        x_f3 = self.slice3(x_f2)
        y_f3 = self.slice3(y_f2)
        
        loss = F.l1_loss(x_f1, y_f1) + F.l1_loss(x_f2, y_f2) + F.l1_loss(x_f3, y_f3)
        return loss
# ==========================================
# 2. EMA (指数移动平均) 核心类
# ==========================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

# ==========================================
# 3. 核心网络组件
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, h * w), qkv)
        sim = torch.einsum('b h d i, b h d j -> b h i j', q * self.scale, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        out = out.reshape(b, -1, h, w).contiguous()
        return self.to_out(out) + x

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base_ch=64):
        super().__init__()
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_ch),
            nn.Linear(base_ch, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        # Encoder
        self.inc = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResnetBlock(base_ch, base_ch, time_dim)
        self.down2 = nn.ModuleList([nn.MaxPool2d(2), ResnetBlock(base_ch, base_ch*2, time_dim)])
        self.down3 = nn.ModuleList([nn.MaxPool2d(2), ResnetBlock(base_ch*2, base_ch*4, time_dim)])
        self.down4 = nn.ModuleList([nn.MaxPool2d(2), ResnetBlock(base_ch*4, base_ch*8, time_dim)])
        # Bottleneck
        self.bot1 = ResnetBlock(base_ch*8, base_ch*16, time_dim)
        self.attn = Attention(base_ch*16)
        self.bot2 = ResnetBlock(base_ch*16, base_ch*16, time_dim)
        # Decoder
        self.up1 = nn.ModuleList([nn.Upsample(scale_factor=2), ResnetBlock(base_ch*16 + base_ch*8, base_ch*8, time_dim)])
        self.up2 = nn.ModuleList([nn.Upsample(scale_factor=2), ResnetBlock(base_ch*8 + base_ch*4, base_ch*4, time_dim)])
        self.up3 = nn.ModuleList([nn.Upsample(scale_factor=2), ResnetBlock(base_ch*4 + base_ch*2, base_ch*2, time_dim)])
        self.up4 = nn.ModuleList([nn.Upsample(scale_factor=2), ResnetBlock(base_ch*2 + base_ch, base_ch, time_dim)])
        self.outc = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        x1 = self.inc(x)
        x1 = self.down1(x1, t_emb)
        x2 = self.down2[1](self.down2[0](x1), t_emb)
        x3 = self.down3[1](self.down3[0](x2), t_emb)
        x4 = self.down4[1](self.down4[0](x3), t_emb)
        
        # 使用下采样进入瓶颈层，修正原代码索引问题
        xb = F.max_pool2d(x4, 2)
        xb = self.bot1(xb, t_emb)
        xb = self.attn(xb)
        xb = self.bot2(xb, t_emb)
        
        u1 = self.up1[1](torch.cat([self.up1[0](xb), x4], 1), t_emb)
        u2 = self.up2[1](torch.cat([self.up2[0](u1), x3], 1), t_emb)
        u3 = self.up3[1](torch.cat([self.up3[0](u2), x2], 1), t_emb)
        u4 = self.up4[1](torch.cat([self.up4[0](u3), x1], 1), t_emb)
        return self.outc(u4)

# ==========================================
# 4. 数据集逻辑
# ==========================================
class AdvSaliencyDataset(Dataset):
    def __init__(self, adv_dir, sal_dir, clean_dir, image_size=640):
        self.image_size = image_size
        self.data_items = []
        
        # 1. 获取所有 ASM 文件 (.npy)
        # 注意：AA 版本保存的是 {id}.npy，不再是 {id}_asm.npy
        sal_files = sorted(glob.glob(os.path.join(sal_dir, '*.npy')))
        
        print(f"[*] 扫描 ASM 目录: {sal_dir}，找到 {len(sal_files)} 个文件")

        for s_p in sal_files:
            # 提取文件 ID (例如 '0001')
            img_id = os.path.splitext(os.path.basename(s_p))[0]
            
            # 根据 AA 脚本的保存逻辑：
            # 对抗图路径: {id}.png
            # 干净图路径: {id}.jpg (或者是 .png，取决于你原始 COCO 的后缀)
            adv_p = os.path.join(adv_dir, f"{img_id}.png")
            
            # 兼容性处理：尝试匹配 .jpg 或 .png 的干净图
            cln_p_jpg = os.path.join(clean_dir, f"{img_id}.jpg")
            cln_p_png = os.path.join(clean_dir, f"{img_id}.png")
            cln_p = cln_p_jpg if os.path.exists(cln_p_jpg) else cln_p_png
            
            if os.path.exists(adv_p) and os.path.exists(cln_p):
                self.data_items.append({
                    'adv': adv_p, 
                    'cln': cln_p, 
                    'sal': s_p, 
                    'name': f"{img_id}.png"
                })
        
        print(f"[*] 成功匹配对齐的样本总数: {len(self.data_items)}")
        
        if len(self.data_items) == 0:
            # 最后的兜底检查，帮助你定位目录问题
            print(f"[!] 错误: 未能在目录下找到匹配的文件对！")
            print(f"    检查示例路径: \n    ADV: {adv_p}\n    CLN: {cln_p}\n    ASM: {s_p}")

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): 
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        x_adv = self.img_tf(Image.open(item['adv']).convert("RGB"))
        x_cln = self.img_tf(Image.open(item['cln']).convert("RGB"))
        
        # 加载 ASM
        sal = torch.from_numpy(np.load(item['sal']).astype(np.float32))
        
        # 确保 sal 是 [1, H, W]
        if len(sal.shape) == 2:
            sal = sal.unsqueeze(0)
        
        # 确保 ASM 尺寸匹配图片
        if sal.shape[-2:] != (self.image_size, self.image_size):
            sal = F.interpolate(sal.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)
            
        return x_adv, sal.clamp(0, 1), x_cln, item['name']

# ==========================================
# 5. 训练与测试函数
# ==========================================
def train(args):
    device = torch.device(args.device)
    ds = AdvSaliencyDataset(args.adv_dir, args.sal_dir, args.clean_dir, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = AttentionUNet(in_ch=4, out_ch=3, base_ch=args.base_ch).to(device)
    
    # 支持断点续训 (非常建议，因为增加 Perceptual Loss 后可以基于旧权重微调)
    if args.resume:
        print(f"[*] Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    FM = ConditionalFlowMatcher(sigma=0.0)
    scaler = GradScaler()
    
    # 初始化感知损失
    if args.lambda_perceptual > 0:
        perceptual_fn = PerceptualLoss(device)
        print(f"[*] Perceptual Loss enabled with weight: {args.lambda_perceptual}")

    print(f"[*] Training on {args.device}...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_loss_fm = 0
        total_loss_per = 0
        
        for i, (x_adv, sal, x_cln, _) in enumerate(dl):
            x_adv, sal, x_cln = x_adv.to(device), sal.to(device), x_cln.to(device)
            
            # 这里的训练策略改为：不仅学习从 x_adv -> x_cln，
            # 同时也学习从“微小扰动的 x_cln” -> x_cln (这能显著提升画质)
            mix = torch.rand(x_adv.size(0), 1, 1, 1, device=device)
            x_start = mix * x_adv + (1 - mix) * x_cln

            with autocast():
                # 1. 计算标准 Flow Matching Loss (MSE on velocity)
                t, xt, ut = FM.sample_location_and_conditional_flow(x_start, x_cln)
                vt_pred = model(torch.cat([xt, sal], dim=1), t)
                
                # ASM 加权 MSE
                loss_fm = torch.mean((1.0 + sal * args.asm_boost) * (vt_pred - ut)**2)

                # 2. 计算预测的“最终结果”以进行感知损失监督
                # 在 CFM 中，xt = (1-t)x0 + t*x1, ut = x1 - x0
                # 我们预测的 x1_hat = xt + (1-t) * vt_pred
                if args.lambda_perceptual > 0:
                    # 只有在 t 比较大的时候计算感知损失才有意义，或者对全时段计算
                    x1_hat = xt + (1.0 - t[:, None, None, None]) * vt_pred
                    loss_per = perceptual_fn(x1_hat, x_cln)
                else:
                    loss_per = torch.tensor(0.0).to(device)

                # 联合损失
                loss = (loss_fm + args.lambda_perceptual * loss_per) / args.accum_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
            
            total_loss += loss.item() * args.accum_steps
            total_loss_fm += loss_fm.item()
            total_loss_per += loss_per.item()

        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch} | Total: {avg_loss:.6f} | FM: {total_loss_fm/len(dl):.6f} | Per: {total_loss_per/len(dl):.6f}")
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            ema.apply_shadow()
            save_name = f"fm_vgg_epoch_{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
            torch.save(model.state_dict(), os.path.join(args.output_dir, "fm_purifier_last.pth"))
            ema.restore()
# 一阶Euler           
@torch.no_grad()
def run_test(args):
    device = torch.device(args.device)
    
    # 1. 模型初始化
    model = AttentionUNet(in_ch=4, out_ch=3, base_ch=args.base_ch).to(device)
    
    # 2. 指向你微调后的权重路径
    # 如果你想测试特定 epoch（比如 40），可以改为 f"fm_vgg_epoch_40.pth"
    ckpt_path = os.path.join(args.output_dir, "fm_purifier_last.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"[!] 警告: 未找到权重文件 {ckpt_path}，请检查路径！")
        return

    print(f"[*] 正在加载微调权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. 数据加载
    ds = AdvSaliencyDataset(args.adv_dir, args.sal_dir, args.clean_dir, args.image_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 4. 自动创建输出目录
    # 如果开启 test_clean，存入 purified_clean；否则存入 purified_adv
    sub_dir = "purified_clean" if args.test_clean else "purified_adv"
    save_path = os.path.join(args.output_dir, sub_dir)
    os.makedirs(save_path, exist_ok=True)

    dt = 1.0 / args.sampling_steps
    print(f"[*] 模式: {'测试干净图净化 (上限)' if args.test_clean else '测试对抗图净化 (防御)'}")
    print(f"[*] 采样步数: {args.sampling_steps} | 保存位置: {save_path}")

    for i, (x_adv, sal, x_cln, file_name) in enumerate(dl):
        x_adv, sal, x_cln = x_adv.to(device), sal.to(device), x_cln.to(device)
        
        # 核心逻辑：决定是从干净图开始还是从对抗图开始
        curr_x = x_cln.clone() if args.test_clean else x_adv.clone()
        
        # ODE 积分演化
        for s in range(args.sampling_steps):
            t = torch.ones(curr_x.size(0), device=device) * (s / args.sampling_steps)
            v = model(torch.cat([curr_x, sal], dim=1), t)
            curr_x = curr_x + v * dt
        
        # 反归一化并保存图片 (去除任何对齐操作，保持纯净重构)
        res = (curr_x.clamp(-1, 1) + 1.0) / 2.0
        save_image(res, os.path.join(save_path, file_name[0]))
        
        if i % 100 == 0: 
            print(f"进度: {i}/{len(dl)}")

    print(f"[√] 净化完成！请前往 {save_path} 进行 YOLO 评估。")

# 二阶Heun
@torch.no_grad()
def run_test_advanced(args):
    device = torch.device(args.device)
    model = AttentionUNet(in_ch=4, out_ch=3, base_ch=args.base_ch).to(device)
    
    ckpt_path = os.path.join(args.output_dir, "fm_purifier_last.pth")
    if not os.path.exists(ckpt_path): return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ds = AdvSaliencyDataset(args.adv_dir, args.sal_dir, args.clean_dir, args.image_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    save_path = os.path.join(args.output_dir, "purified_adv_heun_t09")
    os.makedirs(save_path, exist_ok=True)

    # ==========================================
    # 核心超参数：Heun + Early Stopping
    # ==========================================
    sampling_steps = args.sampling_steps
    stop_time = 0.9  # 停止时间，建议 0.9 或 0.95
    dt = stop_time / sampling_steps 
    
    print(f"[*] 使用 Heun 求解器 | Stop-time: {stop_time} | Steps: {sampling_steps}")

    for i, (x_adv, sal, x_cln, file_name) in enumerate(dl):
        x_adv, sal, x_cln = x_adv.to(device), sal.to(device), x_cln.to(device)
        curr_x = x_adv.clone()
        
        for s in range(sampling_steps):
            t_curr = (s / sampling_steps) * stop_time
            t_vec = torch.ones(curr_x.size(0), device=device) * t_curr
            
            # --- Heun 步骤 ---
            # 1. 预测 (Predictor)
            v_pred = model(torch.cat([curr_x, sal], dim=1), t_vec)
            x_next_euler = curr_x + v_pred * dt
            
            # 2. 校正 (Corrector)
            t_next_vec = torch.ones(curr_x.size(0), device=device) * (t_curr + dt)
            v_next = model(torch.cat([x_next_euler, sal], dim=1), t_next_vec)
            
            # 3. 更新 (平均速度)
            curr_x = curr_x + 0.5 * (v_pred + v_next) * dt
        
        # 反归一化并保存
        res = (curr_x.clamp(-1, 1) + 1.0) / 2.0
        save_image(res, os.path.join(save_path, file_name[0]))
        
        if i % 100 == 0: print(f"进度: {i}/{len(dl)}")

# ==========================================
# 5. 入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--adv_dir", type=str, required=True)
    parser.add_argument("--sal_dir", type=str, required=True)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5) # 引入 VGG 后建议调小学习率
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--asm_boost", type=float, default=5.0)
    parser.add_argument("--sampling_steps", type=int, default=50) # 测试时建议步数多一点，比如 50
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--output_dir", type=str, default="./fm_vgg_out")
    parser.add_argument("--device", type=str, default="cuda:1")
    
    # 新增核心参数
    parser.add_argument("--test_clean", action="store_true", help="是否测试对干净图像的重构效果")
    parser.add_argument("--lambda_perceptual", type=float, default=0.01, help="感知损失权重，建议 0.01 ~ 0.05")
    parser.add_argument("--resume", type=str, default=None, help="加载之前的权重进行微调")
    parser.add_argument('--use_heun', action='store_true', help='使用二阶 Heun 求解器')
    parser.add_argument('--stop_time', type=float, default=1.0, help='ODE 演化的停止时间 (0.0-1.0)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "train": train(args)
    else: run_test(args)

# yolov8开始训练感知损失权重代码：
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode train \
#   --resume /root/wangchao/FlowPure/fm_aa_out/fm_purifier_last.pth \
#   --lambda_perceptual 0.02 \
#   --lr 5e-5 \
#   --asm_boost 8.0 \
#   --output_dir /root/wangchao/FlowPure/fm_vgg_finetune \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_aa_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_aa_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --epochs 50 \
#     > fm_vgg_finetune.log 2>&1 &


# yolov8净化干净图代码
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --test_clean \
#   --sampling_steps 50 \
#   --output_dir /root/wangchao/FlowPure/fm_vgg_finetune \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_aa_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_aa_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --device cuda:1 \
#     > fm_vgg_finetune_clean.log 2>&1 &

# yolov8净化adv图代码
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --sampling_steps 50 \
#   --output_dir /root/wangchao/FlowPure/fm_vgg_finetune \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_aa_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_aa_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --device cuda:1 \
#    > fm_vgg_finetune_adv.log 2>&1 &

# 训练适配 RT-DETR 权重代码：
# 建议在 YOLO 权重的基础上进行微调 (resume)，这样收敛极快
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode train \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_rtdetr_finetune \
#   --resume /root/wangchao/FlowPure/fm_vgg_finetune/fm_purifier_last.pth \
#   --epochs 50 \
#   --lambda_perceptual 0.02 \
#   --device cuda:1 \
#     > fm_rtdetr_finetune.log 2>&1 &

# rt-detr训练200（预训练50轮+150轮）轮代码：
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode train \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_rtdetr_finetune \
#   --resume /root/wangchao/FlowPure/fm_rtdetr_finetune/fm_purifier_last.pth \
#   --epochs 150 \
#   --lambda_perceptual 0.02 \
#   --device cuda:1 \
#     > fm_rtdetr_finetune.log 2>&1 &

# 使用RT-DETR净化权重对RT-DETR对抗样本进行净化
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_rtdetr_finetune \
#   --sampling_steps 50 \
#   --device cuda:1 \
#     > fm_rtdetr_finetune_adv.log 2>&1 &

# 使用二阶Heun来进行净化对抗样本
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_rtdetr_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_rtdetr_finetune \
#   --sampling_steps 50 \
#   --use_heun \
#   --stop_time 0.9 \
#   --device cuda:1 \
#     > fm_rtdetr_heun_t09.log 2>&1 &


# 训练fastercnn净化权重
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode train \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_fasterrcnn_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_fasterrcnn_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_fasterrcnn_finetune \
#   --epochs 200 \
#   --lambda_perceptual 0.02 \
#   --device cuda:1 \
#     > fm_fasterrcnn_finetune.log 2>&1 &

# 使用fasterrcnn对fasterrcnn对抗样本进行净化
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_fasterrcnn_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_fasterrcnn_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_fasterrcnn_finetune \
#   --sampling_steps 50 \
#   --device cuda:1 \
#     > fm_fasterrcnn_finetune_adv.log 2>&1 &

# 训练yolov26净化权重
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode train \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_yolov26_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_yolov26_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_yolov26_finetune \
#   --epochs 200 \
#   --lambda_perceptual 0.02 \
#   --device cuda:0 \
#     > fm_yolov26_finetune.log 2>&1 &

# 使用yolov26对yolov26对抗样本进行净化
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_yolov26_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_yolov26_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_yolov26_finetune \
#   --sampling_steps 50 \
#   --device cuda:0 \
#     > fm_yolov26_finetune_adv.log 2>&1 &

# 训练DINOv2净化权重
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#     --mode train \
#     --adv_dir /root/wangchao/FlowPure/datasets/coco_dinov2_output/adv_images \
#     --sal_dir /root/wangchao/FlowPure/datasets/coco_dinov2_output/asm_npy_maps \
#     --clean_dir /root/wangchao/FlowPure/datasets/coco \
#     --batch_size 2 \
#     --accum_steps 8 \
#     --lr 2e-5 \
#     --epochs 200 \
#     --asm_boost 10.0 \
#     --lambda_perceptual 0.05 \
#     --output_dir ./fm_dinov2_finetune \
#     --device cuda:1 \
#     > fm_dinov2_finetune.log 2>&1 &

# 使用DINOv2对DINOv2对抗样本进行净化
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode test \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_dinov2_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_dinov2_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_dinov2_finetune \
#   --sampling_steps 50 \
#   --device cuda:1 \
#   > fm_dinov2_finetune_adv.log 2>&1 &

# 训练mask2former净化权重
# nohup python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
#   --mode train \
#   --adv_dir /root/wangchao/FlowPure/datasets/coco_mask2former_output/adv_images \
#   --sal_dir /root/wangchao/FlowPure/datasets/coco_mask2former_output/asm_npy_maps \
#   --clean_dir /root/wangchao/FlowPure/datasets/coco \
#   --output_dir ./fm_mask2former_finetune \
#   --image_size 640 \
#   --batch_size 2 \
#   --accum_steps 8 \
#   --lr 2e-5 \
#   --asm_boost 10.0 \
#   --lambda_perceptual 0.05 \
#   --epochs 200 \
#   --device cuda:1 \
#   > train_mask2former.log 2>&1 &