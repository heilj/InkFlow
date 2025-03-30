import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

import matplotlib.pyplot as plt

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================== 配置参数 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
image_size = 28
channels = 1
batch_size = 256
lr = 5e-4
epochs = 100
num_classes = 10

# torch.set_default_dtype(torch.float32)

# ================== 数据加载 ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)  # [-1, 1] 归一化
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ================== 模型架构 ==================
class ConditionedDoubleConv(nn.Module):
    """带条件注入的双卷积模块"""
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels + cond_dim, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
      
    def forward(self, x, cond):
        x = F.silu(self.norm1(self.conv1(x)))
        cond = cond.expand(-1, -1, x.size(2), x.size(3))  # 动态广播条件
        x = torch.cat([x, cond], dim=1)
        return F.silu(self.norm2(self.conv2(x)))

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConditionedDoubleConv(in_channels, out_channels, cond_dim)
      
    def forward(self, x, cond):
        x = self.maxpool(x)
        return self.conv(x, cond)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConditionedDoubleConv(in_channels, out_channels, cond_dim)
      
    def forward(self, x1, x2, cond):
        x1 = self.up(x1)
        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, cond)

class ConditionalUNet(nn.Module):
    """维度安全的条件生成UNet"""
    def __init__(self):
        super().__init__()
        # 统一条件编码维度
        self.t_dim = 16
        self.label_dim = 16
        self.cond_dim = self.t_dim + self.label_dim  # 32
      
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, self.t_dim)
        )
        # 标签嵌入
        self.label_embed = nn.Embedding(num_classes, self.label_dim)
      
        # 编码路径
        self.inc = ConditionedDoubleConv(1, 64, self.cond_dim)
        self.down1 = Down(64, 128, self.cond_dim)
        self.down2 = Down(128, 256, self.cond_dim)
      
        # 解码路径
        self.up1 = Up(256 + 128, 128, self.cond_dim)  # 输入通道修正
        self.up2 = Up(128 + 64, 64, self.cond_dim)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
      
    def forward(self, x, t, labels):
        # 条件编码 (统一维度)
        t_emb = self.time_embed(t.view(-1, 1))  # [B, 16]
        lbl_emb = self.label_embed(labels)      # [B, 16]
        cond = torch.cat([t_emb, lbl_emb], dim=1)  # [B, 32]
        cond = cond.unsqueeze(-1).unsqueeze(-1)    # [B, 32, 1, 1]
      
        # 编码器
        x1 = self.inc(x, cond)
        x2 = self.down1(x1, cond)
        x3 = self.down2(x2, cond)
      
        # 解码器
        x = self.up1(x3, x2, cond)
        x = self.up2(x, x1, cond)
        return self.outc(x)

# ================== 训练与生成 ==================
model = ConditionalUNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def generate_with_label(label, num_samples=16, device=device):
    """生成指定标签的样本（修复条件维度问题）"""
    model.eval()
  
    # 初始噪声和标签
    x0 = torch.randn(num_samples, 1, 28, 28, device=device)
    labels = torch.full((num_samples,), label, device=device, dtype=torch.long)
  
    # 定义ODE函数
    def ode_func(t: torch.Tensor, x: torch.Tensor):
        t_expanded = t.expand(x.size(0))  # [1] -> [num_samples]
        vt = model(x, t_expanded, labels)
        return vt
  
    # 时间点（从0到1）
    t_eval = torch.tensor([0.0, 1.0], device=device)
  
    # 解ODE（自适应步长）

    # class ODEFunc(nn.Module):
    #     def forward(self, t, x):
    #         t_expanded = t.expand(x.size(0))  # Expand time to match batch size
    #         return model(x, t_expanded, labels)

    # # Define the ODE solver using Torchdyn's NeuralODE
    # ode_solver = NeuralODE(ODEFunc(), solver='dopri5', sensitivity='adjoint', atol=1e-6, rtol=1e-5)

    # # Define the time span for integration
    # t_span = torch.linspace(0, 1, 2, device=device, dtype=torch.float32)  # [0, 1] integration

    # # Solve the ODE
    # generated = ode_solver(x0, t_span)[-1]

    generated = odeint(
        ode_func,
        x0,
        t_eval,
        rtol=1e-5,
        atol=1e-6,
        method='dopri5',
        options={'dtype': torch.float32}
    )
    
    # 后处理
    images = (generated[-1].clamp(-1, 1) + 1) / 2  # [0,1]
    return images.cpu().squeeze(1)  # 移除通道维度

def visualize_samples(samples, title="Generated Samples"):
    """可视化生成结果"""
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_100_digits(epoch,image_size=28, device=device):
    """
    生成0-9各10张数字并绘制在10x10网格中
    Args:
        model: 训练好的生成模型
        image_size: 图像尺寸（默认MNIST为28）
        device: 计算设备
    """
    plt.figure(figsize=(8, 8))
  
    # 为每个数字0-9生成10张图
    for label in range(10):
        # 生成当前数字的10个样本
        generated = generate_with_label(
            label=label,
            num_samples=10
        ).numpy()  # 形状 (10, 28, 28)
      
        # 在当前行绘制
        for i in range(10):
            ax = plt.subplot(10, 10, i * 10 + 1 + label)
            plt.imshow(generated[i], cmap='gray')
            ax.axis('off')
            # 在每列第一行添加标签
            if i == 0:
                ax.text(14, -10, str(label), fontsize=20, ha='center')
  
    plt.tight_layout()
    plt.savefig(f"generated_digits_epoch{epoch}.png", dpi=300, bbox_inches='tight') 
    plt.close()

def train():
    """训练循环"""
    for epoch in range(epochs):
        model.train()
        total_loss = 0
      
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
          
            # 动态噪声生成
            noise = torch.randn_like(images)
            t = torch.rand(images.size(0), device=device)
            xt = (1 - t.view(-1,1,1,1)) * noise + t.view(-1,1,1,1) * images
          
            # 前向计算
            vt_pred = model(xt, t, labels)
            loss = F.mse_loss(vt_pred, images - noise)
          
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
          
            total_loss += loss.item()
      
        # 每10个epoch生成示例
        if epoch + 1 % 10 == 0:
            plot_100_digits(epoch)
        
        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train()