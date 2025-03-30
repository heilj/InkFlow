import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Enable MPS fallback if using Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================== Configuration ==================
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
image_size = 28
channels = 1
batch_size = 256
lr = 5e-4
epochs = 100
num_classes = 10
latent_dim = 100  # Noise vector size for the generator

# ================== Data Loading ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 2 * x - 1)  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)

# ================== Model Definitions ==================

class ConditionalUNetGenerator(nn.Module):
    """ U-Net based Generator for Conditional GAN """
    def __init__(self):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = 16
        
        # Noise + Label Embedding
        self.noise_proj = nn.Linear(self.latent_dim, 128 * 7 * 7)  # Project noise to feature map
        self.label_embed = nn.Embedding(num_classes, self.label_dim)

        # Upsampling Path (Decoder)
        self.up1 = nn.ConvTranspose2d(128 + self.label_dim, 128, kernel_size=4, stride=2, padding=1)  # Upsample to 14x14
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 28x28
        self.outc = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Final output layer

    def forward(self, noise, labels):
        # Project noise into a feature map
        x = self.noise_proj(noise).view(-1, 128, 7, 7)

        # Embed label and expand
        lbl_emb = self.label_embed(labels).unsqueeze(-1).unsqueeze(-1)  # [B, 16, 1, 1]
        lbl_emb = lbl_emb.expand(-1, -1, 7, 7)  # Match spatial size

        # Concatenate noise features and label embedding
        x = torch.cat([x, lbl_emb], dim=1)

        # Upsample
        x = F.leaky_relu(self.up1(x), 0.2)
        x = F.leaky_relu(self.up2(x), 0.2)
        return torch.tanh(self.outc(x))  # Normalize output to [-1,1]


class ConditionalDiscriminator(nn.Module):
    """ Discriminator for Conditional GAN """
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)  # Embed labels

        self.conv1 = nn.Conv2d(1 + 16, 64, kernel_size=3, stride=2, padding=1)  # Inject label
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 7 * 7, 1)  # Binary classification

    def forward(self, x, labels):
        lbl_emb = self.label_embed(labels).unsqueeze(-1).unsqueeze(-1)  # [B, 16, 1, 1]
        lbl_emb = lbl_emb.expand(-1, -1, x.shape[2], x.shape[3])  # Broadcast label
        x = torch.cat([x, lbl_emb], dim=1)  # Inject label into image
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        return torch.sigmoid(self.fc(x))  # Output probability


# ================== Initialize Models & Optimizers ==================
generator = ConditionalUNetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()

# ================== Training Loop ==================
def train_gan():
    for epoch in range(epochs):
        for real_images, labels in train_loader:
            real_images, labels = real_images.to(device), labels.to(device)
            
            # ===== Train Discriminator =====
            optimizer_D.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise, labels)
            
            # Compute Discriminator Loss
            real_loss = adversarial_loss(discriminator(real_images, labels), torch.ones(batch_size, 1, device=device))
            fake_loss = adversarial_loss(discriminator(fake_images.detach(), labels), torch.zeros(batch_size, 1, device=device))
            loss_D = (real_loss + fake_loss) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            # ===== Train Generator =====
            optimizer_G.zero_grad()
            
            fake_images = generator(noise, labels)
            loss_G = adversarial_loss(discriminator(fake_images, labels), torch.ones(batch_size, 1, device=device))
            
            loss_G.backward()
            optimizer_G.step()
            
        if epoch == 0:
            plot_100_digits(epoch)
        elif (epoch + 1) % 10 == 0:
            # visualize_samples(fake_images, title=f"Generated Samples at Epoch {epoch}")
            plot_100_digits(epoch)

        print(f"Epoch {epoch} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        

# ================== Image Generation ==================
def generate_with_label(label, num_samples=16, device=device):
    """Generate images for a specific label."""
    generator.eval()
    
    noise = torch.randn(num_samples, latent_dim, device=device)
    labels = torch.full((num_samples,), label, device=device, dtype=torch.long)
    
    with torch.no_grad():
        generated_images = generator(noise, labels)

    return (generated_images + 1) / 2  # Normalize to [0,1] for visualization

def visualize_samples(samples, title="Generated Samples"):
    """Visualize generated images."""
    plt.figure(figsize=(10, 10))
    for i in range(samples.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_100_digits(epoch, device=device):
    """Generate 10x10 grid of digits (0-9)."""
    plt.figure(figsize=(8, 8))
    for label in range(10):
        generated = generate_with_label(label, num_samples=10).cpu().numpy() # (10, 28, 28)
        generated = generated.squeeze(1)
        for i in range(10):
            ax = plt.subplot(10, 10, i * 10 + 1 + label)
            plt.imshow(generated[i], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.text(14, -10, str(label), fontsize=20, ha='center')
    plt.tight_layout()
    plt.savefig(f"./GAN_plt/generated_digits_epoch{epoch}.png", dpi=300, bbox_inches='tight') 
    plt.close()

# ================== Run Training ==================
if __name__ == "__main__":
    train_gan()