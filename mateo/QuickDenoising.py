import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torch.cuda.amp import autocast, GradScaler
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

img_size = (128, 128)
batch_size = 16
num_epochs = 100
learning_rate = 0.0001
num_noise_steps = 300
beta_min = 1e-4
beta_max = 0.02
variance_schedule = np.linspace(beta_min, beta_max, num_noise_steps)

image_dir = 'Datas/C128Sin'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_list = []
max_images = 8000

image_files = os.listdir(image_dir)

for img_file in image_files[:max_images]:
    img_path = os.path.join(image_dir, img_file)
    
    if os.path.isfile(img_path):
        image = Image.open(img_path).convert('L') 
        image = transform(image)    
        image_list.append(image)
    else:
        print(f"Skipped: {img_path} (not a file)")

image_tensor = torch.stack(image_list)
train_loader = DataLoader(image_tensor, batch_size=batch_size, shuffle=True,pin_memory=True)







def get_model_filename(base_name="NETWO"):
    version = 1
    while os.path.exists(f"{base_name}_{version}.pt"):  
        version += 1
    return f"{base_name}_{version}.pt"  



class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super(SinusoidalPositionEncoding, self).__init__()
        self.dim = dim

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t):
        return self.pe[t]

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, filter_size, num_groups, name):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups, num_channels)
        self.swish = nn.SiLU()  
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=filter_size, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=filter_size, padding=1)
        
        
        self.fc = nn.Linear(256, num_channels)  

    def forward(self, x, embedding):
        
        out = self.norm1(x)
        out = self.swish(out)
        out = self.conv1(out)
    
        
        
        
        
        if len(embedding.shape) == 1:  
            embedding = embedding.unsqueeze(0)  
        
        
        if len(embedding.shape) == 2:  
            embedding = self.fc(embedding)        
        
        embedding = embedding.unsqueeze(-1).unsqueeze(-1)
        
        
                
        
        embedding = embedding.expand(-1, -1, out.shape[2], out.shape[3])  
        
        
        out = out + embedding
        
        
        out = self.norm2(out)
        out = self.swish(out)
        out = self.conv2(out)
    
        return x + out



class AttentionBlock(nn.Module):
    def __init__(self, num_heads, num_key_channels, num_groups, name):
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_key_channels)
        self.self_attention = nn.MultiheadAttention(num_key_channels, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)  
        out, _ = self.self_attention(x_flat, x_flat, x_flat)
        out = out.permute(1, 2, 0).view(B, C, H, W)  
        return x + out  



class DiffusionUNet(nn.Module):
    def __init__(self, num_image_channels=1, initial_num_channels=64, num_groups=32, num_heads=1):
        super(DiffusionUNet, self).__init__()

        self.initial_num_channels = initial_num_channels

        
        self.conv_in = nn.Conv2d(num_image_channels, initial_num_channels, kernel_size=3, padding=1)

        
        self.res_block1 = ResidualBlock(initial_num_channels, 3, num_groups, "1")
        self.res_block2 = ResidualBlock(initial_num_channels, 3, num_groups, "2")

        
        self.downsample2 = nn.Conv2d(initial_num_channels, 2 * initial_num_channels, kernel_size=3, padding=1, stride=2)
        self.res_block3 = ResidualBlock(2 * initial_num_channels, 3, num_groups, "3")
        self.attn_block3 = AttentionBlock(num_heads, 2 * initial_num_channels, num_groups, "3")

        
        self.downsample4 = nn.Conv2d(2 * initial_num_channels, 4 * initial_num_channels, kernel_size=3, padding=1, stride=2)
        self.res_block5 = ResidualBlock(4 * initial_num_channels, 3, num_groups, "5")


        self.res_block7 = ResidualBlock(4 * initial_num_channels, 3, num_groups, "7")
        self.attn_block7 = AttentionBlock(num_heads, 4 * initial_num_channels, num_groups, "7")


        self.upsample4 = nn.ConvTranspose2d(4 * initial_num_channels, 2 * initial_num_channels, kernel_size=2, stride=2)
        self.res_block9 = ResidualBlock(2 * initial_num_channels, 3, num_groups, "9")

        self.upsample2 = nn.ConvTranspose2d(2 * initial_num_channels, initial_num_channels, kernel_size=2, stride=2)
        self.res_block11 = ResidualBlock(initial_num_channels, 3, num_groups, "11")


        self.conv_out = nn.Conv2d(initial_num_channels, num_image_channels, kernel_size=3, padding=1)


        self.position_encoding = SinusoidalPositionEncoding(4 * initial_num_channels)
        self.fc_embed = nn.Sequential(
            nn.Linear(4 * initial_num_channels, 4 * initial_num_channels),
            nn.SiLU(),
            nn.Linear(4 * initial_num_channels, 4 * initial_num_channels)
        )

    def forward(self, x, t):

        t_emb = self.position_encoding(t)
        t_emb = self.fc_embed(t_emb)


        x1 = self.conv_in(x)
        x2 = self.res_block1(x1, t_emb)
        x3 = self.res_block2(x2, t_emb)


        x4 = self.downsample2(x3)
        x5 = self.res_block3(x4, t_emb)
        x6 = self.attn_block3(x5)


        x7 = self.downsample4(x6)
        x8 = self.res_block7(x7, t_emb)
        x9 = self.attn_block7(x8)


        x10 = self.upsample4(x9)
        x11 = self.res_block9(x10 + x6, t_emb)

        x12 = self.upsample2(x11)
        x13 = self.res_block11(x12 + x3, t_emb)


        output = self.conv_out(x13 + x1)
        return output



def apply_noise_to_image(img, noise, noise_step, variance_schedule):
    alpha_bar = np.cumprod(1 - variance_schedule)
    alpha_bar_t = alpha_bar[noise_step]

    noisy_img = torch.sqrt(torch.tensor(alpha_bar_t)) * img + torch.sqrt(1 - torch.tensor(alpha_bar_t)) * noise
    return noisy_img

def model_loss(net, noisy_img, noise_step, target_noise):
    predicted_noise = net(noisy_img, noise_step)
    return nn.MSELoss()(predicted_noise, target_noise)




model_save_path = 'Models/128Sin.pth'
net = DiffusionUNet(num_image_channels=1, initial_num_channels=64, num_groups=32, num_heads=1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
do_training = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
best_loss = float('inf')
best_model_weights = None

if do_training:
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0

        for i, images in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            optimizer.zero_grad()
            noise = torch.randn_like(images)
            noise_step = torch.randint(0, num_noise_steps, (1,)).item()
            noisy_images = apply_noise_to_image(images, noise, noise_step, variance_schedule)
            loss = model_loss(net, noisy_images, noise_step, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)
        avg_loss = total_loss / len(train_loader)

        # Vérifie si la perte est la meilleure et sauvegarde le modèle
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights = net.state_dict()
            # Sauvegarde les poids du modèle
            torch.save(best_model_weights, model_save_path)
            print(f"Modèle sauvegardé avec une perte moyenne de : {avg_loss:.4f} à l'époque {epoch + 1}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")


# Si tu souhaites sauvegarder les meilleurs poids après l'entraînement complet
if best_model_weights is not None:
    torch.save(best_model_weights, model_save_path)
    print(f"Meilleur modèle chargé avec une perte de : {best_loss:.4f} et sauvegardé sous le nom : {model_save_path}")
