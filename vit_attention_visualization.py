import os
import timm
import timm.data
import torch

from PIL import Image

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

os.environ['TCL_LIBRARY'] = r'C:/Program Files/Python313/tcl/tcl8.6'

def show_img(img):
    """
    Displays a single image
    """
    img = np.asarray(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_img2(img1, img2, alpha=0.8):
    """
    Displays base image with translucent overlay
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(8, 8))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    plt.show()

def forward_wrapper(self):
    """
    Wraps attention module to capture attention maps during forward pass
    """
    def forward(x):
        B, N, C = x.shape
        #generate qkv projections and permute to [3,B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #apply normalizations on q and k
        q, k = self.q_norm(q), self.k_norm(k)
        #compute attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #save the attention map for visualizations
        self.attn_map = attn
        #apply the attention scores to values
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward


def get_attn_map_at_head_and_index(attn_map, head, index, height, width):
    """
    Reshape attention map for specific head and token indexes
    """
    #attn_map initial dimension: H, L, L
    #organise attn_map into (H,W) 

    #if head=-1, get average attn score over all attention heads
    if head == -1:
        attn = attn_map.mean(dim=0)
    else:
        attn = attn_map[head]
    #get all attention scores, excluding 0th elemtent [CLS token], and reshape to heigh and width
    return attn[index, 1:1+height*width].view(height, width)

def denormalize_image(tensor, mean, std):
    """
    Reverse normalization on image tensor
    """
    #create a copy to preserve original tensor
    tensor = tensor.clone()
    #reverse normalization per channel
    for t, m, s in zip(tensor, mean, std):
        # (normalization*std) + mean
        t.mul_(s).add(m)
    return tensor.permute(1,2,0).numpy()

def visualize_attention(model_name, img_path):
    """
    attention visualization for the model provided
    """
    #load model
    model = timm.create_model(model_name, pretrained=True).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    #modify the last attnetion block to capture maps
    model.blocks[-1].attn.forward = forward_wrapper(model.blocks[-1].attn)
    #load and transform image
    img = Image.open(img_path)
    x = transforms(img)
    #forward pass to generate attention maps
    with torch.no_grad():
        model(x.unsqueeze(0))
    #get attnetion maps from the last layer
    attn_map = model.blocks[-1].attn.attn_map.detach().squeeze(0)
    #calculate dimensions based on patch size
    num_tokens = attn_map.shape[1]
    #exclude the CLS token
    grid_size = int((num_tokens-1) **0.5) 
    patch_size = data_config['input_size'][1] // grid_size

    #show original image
    print("\n" + "="*50)
    print(f"Visualizing: {model_name} | Grid: {grid_size}x{grid_size} | Patch: {patch_size}px")
    print("="*50)
    show_img(img)

    #visualize parameters
    denorm_mean = [0.485, 0.456, 0.406]
    denorm_std = [0.229, 0.224, 0.225]
    denorm_img = denormalize_image(x, denorm_mean, denorm_std)
    #visualize each attention head
    for i in range(attn_map.shape[0]):
        print(f"\nHead {i+1}/{attn_map.shape[0]} attention:")
        #get attention for CLS token
        attn_heatmap = get_attn_map_at_head_and_index(attn_map, i, 0, grid_size, grid_size)
        #show raw heatmap
        show_img(attn_heatmap)
        #show overlay
        upsampled_heatmap = attn_heatmap.repeat_interleave(patch_size, 0)
        upsampled_heatmap = upsampled_heatmap.repeat_interleave(patch_size, 1)
        show_img2(denorm_img, upsampled_heatmap, alpha=0.7)

    #visualize average attention across heads
    print("\nAverage attention across heads:")
    avg_attn = get_attn_map_at_head_and_index(attn_map, -1, 0, grid_size, grid_size)
    show_img(avg_attn)
    
    upsampled_avg = avg_attn.repeat_interleave(patch_size, 0)
    upsampled_avg = upsampled_avg.repeat_interleave(patch_size, 1)
    show_img2(denorm_img, upsampled_avg, alpha=0.7)

if __name__ == "__main__":
    # Configuration
    img_pth = 'dog.jpg'  # Update with your image path
    models = [
        'vit_small_patch16_224.dino',
        'vit_small_patch8_224.dino'
    ]

    # Run visualization for each model
    for model_name in models:
        visualize_attention(model_name, img_pth)
