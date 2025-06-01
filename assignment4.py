import os
import timm
import torch

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

os.environ['TCL_LIBRARY'] = r'C:/Program Files/Python313/tcl/tcl8.6'

def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_img2(img1, img2, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(8, 8))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    plt.show()

def forward_wrapper(self):
    # Task 1
    # Replace the line below with your code.

    def forward(x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

    
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn_map = attn
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        #x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward


def get_attn_map_at_head_and_index(attn_map, head, index, height, width):
    # Task 2
    # Replace the line below with your code.
    #attn_map initial dimension: H, L, L
    #organise attn_map into (H,W) 

    #if head=-1, get average attn score over all attention heads
    if head == -1:
        attn = attn_map.mean(dim=0)
    else:
        attn = attn_map[head]
    #get all attention scores, excluding 0th elemtent [CLS token], and reshape to heigh and width
    return attn[index, 1:1+height*width].view(height, width)

if __name__ == "__main__":

    # Test code
    img = Image.open('dog.jpg')

    model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.blocks[-1].attn.forward = forward_wrapper(model.blocks[-1].attn)
    x = transforms(img)
    with torch.no_grad():
        model(x.unsqueeze(0))
    attn_map = model.blocks[-1].attn.attn_map.detach().squeeze(0)
    show_img(img)
    for i in range(attn_map.shape[0]):
        attn_map_i = get_attn_map_at_head_and_index(attn_map, i, 0, 14, 14)
        show_img(attn_map_i)
        show_img2(x.permute(1, 2, 0) * torch.as_tensor([0.229, 0.224, 0.225]) + torch.as_tensor([0.485, 0.456, 0.406]), attn_map_i.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1))
        show_img(attn_map[i])
        show_img(attn_map[i, 1:, 1:].reshape(14, 14, 14, 14).permute(0, 2, 1, 3).reshape(196, 196))
    attn_map_mean = get_attn_map_at_head_and_index(attn_map, -1, 0, 14, 14)
    show_img(attn_map_mean)
    show_img2(x.permute(1, 2, 0) * torch.as_tensor([0.229, 0.224, 0.225]) + torch.as_tensor([0.485, 0.456, 0.406]), attn_map_mean.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1))
    show_img(attn_map.mean(0))
    show_img(attn_map.mean(0)[1:, 1:].reshape(14, 14, 14, 14).permute(0, 2, 1, 3).reshape(196, 196))

    model = timm.create_model('vit_small_patch8_224.dino', pretrained=True).eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    model.blocks[-1].attn.forward = forward_wrapper(model.blocks[-1].attn)
    x = transforms(img)
    with torch.no_grad():
        model(x.unsqueeze(0))
    attn_map = model.blocks[-1].attn.attn_map.detach().squeeze(0)
    show_img(img)
    for i in range(attn_map.shape[0]):
        attn_map_i = get_attn_map_at_head_and_index(attn_map, i, 0, 28, 28)
        show_img(attn_map_i)
        show_img2(x.permute(1, 2, 0) * torch.as_tensor([0.229, 0.224, 0.225]) + torch.as_tensor([0.485, 0.456, 0.406]), attn_map_i.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1))
        show_img(attn_map[i])
        show_img(attn_map[i, 1:, 1:].reshape(28, 28, 28, 28).permute(0, 2, 1, 3).reshape(784, 784))
    attn_map_mean = get_attn_map_at_head_and_index(attn_map, -1, 0, 28, 28)
    show_img(attn_map_mean)
    show_img2(x.permute(1, 2, 0) * torch.as_tensor([0.229, 0.224, 0.225]) + torch.as_tensor([0.485, 0.456, 0.406]), attn_map_mean.repeat_interleave(8, dim=0).repeat_interleave(8, dim=1))
    show_img(attn_map.mean(0))
    show_img(attn_map.mean(0)[1:, 1:].reshape(28, 28, 28, 28).permute(0, 2, 1, 3).reshape(784, 784))
