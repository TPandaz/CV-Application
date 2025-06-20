# ViT Saliency Map Visualization
This project visualizes attention mechanisms in ViTs using DINO-pretrained models. It does this by generating heatmaps showing where the model focuses on when processing images, providing interpretability for transformer-based computer vision models for educational purposes.

## Installation Process

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
    ```bash
    pip install timm torch matplotlib pillow numpy
    ```

## Usage
1. Place input image into project directory
2. Run script: 
    ```bash
    python3 vit_attention_visualization.py
    ```
The script processes images through 2 models:
- vit_small_patch16_224.dino
- vit_small_patch8_224.dino