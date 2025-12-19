import torch
import torch.nn as nn
import torch.nn.functional as F # Added for interpolation
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from peft import LoraConfig, get_peft_model
from mobilesamv2 import sam_model_registry

class Args:
    encoder_type = "tiny_vit"
    encoder_path = "./weight/mobile_sam.pt"
    Prompt_guided_Mask_Decoder_path = "./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt"
    # Use forward slashes for Windows paths to avoid errors
    img_dir = "c:/sinergia_ak/SCHNELL/synthetic_rebars_ds/t/images"
    mask_dir = "c:/sinergia_ak/SCHNELL/synthetic_rebars_ds/t/labels"
    epochs = 2 
    batch_size = 2        
    learning_rate = 1e-4

args = Args()

class MaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=1024):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize to 1024x1024 for TinyViT requirements
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 128).astype(np.float32)

        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) > 0:
            bbox = np.array([np.min(x_indices), np.min(y_indices), 
                             np.max(x_indices), np.max(y_indices)], dtype=np.float32)
        else:
            bbox = np.array([0, 0, 10, 10], dtype=np.float32)

        image_tensor = torch.tensor(image).permute(2,0,1).float() / 255.0
        mask_tensor = torch.tensor(mask).float()
        bbox_tensor = torch.tensor(bbox).float()

        return image_tensor, mask_tensor, bbox_tensor

import time # Add this at the top with other imports

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Model Components
    PromptGuidedDecoder = sam_model_registry["PromptGuidedDecoder"](args.Prompt_guided_Mask_Decoder_path)
    # Load encoder using registry (handles "image_encoder." prefix stripping automatically)
    image_encoder = sam_model_registry[args.encoder_type](args.encoder_path)
    
    # 2. Apply LoRA
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["qkv"], 
        lora_dropout=0.05, 
        bias="none"
    )
    image_encoder_lora = get_peft_model(image_encoder, lora_config)
    
    model = sam_model_registry["vit_h"]()
    model.image_encoder = image_encoder_lora
    model.prompt_encoder = PromptGuidedDecoder["PromtEncoder"]
    model.mask_decoder = PromptGuidedDecoder["MaskDecoder"]
    model.to(device)

    # ==========================================
    # NEW LOGS: PARAMETER COUNT
    # ==========================================
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*30}")
    print(f"DEVICE: {device}")
    print(f"TOTAL PARAMS: {total_params:,}")
    print(f"TRAINABLE PARAMS: {trainable_params:,}")
    print(f"PERCENTAGE TRAINABLE: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*30}\n")

    optimizer = torch.optim.Adam([
        {'params': model.image_encoder.parameters()}, 
        {'params': model.mask_decoder.parameters()}
    ], lr=args.learning_rate)

    dataset = MaskDataset(args.img_dir, args.mask_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Starting LoRA training...")
    

    model.train()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        
        for batch_idx, (imgs, masks, bboxes) in enumerate(dataloader):
            batch_start_time = time.time()
            imgs, masks, bboxes = imgs.to(device), masks.to(device), bboxes.to(device)
            
            features = model.image_encoder(imgs)
            sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=bboxes, masks=None)
            
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=features,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                simple_type=True,
            )

            gt_masks_256 = F.interpolate(masks.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False).squeeze(1)

            pred = low_res_masks.squeeze(1)
            bce = nn.functional.binary_cross_entropy_with_logits(pred, gt_masks_256)
            pred_soft = torch.sigmoid(pred)
            inter = (pred_soft * gt_masks_256).sum()
            dice = 1 - (2. * inter + 1e-5) / (pred_soft.sum() + gt_masks_256.sum() + 1e-5)
            loss = bce + dice
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()

            # ==========================================
            # NEW LOGS: BATCH PROGRESS
            # ==========================================
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                batch_duration = time.time() - batch_start_time
                print(f"Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"| Loss: {loss.item():.4f} | Time/Batch: {batch_duration:.2f}s")

        epoch_duration = time.time() - epoch_start_time
        print(f"\n>>> Epoch {epoch+1} FINISHED | Avg Loss: {epoch_loss/len(dataloader):.4f} "
              f"| Total Epoch Time: {epoch_duration:.2f}s\n")

    torch.save(model.image_encoder.state_dict(), "lora_weights.pt")
    torch.save(model.mask_decoder.state_dict(), "trained_decoder.pt")
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train()