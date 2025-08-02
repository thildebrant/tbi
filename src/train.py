#!/usr/bin/env python3
"""
Training pipeline for TBI lesion segmentation using MR Brain Segmentation Challenge data.
"""

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import logging
import json
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

logger = logging.getLogger(__name__)


class BrainSegmentationDataset(Dataset):
    """Dataset for brain segmentation training data."""
    
    def __init__(self, data_paths: List[Dict], transform=None):
        self.data_paths = data_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_info = self.data_paths[idx]
        
        # Load multi-modal images
        t1_nii = nib.load(data_info['t1'])
        flair_nii = nib.load(data_info['flair'])
        segm_nii = nib.load(data_info['segm'])
        
        t1_data = t1_nii.get_fdata()
        flair_data = flair_nii.get_fdata()
        segm_data = segm_nii.get_fdata()
        
        # Ensure same shape by taking minimum dimensions
        min_shape = np.minimum(np.minimum(t1_data.shape, flair_data.shape), segm_data.shape)
        t1_data = t1_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        flair_data = flair_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        segm_data = segm_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Normalize intensities
        t1_data = self._normalize(t1_data)
        flair_data = self._normalize(flair_data)
        
        # Stack modalities as channels
        image = np.stack([t1_data, flair_data], axis=0)  # (2, H, W, D)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        segmentation = torch.from_numpy(segm_data).long()
        
        if self.transform:
            image, segmentation = self.transform(image, segmentation)
            
        return image, segmentation, data_info['subject_id']
    
    def _normalize(self, data):
        """Normalize image data to [0, 1]."""
        data = data.astype(np.float32)
        # Remove background
        brain_mask = data > 0
        if np.any(brain_mask):
            data_brain = data[brain_mask]
            p1, p99 = np.percentile(data_brain, [1, 99])
            data = np.clip(data, p1, p99)
            data = (data - p1) / (p99 - p1 + 1e-8)
        return data


class UNet3D(nn.Module):
    """3D U-Net for brain segmentation."""
    
    def __init__(self, in_channels=2, out_channels=11, base_filters=32):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._double_conv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._double_conv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = self._double_conv(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = self._double_conv(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._double_conv(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._double_conv(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._double_conv(base_filters * 2, base_filters)
        
        # Output
        self.final_conv = nn.Conv3d(base_filters, out_channels, 1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TBISegmentationTrainer:
    """Trainer for TBI segmentation model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = UNet3D(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            base_filters=config['base_filters']
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config['lr_step_size'], 
            gamma=config['lr_gamma']
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def load_data(self, data_dir: Path) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare training data."""
        logger.info("Loading training data...")
        
        data_paths = []
        for subject_dir in data_dir.iterdir():
            if subject_dir.is_dir():
                pre_dir = subject_dir / 'pre'
                if (pre_dir / 'T1.nii.gz').exists() and (pre_dir / 'FLAIR.nii.gz').exists():
                    data_paths.append({
                        'subject_id': subject_dir.name,
                        't1': str(pre_dir / 'T1.nii.gz'),
                        'flair': str(pre_dir / 'FLAIR.nii.gz'),
                        'segm': str(subject_dir / 'segm.nii.gz')
                    })
        
        logger.info(f"Found {len(data_paths)} subjects")
        
        # Split data
        train_paths, val_paths = train_test_split(
            data_paths, 
            test_size=self.config['val_split'], 
            random_state=42
        )
        
        # Create datasets
        train_dataset = BrainSegmentationDataset(train_paths)
        val_dataset = BrainSegmentationDataset(val_paths)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets, subject_ids) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets, subject_ids in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, data_dir: Path, output_dir: Path):
        """Main training loop."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        train_loader, val_loader = self.load_data(data_dir)
        
        best_val_loss = float('inf')
        
        logger.info("Starting training...")
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            logger.info(f'  Train Loss: {train_loss:.4f}')
            logger.info(f'  Val Loss: {val_loss:.4f}')
            logger.info(f'  Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, output_dir / 'best_model.pth')
                logger.info(f'  New best model saved!')
            
            # Save training plots
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves(output_dir)
        
        logger.info("Training completed!")
        
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, output_dir / 'final_model.pth')
        
        # Final plots
        self.plot_training_curves(output_dir)
    
    def plot_training_curves(self, output_dir: Path):
        """Plot training and validation curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TBI segmentation model')
    parser.add_argument('--data-dir', type=Path, default='data/training',
                       help='Path to training data directory')
    parser.add_argument('--output-dir', type=Path, default='models/brain_segmentation',
                       help='Output directory for trained models')
    parser.add_argument('--config-file', type=Path, default='config/training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default configuration
    config = {
        'in_channels': 2,  # T1 + FLAIR
        'out_channels': 11,  # Brain segmentation classes
        'base_filters': 32,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lr_step_size': 30,
        'lr_gamma': 0.1,
        'val_split': 0.2,
        'num_workers': 2
    }
    
    # Load config file if exists
    if args.config_file.exists():
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Save config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = TBISegmentationTrainer(config)
    
    # Start training
    trainer.train(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()