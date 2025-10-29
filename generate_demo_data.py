#!/usr/bin/env python3
"""
Generate realistic training log files for testing MCP server
"""

import pandas as pd
import numpy as np
import os

# Create demo_data directory
os.makedirs('demo_data', exist_ok=True)


def generate_perfect_training():
    """Smooth convergence, good generalization"""
    print("Generating perfect_training.csv...")
    epochs = 50
    
    # Smooth learning curves
    train_acc = np.linspace(0.5, 0.94, epochs) + np.random.normal(0, 0.01, epochs)
    val_acc = np.linspace(0.5, 0.92, epochs) + np.random.normal(0, 0.015, epochs)
    
    df = pd.DataFrame({
        'epoch': range(epochs),
        'train_loss': 1 - train_acc + np.random.normal(0, 0.02, epochs),
        'train_accuracy': train_acc,
        'val_loss': 1 - val_acc + np.random.normal(0, 0.025, epochs),
        'val_accuracy': val_acc,
        'learning_rate': [0.001] * 30 + [0.0001] * 20
    })
    
    df.to_csv('demo_data/perfect_training.csv', index=False)
    print("✓ Created perfect_training.csv")


def generate_overfitting():
    """Train accuracy high, validation plateaus"""
    print("Generating overfitting.csv...")
    epochs = 50
    
    # Training keeps improving, validation plateaus
    train_acc = np.linspace(0.5, 0.95, epochs) + np.random.normal(0, 0.01, epochs)
    
    # Validation improves then plateaus at 72%
    val_acc_part1 = np.linspace(0.5, 0.75, 30)
    val_acc_part2 = np.full(20, 0.72) + np.random.normal(0, 0.02, 20)
    val_acc = np.concatenate([val_acc_part1, val_acc_part2])
    
    df = pd.DataFrame({
        'epoch': range(epochs),
        'train_loss': 1 - train_acc + np.random.normal(0, 0.01, epochs),
        'train_accuracy': train_acc,
        'val_loss': 1 - val_acc + np.random.normal(0, 0.03, epochs),
        'val_accuracy': val_acc,
        'learning_rate': [0.001] * epochs
    })
    
    df.to_csv('demo_data/overfitting.csv', index=False)
    print("✓ Created overfitting.csv")


def generate_divergence():
    """Loss explodes at epoch 45"""
    print("Generating divergence.csv...")
    epochs = 60
    
    # Normal training until epoch 45, then explosion
    train_acc = np.linspace(0.5, 0.85, 45).tolist() + [0.3, 0.15, 0.1, 0.05, 0.02] + [0.01] * 10
    val_acc = np.linspace(0.5, 0.80, 45).tolist() + [0.25, 0.12, 0.08, 0.04, 0.02] + [0.01] * 10
    
    train_loss = (1 - np.array(train_acc[:45])).tolist() + [2.5, 5.0, 8.0, 12.0, 15.0] + [20.0] * 10
    val_loss = (1 - np.array(val_acc[:45])).tolist() + [3.0, 6.0, 9.0, 13.0, 16.0] + [22.0] * 10
    
    df = pd.DataFrame({
        'epoch': range(epochs),
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'learning_rate': [0.001] * epochs
    })
    
    df.to_csv('demo_data/divergence.csv', index=False)
    print("✓ Created divergence.csv")


def generate_slow_convergence():
    """Learning too slow"""
    print("Generating slow_convergence.csv...")
    epochs = 100
    
    # Very slow improvement
    train_acc = np.linspace(0.5, 0.68, epochs) + np.random.normal(0, 0.01, epochs)
    val_acc = np.linspace(0.5, 0.65, epochs) + np.random.normal(0, 0.015, epochs)
    
    df = pd.DataFrame({
        'epoch': range(epochs),
        'train_loss': 1 - train_acc + np.random.normal(0, 0.02, epochs),
        'train_accuracy': train_acc,
        'val_loss': 1 - val_acc + np.random.normal(0, 0.025, epochs),
        'val_accuracy': val_acc,
        'learning_rate': [0.0001] * epochs  # LR too low
    })
    
    df.to_csv('demo_data/slow_convergence.csv', index=False)
    print("✓ Created slow_convergence.csv")


def generate_json_log():
    """Create a JSON format log"""
    print("Generating training_log.json...")
    import json
    
    data = {
        "model": "ResNet50",
        "dataset": "ImageNet",
        "batch_size": 32,
        "initial_lr": 0.001,
        "epochs": [
            {
                "epoch": 1,
                "train_loss": 0.65,
                "train_accuracy": 0.55,
                "val_loss": 0.62,
                "val_accuracy": 0.58,
                "learning_rate": 0.001
            },
            {
                "epoch": 2,
                "train_loss": 0.45,
                "train_accuracy": 0.75,
                "val_loss": 0.48,
                "val_accuracy": 0.72,
                "learning_rate": 0.001
            },
            {
                "epoch": 3,
                "train_loss": 0.35,
                "train_accuracy": 0.85,
                "val_loss": 0.40,
                "val_accuracy": 0.80,
                "learning_rate": 0.001
            }
        ]
    }
    
    with open('demo_data/training_log.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✓ Created training_log.json")


def generate_yaml_config():
    """Create a YAML config file"""
    print("Generating model_config.yaml...")
    
    yaml_content = """
model:
  name: ResNet50
  architecture:
    layers: 50
    dropout: 0.5
  
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: Adam
  
dataset:
  name: CIFAR-10
  train_size: 50000
  val_size: 10000
  
hyperparameters:
  weight_decay: 0.0001
  momentum: 0.9
  lr_schedule: step_decay
  step_size: 20
  gamma: 0.1
"""
    
    with open('demo_data/model_config.yaml', 'w') as f:
        f.write(yaml_content.strip())
    
    print("✓ Created model_config.yaml")


def print_summary():
    """Print summary of generated files"""
    print("\n" + "="*50)
    print("Demo Data Generated Successfully!")
    print("="*50)
    print("\nGenerated files in demo_data/:")
    print("  1. perfect_training.csv    - Smooth convergence")
    print("  2. overfitting.csv         - Train 95%, Val 72%")
    print("  3. divergence.csv          - Loss explosion at epoch 45")
    print("  4. slow_convergence.csv    - LR too low")
    print("  5. training_log.json       - JSON format example")
    print("  6. model_config.yaml       - YAML config example")
    print("\nTest your MCP server with:")
    print("  - demo_data/overfitting.csv")
    print("  - demo_data/divergence.csv")
    print("="*50 + "\n")


if __name__ == "__main__":
    print("\n🚀 Generating demo training logs...\n")
    
    # Generate all demo files
    generate_perfect_training()
    generate_overfitting()
    generate_divergence()
    generate_slow_convergence()
    generate_json_log()
    generate_yaml_config()
    
    # Print summary
    print_summary()