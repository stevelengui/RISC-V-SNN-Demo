

import torch
import snntorch as snn
from snntorch import functional as SF
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Hardware-Optimized SNN Model
class SNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs):
        super().__init__()
        # Linear layers with constrained dimensions
        self.fc1 = torch.nn.Linear(num_inputs, num_hidden1)
        self.fc2 = torch.nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = torch.nn.Linear(num_hidden2, num_outputs)
        
        # Hardware-friendly LIF neurons
        self.lif1 = snn.Leaky(beta=0.85, threshold=0.5, reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta=0.85, threshold=0.5, reset_mechanism="zero")
        self.lif3 = snn.Leaky(beta=0.85, threshold=0.4, reset_mechanism="zero")
        
        # Regularization
        self.dropout1 = torch.nn.Dropout(0.4)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_rec = []

        for step in range(x.shape[1]):
            # Layer 1
            cur1 = self.dropout1(self.fc1(x[:, step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.dropout2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Output Layer
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk_rec.append(spk3)

        return torch.stack(spk_rec)

# Network Configuration
num_inputs = 784
num_hidden1 = 32
num_hidden2 = 16
num_outputs = 10
time_steps = 25
batch_size = 128
num_epochs = 15
learning_rate = 1e-3

# Data Loading with Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model Initialization
model = SNN(num_inputs, num_hidden1, num_hidden2, num_outputs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Weight Initialization
def kaiming_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0.1)
model.apply(kaiming_init)

# Tining Setup
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
criterion = SF.mse_count_loss()

# Enhanced Training Function
def train(model, loader):
    model.train()
    total_loss = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        # Check data size before reshaping
        print(f"Original data shape: {data.shape}")
        
        # Flatten and reshape data
        data = data.view(data.size(0), -1)  # Flatten to (batch_size, 784)
        data = data.unsqueeze(1).repeat(1, time_steps, 1)  # Reshape to (batch_size, time_steps, 784)
        
        print(f"Reshaped data: {data.shape}")
        
        optimizer.zero_grad()
        spk_rec = model(data)
        
        # Loss with spike regularization
        loss = criterion(spk_rec, targets)
        spike_reg = torch.mean(torch.sigmoid(5 * (spk_rec - 0.1)))
        loss += 0.1 * spike_reg
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation Function
def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            
            # Check data size before reshaping
            print(f"Original data shape (validate): {data.shape}")
            
            # Flatten and reshape data
            data = data.view(data.size(0), -1)  # Flatten to (batch_size, 784)
            data = data.unsqueeze(1).repeat(1, time_steps, 1)  # Reshape to (batch_size, time_steps, 784)
            
            print(f"Reshaped data (validate): {data.shape}")
            
            spk_rec = model(data)
            loss = criterion(spk_rec, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(spk_rec.sum(0), 1)
            correct += (predicted == targets).sum().item()
    
    return total_loss / len(loader), 100 * correct / len(loader.dataset)

# Training Loop
train_losses = []
val_losses = []
accuracies = []

for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = train(model, train_loader)
    val_loss, accuracy = validate(model, test_loader)
    
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]['lr']
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append(accuracy)
    
    print(f"Epoch {epoch+1:2d} | Time: {time.time()-start_time:.1f}s | "
          f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | "
          f"Acc: {accuracy:.1f}% | LR: {lr:.1e}")

# Plot Results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title("Loss Curves")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Validation Accuracy")
plt.tight_layout()
plt.savefig('training_results.png')

# Weight Export Function
def export_weights(model):
    with open('weights.c', 'w') as f_c, open('weights.h', 'w') as f_h:
        # Generate weights.h
        f_h.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f_h.write("#include <stdint.h>\n\n")
        f_h.write(f"#define NUM_INPUTS {num_inputs}\n")
        f_h.write(f"#define NUM_HIDDEN1 {num_hidden1}\n")
        f_h.write(f"#define NUM_HIDDEN2 {num_hidden2}\n")
        f_h.write(f"#define NUM_OUTPUTS {num_outputs}\n")
        f_h.write(f"#define TIME_STEPS {time_steps}\n\n")

        # Generate weights.c
        f_c.write("#include \"weights.h\"\n\n")
        
        # Quantize and export parameters
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy()
            quantized = np.clip(data * 256, -128, 127).astype(np.int8)
            
            # Write to .h
            f_h.write(f"extern const int8_t {name.replace('.', '_')}[{quantized.size}];\n")
            
            # Write to .c
            f_c.write(f"const int8_t {name.replace('.', '_')}[] = {{\n")
            f_c.write(", ".join(map(str, quantized.flatten())))
            f_c.write("\n};\n\n")
        
        f_h.write("\n#endif // WEIGHTS_H\n")

# Export weights for RISC-V deployment
export_weights(model)
print("Weights exported to weights.c and weights.h")
