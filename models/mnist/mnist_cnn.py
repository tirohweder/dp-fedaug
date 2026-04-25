import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    """
    CNN architecture for MNIST, from thesis section 3.3.2.
    
    Architecture:
    - Conv2D: 32 neurons, kernel (5,5), padding=same, activation=ReLU
    - Conv2D: 32 neurons, kernel (5,5), padding=same, activation=ReLU
    - Max pooling 2D
    - Conv2D: 64 neurons, kernel (3,3), padding=same, activation=ReLU
    - Conv2D: 64 neurons, kernel (3,3), padding=same, activation=ReLU
    - Max pooling 2D
    - Flatten
    - Dense: 128 neurons, activation=ReLU
    - Dense output: 10 neurons
    """
    def __init__(self, use_dropout: bool = False, dropout_rate: float = 0.1):
        super(MNIST_CNN, self).__init__()
        
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        
        # First convolutional block
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # padding=2 for 'same'
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # padding=1 for 'same'
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After convolutions: 32 -> 16 -> 8, so 8x8x64 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout layers
        if self.use_dropout:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        if self.use_dropout:
            x = self.dropout1(x)
        
        # Second convolutional block
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        if self.use_dropout:
            x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout3(x)
        
        # Output (no softmax needed with CrossEntropyLoss)
        x = self.fc2(x)
        
        return x
