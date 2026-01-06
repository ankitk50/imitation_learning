import torch
import numpy as np

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


class BinaryClassificationNetwork(torch.nn.Module):
    """
    Multi-label binary classification network for self-driving car control.
    
    This network predicts 4 independent binary classes representing keyboard arrow keys:
    - Class 0: Steer Left (left arrow)
    - Class 1: Steer Right (right arrow)
    - Class 2: Accelerate (up arrow)
    - Class 3: Brake (down arrow)
    
    Each class is predicted independently using sigmoid activation, allowing
    combinations like "accelerate + steer left" to be predicted simultaneously.
    """
    
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        # Setting device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CNN layers for image processing (same architecture as ClassificationNetwork)
        self.cnn = torch.nn.Sequential(
            # 1st conv layer
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # 2nd conv layer
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            # 3rd conv layer
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Flatten()
        ).to(self.device)
        
        # Fully connected layers - CNN features (64*12*12) + sensor values (1+4+1+1=7)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64 * 12 * 12 + 7, out_features=512),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(in_features=256, out_features=4),  # 4 binary outputs
            torch.nn.Sigmoid()  # Sigmoid for independent binary predictions
        ).to(self.device)

    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, 4) with values in [0, 1]
                       representing probabilities for each binary class:
                       [steer_left, steer_right, accelerate, brake]
        """
        batch_size = observation.shape[0]
        
        # Convert from (batch_size, H, W, C) to (batch_size, C, H, W)
        observation_permuted = observation.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] if not already normalized
        if observation_permuted.dtype == torch.uint8:
            observation_permuted = observation_permuted.float() / 255.0
        
        # Move to device
        observation_permuted = observation_permuted.to(self.device)
        observation = observation.to(self.device)
        
        # Extract CNN features
        cnn_features = self.cnn(observation_permuted)
        
        # Extract sensor values from original observation
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, batch_size)
        
        # Concatenate CNN features with sensor values
        combined_features = torch.cat([cnn_features, speed, abs_sensors, steering, gyroscope], dim=1)
        
        # Forward pass through fully connected layers (includes sigmoid)
        return self.fc(combined_features)

    def actions_to_classes(self, actions):
        """
        For a given set of actions, map every action to its corresponding
        binary class representation. Each action maps to a 4-dim binary vector.
        
        Binary class mapping:
        - Index 0: Steer Left (1 if steering < 0, else 0)
        - Index 1: Steer Right (1 if steering > 0, else 0)
        - Index 2: Accelerate (1 if gas > 0, else 0)
        - Index 3: Brake (1 if brake > 0, else 0)
        
        actions:        python list of N torch.Tensors of size 3 [steer, gas, brake]
        return          python list of N torch.Tensors of size 4
        """
        class_labels = []
        
        for action in actions:
            steer = action[0].item()
            gas = action[1].item()
            brake = action[2].item()
            
            # Create binary vector [steer_left, steer_right, accelerate, brake]
            steer_left = 1.0 if steer < 0 else 0.0
            steer_right = 1.0 if steer > 0 else 0.0
            accelerate = 1.0 if gas > 0 else 0.0
            brake_active = 1.0 if brake > 0 else 0.0
            
            binary_label = torch.tensor([steer_left, steer_right, accelerate, brake_active], 
                                         dtype=torch.float32)
            class_labels.append(binary_label)
        
        return class_labels

    def scores_to_action(self, scores):
        """
        Maps the binary scores predicted by the network to continuous action values.
        
        scores:         torch.Tensor of size (batch_size, 4) or (4,)
                        [steer_left_prob, steer_right_prob, accelerate_prob, brake_prob]
        return          (float, float, float) representing [steer, gas, brake]
        """
        # Handle both batched and single predictions
        if scores.dim() > 1:
            scores = scores.squeeze(0)
        
        # Threshold at 0.5 for binary decisions
        steer_left = scores[0].item() > 0.5
        steer_right = scores[1].item() > 0.5
        accelerate = scores[2].item() > 0.5
        brake = scores[3].item() > 0.5
        
        # Convert binary decisions to continuous actions
        # Steering: -1 (left), 0 (straight), 1 (right)
        if steer_left and not steer_right:
            steer = -1.0
        elif steer_right and not steer_left:
            steer = 1.0
        elif steer_left and steer_right:
            # Both pressed - use probability difference
            steer = scores[1].item() - scores[0].item()
        else:
            steer = 0.0
        
        # Gas: 0.5 if accelerating (matching original network values)
        gas = 0.5 if accelerate and not brake else 0.0
        
        # Brake: 0.8 if braking (matching original network values)
        brake_val = 0.8 if brake else 0.0
        
        return (steer, gas, brake_val)

    def extract_sensor_values(self, observation, batch_size):
        """
        Extract sensor values from the observation image.
        Same implementation as ClassificationNetwork for consistency.
        """
        # Speed extraction
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        # ABS sensors extraction
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        # Steering extraction
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        # Gyroscope extraction
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
