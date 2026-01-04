import torch
import numpy as np

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        # setting device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CNN layers for image processing
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
        
        # Fully connected layers - now taking CNN features (64*12*12) + sensor values (1+4+1+1=7)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64 * 12 * 12 + 7, out_features=512),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(in_features=256, out_features=9)  # 9 action classes
        ).to(self.device)
        
        # Define 5 action classes:
        # 0: straight + accelerate
        # 1: left turn + accelerate  
        # 2: right turn + accelerate
        # 3: straight + brake
        # 4: turn + brake

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
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
        
        # Forward pass through fully connected layers
        return self.fc(combined_features)

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        Straight + gas 0
        Straight + brake 1
        Straight + no action 2
        Left + gas 3
        Left + brake 4
        Left + no action 5
        Right + gas 6
        Right + brake 7
        Right + no action 8
        """
        class_labels = []
        
        for action in actions:
            steer = action[0].item()
            gas = action[1].item()
            brake = action[2].item()
            
            # Determine class based on action values
            if brake == 0.8:  # braking
                if steer == -1:
                    class_idx = 4  # left + brake
                elif steer == 1:
                    class_idx = 7  # right + brake
                else:
                    class_idx = 1  # straight + brake
            else:  # not braking (gas or coast)
                if gas == 0.0: # coasting
                    if steer == -1:
                        class_idx = 5  # left + no action
                    elif steer == 1:
                        class_idx = 8  # right + no action
                    else:
                        class_idx = 2  # straight + no action
                elif gas == 0.5: # accelerating
                    if steer == 0:
                        class_idx = 0  # straight + gas
                    elif steer == -1:
                        class_idx = 3  # left + gas
                    elif steer == 1:
                        class_idx = 6  # right + gas
                
            
            class_labels.append(torch.tensor([class_idx], dtype=torch.long))
        
        return class_labels

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        # Define action mapping
        action_map = {
            0: torch.tensor([0.0, 0.5, 0.0]),   # straight + gas
            1: torch.tensor([-1.0, 0.0, 0.8]),  # left + brake
            2: torch.tensor([0.0, 0.0, 0.0]),   # straight + no action
            3: torch.tensor([-1.0, 0.5, 0.0]),  # left + gas
            4: torch.tensor([1.0, 0.0, 0.8]),   # right + brake
            5: torch.tensor([-1.0, 0.0, 0.0]),  # left + no action
            6: torch.tensor([1.0, 0.5, 0.0]),   # right + gas
            7: torch.tensor([1.0, 0.0, 0.8]),   # right + brake
            8: torch.tensor([1.0, 0.0, 0.0])    # right + no action
        }
        
        # Get class with highest score
        predicted_class = torch.argmax(scores).item()
        
        # Map class to action
        return action_map[predicted_class]

    def extract_sensor_values(self, observation, batch_size):
        # just approximately normalized, usually this suffices.
        # can be changed by you
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope