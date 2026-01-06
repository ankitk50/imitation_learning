import torch
import torch.nn as nn
import random
import time
from network import ClassificationNetwork
from network_binary import BinaryClassificationNetwork
from demonstrations import load_demonstrations
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


def train(data_folder, trained_network_file, args):
    """
    Function for training the network.
    Supports both multi-class classification (default) and binary classification.
    """
    # Select network type based on args
    use_binary = getattr(args, 'binary', False)
    
    if use_binary:
        print("Using Binary Classification Network (4 independent binary outputs)")
        infer_action = BinaryClassificationNetwork()
        # Binary Cross Entropy loss for multi-label binary classification
        loss_function = nn.BCELoss()
    else:
        print("Using Multi-class Classification Network (9 classes)")
        infer_action = ClassificationNetwork()
        # Cross Entropy loss for multi-class classification
        loss_function = nn.CrossEntropyLoss()

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action = infer_action.to(device)
    
    # Use a lower learning rate for binary classification (BCE loss is more sensitive)
    lr = args.lr if not use_binary else min(args.lr, 1e-4)
    print(f"Using learning rate: {lr}")
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=lr)

    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]

    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        num_batches = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                
                if use_binary:
                    # For binary classification: reshape to (batch, 4)
                    batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, 4))
                else:
                    # For multi-class: reshape to (batch,)
                    batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1,))

                batch_out = infer_action(batch_in)
                loss = loss_function(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        avg_loss = total_loss / num_batches  # Average loss per batch
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, avg_loss, time_left))

    torch.save(infer_action, trained_network_file)
    print(f"Model saved to {trained_network_file}")