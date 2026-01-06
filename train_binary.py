import torch
import random
import time
import torch.nn as nn
from demonstrations import load_demonstrations
from network import BinaryClassificationNetwork

def train_binary_classification(data_folder, trained_network_file, args):
    """
    Training loop for BinaryClassificationNetwork using BCELoss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize network
    model = BinaryClassificationNetwork()
    model.to(device)
    model.train()  # Set model to training mode
    
    # Binary Cross Entropy with Logits Loss - more numerically stable
    # Works with raw logits (model returns logits, not sigmoid)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer - use a lower learning rate for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      model.actions_to_classes(actions))]

    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    start_time = time.time()
    
    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0.0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, 4))

                batch_out = model(batch_in)
                loss = criterion(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()  # Use .item() to avoid gradient accumulation

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(model, trained_network_file)


def compute_binary_accuracy(model, dataloader):
    """
    Compute per-class and overall accuracy for binary classification.
    """
    device = model.device
    model.eval()
    
    correct = torch.zeros(4)
    total = 0
    
    with torch.no_grad():
        for observations, actions in dataloader:
            binary_labels = model.actions_to_binary_labels(actions)
            binary_labels = torch.stack(binary_labels).to(device)
            
            predictions = model(observations)
            predicted_binary = (predictions > 0.5).float()
            
            correct += (predicted_binary == binary_labels).sum(dim=0).cpu()
            total += binary_labels.size(0)
    
    per_class_accuracy = correct / total
    overall_accuracy = correct.sum() / (total * 4)
    
    print(f"Per-class accuracy [left, right, gas, brake]: {per_class_accuracy}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    return per_class_accuracy, overall_accuracy
