import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.loss import Loss
from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import matplotlib.pyplot as plt


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    model.cuda()
    loss_total = 0
    counter = 0
    for data in dataloader:
        counter += 1
        targets_vec = []
        actions = ["throttle", "brake", "steer"]
        for target_name in actions:
            targets_vec.append(data[target_name])
        torch.cat(targets_vec, 1)
        with torch.no_grad():
            params = {'branches': model(torch.squeeze(data['rgb'].cuda()), data['speed'].cuda()),
                      'targets': torch.cat(targets_vec, 1).cuda(),
                      'branch_weights': [0.95, 0.95, 0.95, 0.95, 0.05],
                      'inputs': data['speed'].cuda(),
                      'controls': data['command'].cuda(),
                      'variable_weights': {'steer': 0.5, 'throttle': 0.45, 'brake': 0.05}}
            l = Loss(params)
            loss_total += l.item()
        #if counter % 500 == 0:
            #print("val iter " + str(counter))
    print("Val Loss:" + str(loss_total))
    return loss_total / counter


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_total = 0
    counter = 0
    for data in dataloader:
        counter += 1
        model.zero_grad()
        targets_vec = []
        branch_w = [0.95, 0.95, 0.95, 0.95, 0.05]
        variable_w = {'steer': 0.5, 'throttle': 0.45, 'brake': 0.05}
        actions = ["throttle", "brake", "steer"]
        for target_name in actions:
            targets_vec.append(data[target_name])
        torch.cat(targets_vec, 1)
        params = {'branches': model(torch.squeeze(data['rgb'].cuda()), data['speed'].cuda()),
                  'targets': torch.cat(targets_vec, 1).cuda(),
                  'branch_weights': branch_w,
                  'inputs': data['speed'].cuda(),
                  'controls': data['command'].cuda(),
                  'variable_weights': variable_w}
        l = Loss(params)
        l.backward()
        optimizer.step()
        loss_total += l.item()
        torch.cuda.empty_cache()
        #if counter % 500 == 0:
            #print("train iter " + str(counter))
    print("Train Loss :" + str(loss_total))
    return loss_total / counter


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('CILRS Plots')
    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    ax1.set_title('Train Loss')
    ax2.set_title('Validation Loss')
    train_loss_x = range(len(train_loss))
    val_loss_x = range(len(val_loss))
    ax1.plot(train_loss_x, train_loss)
    ax2.plot(val_loss_x, val_loss)
    plt.savefig('cilrs_loss.png')


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/eozsuer16/expert_data/train"
    val_root = "/userfiles/eozsuer16/expert_data/val"
    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        print('Epoch: ' + str(i))
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
