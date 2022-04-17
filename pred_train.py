import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
from utils.loss import CAL_loss


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    model.cuda()
    loss_total = 0
    counter = 0
    for data in dataloader:
        counter += 1
        with torch.no_grad():
            params = {'preds': model(torch.squeeze(data['rgb'].cuda()), data['command'].cuda()),
                      'labels': {'lane_dist': data['lane_dist'].cuda(),
                                 'route_angle': data['route_angle'].cuda(),
                                 'tl_state': data['tl_state'].cuda(),
                                 'tl_dist': data['tl_dist'].cuda()},
                      'weights': {
                          'tl_state': torch.Tensor([0.1109, 10.0]).cuda(),
                          'route_angle': torch.Tensor([1]).cuda(),
                          'lane_dist': torch.Tensor([1]).cuda(),
                          'tl_dist': torch.Tensor([1]).cuda()}
                      }
            l = CAL_loss(params)
            loss_total += l.item()
        if counter % 200 == 0:
            print("val iter " + str(counter))
    print("val loss:" + str(loss_total))
    return loss_total / counter


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_total = 0
    counter = 0
    for data in dataloader:
        counter += 1
        params = {'preds': model(torch.squeeze(data['rgb'].cuda()), data['command'].cuda()),
                  'labels': {'lane_dist': data['lane_dist'].cuda(),
                             'route_angle': data['route_angle'].cuda(),
                             'tl_state': data['tl_state'].cuda(),
                             'tl_dist': data['tl_dist'].cuda()},
                  'weights': {
                      'tl_state': torch.Tensor([0.1109, 10.0]).cuda(),
                      'route_angle': torch.Tensor([1]).cuda(),
                      'lane_dist': torch.Tensor([1]).cuda(),
                      'tl_dist': torch.Tensor([1]).cuda()}
                  }

        l = CAL_loss(params, optimizer)
        loss_total += l.item()
        torch.cuda.empty_cache()
        if counter % 500 == 0:
            print("train iter " + str(counter))
            print(l.item())
    print("Train loss:" + str(loss_total))
    return loss_total / counter


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Train Loss & Val Loss')
    train_loss_x = range(len(train_loss))
    val_loss_x = range(len(val_loss))
    ax1.plot(train_loss_x, train_loss)
    ax2.plot(val_loss_x, val_loss)
    plt.savefig('cal_loss.png')


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/eozsuer16/expert_data/train"
    val_root = "/userfiles/eozsuer16/expert_data/val"
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 2
    batch_size = 8
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
