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
    tl_state_loss = 0
    route_angle_loss = 0
    tl_dist_loss = 0
    lane_dist_loss = 0
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
            affordance_l, total_l = CAL_loss(params)
            loss_total += total_l.item()
            tl_state_loss += affordance_l['tl_state'].item()
            lane_dist_loss += affordance_l['lane_dist'].item()
            tl_dist_loss += affordance_l['tl_dist'].item()
            route_angle_loss += affordance_l['route_angle'].item()
        if counter % 200 == 0:
            print("val iter " + str(counter))
    print("val loss:" + str(loss_total))
    avg_loss = loss_total / counter
    losses = {'avg_loss': avg_loss, 'tl_state': tl_state_loss,
              'tl_dist': tl_dist_loss, 'route_angle': route_angle_loss,
              'lane_dist': lane_dist_loss}
    return losses


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_total = 0
    tl_state_loss = 0
    route_angle_loss = 0
    tl_dist_loss = 0
    lane_dist_loss = 0
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

        affordance_l, total_l = CAL_loss(params, optimizer)
        loss_total += total_l.item()
        tl_state_loss += affordance_l['tl_state'].item()
        lane_dist_loss += affordance_l['lane_dist'].item()
        tl_dist_loss += affordance_l['tl_dist'].item()
        route_angle_loss += affordance_l['route_angle'].item()
        torch.cuda.empty_cache()
        if counter % 500 == 0:
            print("train iter " + str(counter))
    print("Train loss:" + str(loss_total))
    avg_loss = loss_total / counter
    losses = {'avg_loss':avg_loss, 'tl_state':tl_state_loss,
              'tl_dist': tl_dist_loss,'route_angle':route_angle_loss,
              'lane_dist':lane_dist_loss}
    return losses


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    avg_loss_train = train_loss['avg_loss']
    tl_state_loss_train = train_loss['tl_state']
    route_angle_loss_train = train_loss['route_angle']
    tl_dist_loss_train = train_loss['tl_dist']
    lane_dist_loss_train = train_loss['lane_dist']

    avg_loss_val = val_loss['avg_loss']
    tl_state_loss_val = val_loss['tl_state']
    route_angle_loss_val = val_loss['route_angle']
    tl_dist_loss_val  = val_loss['tl_dist']
    lane_dist_loss_val = val_loss['lane_dist']

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Affordance Plots')
    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    ax1.set_title('Train Loss')
    ax2.set_title('Validation Loss')

    avg_loss_train_x = range(len(avg_loss_train))
    tl_state_loss_train_x = range(len(tl_state_loss_train))
    route_angle_loss_train_x = range(len(route_angle_loss_train))
    tl_dist_loss_train_x = range(len(tl_dist_loss_train))
    lane_dist_loss_train_x = range(len(lane_dist_loss_train))

    avg_loss_val_x = range(len(avg_loss_val))
    tl_state_loss_val_x = range(len(tl_state_loss_val))
    route_angle_loss_val_x = range(len(route_angle_loss_val))
    tl_dist_loss_val_x = range(len(tl_dist_loss_val))
    lane_dist_loss_val_x = range(len(lane_dist_loss_val))

    ax1.plot(avg_loss_train_x, avg_loss_train, 'b', label='avg_loss')
    ax1.plot(tl_state_loss_train_x, tl_state_loss_train, 'g', label='tl_state')
    ax1.plot(route_angle_loss_train_x, route_angle_loss_train, 'r', label='route_angle')
    ax1.plot(tl_dist_loss_train_x, tl_dist_loss_train, 'c', label='tl_dist')
    ax1.plot(lane_dist_loss_train_x, lane_dist_loss_train, 'm', label='lane_dist')
    ax1.legend()

    ax2.plot(avg_loss_val_x, avg_loss_val, 'b', label='avg_loss')
    ax2.plot(tl_state_loss_val_x, tl_state_loss_val, 'g', label='tl_state')
    ax2.plot(route_angle_loss_val_x, route_angle_loss_val, 'r', label='route_angle')
    ax2.plot(tl_dist_loss_val_x, tl_dist_loss_val, 'c', label='tl_dist')
    ax2.plot(lane_dist_loss_val_x, lane_dist_loss_val, 'm', label='lane_dist')
    ax2.legend()

    plt.savefig('cal_loss.png')


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/userfiles/eozsuer16/expert_data/train"
    val_root = "/userfiles/eozsuer16/expert_data/val"
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 8
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
