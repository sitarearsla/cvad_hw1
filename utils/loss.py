import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_masks(controls):
    # c = 3, branch 1 (lanefollow) is activated
    c_b1 = create_mask(controls, 3)
    # c = 0, branch 2 (turn left) is activated
    c_b2 = create_mask(controls, 0)
    # c = 1, branch 3 (turn right) is activated
    c_b3 = create_mask(controls, 1)
    # c = 2, branch 4 (go straight) is activated
    c_b4 = create_mask(controls, 2)
    return [c_b1, c_b2, c_b3, c_b4]


def create_mask(controls, given):
    num_of_actions = 3
    branch = (controls == given)
    branch = torch.tensor(branch, dtype=torch.float32).cuda()
    return torch.cat([branch] * num_of_actions, 1)


def l2(params):
    predicted_speed = params['branches'][-1]
    loss_l2 = nn.MSELoss()
    loss_branches_vec = []
    for i in range(4):
        action_loss = loss_l2(params['branches'][i], params['targets']) * params['controls_mask'][i] * 0.95
        loss_branches_vec.append(action_loss)
    speed_loss = loss_l2(predicted_speed, params['speed']) * 0.05
    loss_branches_vec.append(speed_loss)
    return loss_branches_vec


def Loss(params):
    controls_mask = compute_masks(params['controls'])
    params.update({'controls_mask': controls_mask})
    #  loss for each branch
    loss_branches_vec = l2(params)
    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] + loss_branches_vec[i][:, 1] + loss_branches_vec[i][:, 2]

    loss_function = loss_branches_vec[0] + loss_branches_vec[1] + loss_branches_vec[2] + \
                    loss_branches_vec[3]

    speed_loss = loss_branches_vec[4] / (params['branches'][0].shape[0])

    return torch.sum(loss_function) / (params['branches'][0].shape[0]) \
           + torch.sum(speed_loss) / (params['branches'][0].shape[0])


def WCE(x, y, w):
    """weighted mean average"""
    t = F.cross_entropy(x, torch.argmax(y, dim=1), weight=w)
    return t


def MAE(x, y, w):
    return F.l1_loss(x, y) * w


def CAL_loss(params, opt=None):
    preds = params['preds']
    labels = params['labels']
    weights = params['weights']
    losses = {}
    for i in preds:
        value = preds[i]
        if value.shape[1] == 1:
            loss = MAE(value, labels[i], weights[i])
        else:
            loss = WCE(value, labels[i], weights[i])
        losses[i] = loss

    total_loss = losses['lane_dist'] + losses['route_angle'] + losses['tl_state'] + losses['tl_dist']

    if opt is not None:
        total_loss.backward()
        opt.step()
        opt.zero_grad()

    return losses, total_loss
