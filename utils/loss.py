import torch
import torch.nn.functional as F


def compute_masks(controls, num_targets):
    c_masks = []
    # c = 3, branch 1 (lanefollow) is activated
    c_b1 = (controls == 3)
    c_b1 = torch.tensor(c_b1, dtype=torch.float32).cuda()
    c_b1 = torch.cat([c_b1] * num_targets, 1)
    c_masks.append(c_b1)
    # c = 0, branch 2 (turn left) is activated
    c_b2 = (controls == 0)
    c_b2 = torch.tensor(c_b2, dtype=torch.float32).cuda()
    c_b2 = torch.cat([c_b2] * num_targets, 1)
    c_masks.append(c_b2)
    # c = 1, branch 3 (turn right) is activated
    c_b3 = (controls == 1)
    c_b3 = torch.tensor(c_b3, dtype=torch.float32).cuda()
    c_b3 = torch.cat([c_b3] * num_targets, 1)
    c_masks.append(c_b3)
    # c = 2, branch 4 (go straight) is activated
    c_b4 = (controls == 2)
    c_b4 = torch.tensor(c_b4, dtype=torch.float32).cuda()
    c_b4 = torch.cat([c_b4] * num_targets, 1)
    c_masks.append(c_b4)
    return c_masks


def l2(params):
    loss_branches_vec = []
    for i in range(4):
        loss_branches_vec.append(((params['branches'][i] - params['targets']) ** 2
                                  * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    loss_branches_vec.append((params['branches'][-1] - params['inputs']) ** 2
                             * params['branch_weights'][-1])
    return loss_branches_vec


def Loss(params):
    controls_mask = compute_masks(params['controls'], params['branches'][0].shape[1])
    params.update({'controls_mask': controls_mask})
    #  loss for each branch
    loss_branches_vec = l2(params)
    for i in range(4):
        loss_branches_vec[i] = loss_branches_vec[i][:, 0] * params['variable_weights']['steer'] \
                               + loss_branches_vec[i][:, 1] * params['variable_weights']['throttle'] \
                               + loss_branches_vec[i][:, 2] * params['variable_weights']['brake']

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
