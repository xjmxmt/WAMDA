import torch


def Binary_entropyloss(logits_domain,labels):
    loss_source = 0
    loss_target = 0
    num_source = 0
    num_target = 0
    for i in range(0, 4):
        if labels[i] == 0:
            output = torch.log(logits_domain[i][0])
            loss_source = torch.add(loss_source, output)
            num_source += 1
        else:
            output = torch.log(1 - logits_domain[i][1])
            loss_target = torch.add(loss_target, output)
            num_target += 1
    loss = -torch.div(loss_source, num_source) - torch.div(loss_target, num_target)
    return loss