from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=5, start_warmup_value=1e-6):
    """
    Create a cosine annealing schedule with warmup (adapted from B-cos)
    
    Args:
        base_value: Base learning rate after warmup
        final_value: Final learning rate at the end of training
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        start_warmup_value: Starting learning rate for warmup
    
    Returns:
        numpy array of learning rates for each epoch
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule

def apply_learning_rate_schedule(optimizer, lr_schedule, epoch):
    """
    Apply learning rate from pre-computed schedule to optimizer
    
    Args:
        optimizer: The optimizer to update
        lr_schedule: Pre-computed learning rate schedule (numpy array)
        epoch: Current epoch (0-indexed)
    """
    if epoch < len(lr_schedule):
        lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, lr_net_schedule, lr_classifier_schedule, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):
    """
    Train PIPNet with cosine annealing learning rate schedule
    
    Args:
        net: The PIPNet model
        train_loader: Training data loader
        optimizer_net: Network optimizer
        optimizer_classifier: Classifier optimizer
        lr_net_schedule: Pre-computed learning rate schedule for network (numpy array)
        lr_classifier_schedule: Pre-computed learning rate schedule for classifier (numpy array)
        criterion: Loss criterion
        epoch: Current epoch (0-indexed)
        nr_epochs: Total number of epochs
        device: Device to run on
        pretrain: Whether in pretraining mode
        finetune: Whether in finetuning mode
        progress_prefix: Prefix for progress bar
    """

    # Apply learning rate schedules
    apply_learning_rate_schedule(optimizer_net, lr_net_schedule, epoch)
    apply_learning_rate_schedule(optimizer_classifier, lr_classifier_schedule, epoch)

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, net.module._classification.normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)
        
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()   
            # Get current learning rate from schedule
            if epoch < len(lr_classifier_schedule):
                lrs_class.append(lr_classifier_schedule[epoch])
            else:
                lrs_class.append(optimizer_classifier.param_groups[0]['lr'])
     
        if not finetune:
            optimizer_net.step()
            # Get current learning rate from schedule  
            if epoch < len(lr_net_schedule):
                lrs_net.append(lr_net_schedule[epoch])
            else:
                lrs_net.append(optimizer_net.param_groups[0]['lr'])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))
    
    # Learning rates are already applied at the beginning of each epoch
    # No need to step schedulers
        
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info

#TODO - replace with BCE loss
def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, print=True, EPS=1e-10):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.

    if not finetune:
        loss = align_pf_weight*a_loss_pf
        loss += t_weight * tanh_loss
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        
        if finetune:
            loss= cl_weight * class_loss
        else:
            loss+= cl_weight * class_loss
    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    # else:
    #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc=0.
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    if print: 
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
            else:
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)            
    return loss, acc

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def create_cosine_schedule_simple(base_lr, total_epochs, final_lr_ratio=0.01, warmup_epochs=5):
    """
    Simple wrapper to create a cosine schedule compatible with B-cos approach
    
    Args:
        base_lr: Base learning rate after warmup
        total_epochs: Total number of epochs 
        final_lr_ratio: Final LR as ratio of base LR (default: 0.01 = 1%)
        warmup_epochs: Number of warmup epochs (default: 5)
    
    Returns:
        numpy array of learning rates for each epoch
    
    Example usage:
        lr_schedule = create_cosine_schedule_simple(0.003, 200, 0.01, 5)
        apply_learning_rate_schedule(optimizer, lr_schedule, epoch)
    """
    final_lr = base_lr * final_lr_ratio
    start_warmup_lr = base_lr / 1000.
    return cosine_scheduler(base_lr, final_lr, total_epochs, warmup_epochs, start_warmup_lr)

def create_cosine_schedule_warm_restarts(base_lr, total_epochs, restart_period=10, final_lr=0.001):
    """
    Create a cosine schedule with warm restarts (multiple cycles)
    
    Args:
        base_lr: Base learning rate for each cycle
        total_epochs: Total number of epochs
        restart_period: Length of each cycle  
        final_lr: Final learning rate for each cycle
    
    Returns:
        numpy array of learning rates for each epoch
    
    Example usage:
        lr_schedule = create_cosine_schedule_warm_restarts(0.01, 200, 10, 0.001)
        apply_learning_rate_schedule(optimizer, lr_schedule, epoch)
    """
    import numpy as np
    
    num_cycles = max(1, total_epochs // restart_period)
    lr_schedule = []
    
    for cycle in range(num_cycles):
        cycle_epochs = min(restart_period, total_epochs - cycle * restart_period)
        if cycle_epochs > 0:
            cycle_schedule = cosine_scheduler(
                base_value=base_lr,
                final_value=final_lr,
                epochs=cycle_epochs,
                warmup_epochs=0,  # No warmup for restarts
                start_warmup_value=final_lr
            )
            lr_schedule.extend(cycle_schedule)
    
    # Pad or trim to exact number of epochs
    if len(lr_schedule) > total_epochs:
        lr_schedule = lr_schedule[:total_epochs]
    elif len(lr_schedule) < total_epochs:
        final_lr_val = lr_schedule[-1] if lr_schedule else final_lr
        lr_schedule.extend([final_lr_val] * (total_epochs - len(lr_schedule)))
    
    return np.array(lr_schedule)