import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm


def batch_epoch(model, dl, criterion, device, optimizer=None):
    running_loss = 0
    running_correct = 0
    for batch, (x_batch, y_batch) in enumerate(dl, start=1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        y_logits = model(x_batch)
        y_preds = y_logits.softmax(dim=1).argmax(dim=1)
        loss = criterion(y_logits, y_batch)
        correct = torch.eq(y_preds, y_batch).sum().item()
        
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * x_batch.shape[0]
        running_correct += correct
    n_data = len(dl.dataset)
    loss = running_loss / n_data
    acc = running_correct / n_data
    
    return loss, acc


@torch.inference_mode()
def evaluation(model, test_dl, criterion, device):
    model.eval()
    test_loss, test_acc = batch_epoch(model, test_dl, criterion, device)
    return test_loss, test_acc


def train(model, train_dl, val_dl, criterion, optimizer, epochs, device, run):
    if run:
        run.watch(model)
    
    loss_history = dict(train=[], val=[])
    acc_history = dict(train=[], val=[])
    with tqdm(range(1, epochs + 1)) as pbar:
        for epoch in pbar:
            pbar.set_description(f'Epoch={epoch}')
            
            model.train()
            train_loss, train_acc = batch_epoch(model, train_dl, criterion, device, optimizer)
            loss_history['train'].append(train_loss)
            acc_history['train'].append(train_acc)
            
            val_loss, val_acc = evaluation(model, val_dl, criterion, device)
            loss_history['val'].append(val_loss)
            acc_history['val'].append(val_acc)
            
            if run:
                wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc},
                          step=epoch)
            pbar.set_postfix(dict(train_loss=f'{train_loss:.3f}', train_acc=f'{train_acc:.2%}',
                                  val_loss=f'{val_loss:.3f}', val_acc=f'{val_acc:.2%}'))
    return loss_history, acc_history


def plot_train_history(loss_history, acc_history):
    _, axs = plt.subplots(1, 2, figsize=(8, 3))
    epochs = len(loss_history['train'])
    
    axs[0].plot(range(1, epochs + 1), loss_history['train'], label='train')
    axs[0].plot(range(1, epochs + 1), loss_history['val'], label='val')
    axs[0].set(xlabel='Epoch', ylabel='loss', title='Training Loss')
    axs[0].legend()
    
    axs[1].plot(range(1, epochs + 1), acc_history['train'], label='train')
    axs[1].plot(range(1, epochs + 1), acc_history['val'], label='val')
    axs[1].set(xlabel='Epoch', ylabel='accuracy', title='Training Accuracy')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
