
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# metric and test structure inspired by
# Consistency-Aware Graph Network for Human Interaction Understanding by Wang et al.
# https://github.com/deepgogogo/CAGNet?v=1

def train_loop(modalities, models, train_dataloader, loss_fns, optimizers):

    '''Runs a single training epoch'''

    basic_modalities = ['vv', 'va', 'av' , 'aa']

    train_loss = 0
    y_trues = []
    y_preds_basic = {k : [] for k in basic_modalities}
    y_preds = { m : [] for m in modalities }

    for (f1, f2, aud, pad_aud, y) in train_dataloader:
        
        for (typ, (model, loss_fn, opt)) in enumerate(zip(models, loss_fns, optimizers)):

            opt.zero_grad()
            pred = model(f1, f2, aud, pad_aud)
            loss = loss_fn(pred, y)
            loss.backward()
            train_loss += loss.item()
            opt.step()

            y_preds_basic[basic_modalities[typ]] = pred

        for mode in modalities:
            curr_preds = [y_preds_basic[m] for m in mode.split('_') ]
            y_preds[mode] += list(torch.argmax(sum(curr_preds), dim=1))

        y_trues += list(y)

    train_loss /= (len(train_dataloader) * len(models))

    print(len(y_trues))

    return train_loss, {m : accuracy_score(y_trues, y_preds[m]) for m in modalities}

def test(modalities, models, dataloader, loss_fns):

    '''Runs a single training epoch'''

    basic_modalities = ['vv', 'va', 'av', 'aa']
    test_loss = 0
    y_trues = []
    y_preds_basic = {'vv': [], 'va': [], 'av': [], 'aa': []}
    y_preds = { m : [] for m in modalities }

    with torch.no_grad():
        for (f1, f2, aud, pad_aud, y) in dataloader:
            
            for (typ, (model, loss_fn)) in enumerate(zip(models, loss_fns)):

                pred = model(f1, f2, aud, pad_aud)
                loss = loss_fn(pred, y)
                test_loss += loss.item()

                y_preds_basic[basic_modalities[typ]] = pred

            for mode in modalities:
                curr_preds = [y_preds_basic[m] for m in mode.split('_') ]
                y_preds[mode] += list(torch.argmax(sum(curr_preds), dim=1))

            y_trues += list(y)

    test_loss /= (len(dataloader) * len(models))

    log = dict()
    log['loss'] = test_loss

    for m in modalities:
        log[f'accuracy_{m}'] = accuracy_score(y_trues, y_preds[m])
        log[f'precision_{m}'] = precision_score(y_trues, y_preds[m], average='macro', zero_division=0)
        log[f'recall_{m}'] = recall_score(y_trues, y_preds[m], average='macro', zero_division=0)
        log[f'f1_{m}'] = f1_score(y_trues, y_preds[m], average='macro', zero_division=0)

    return log


def train(modalities, models, train_dataset, test_dataset, losses, optimizers, writers, epochs: int = 30):

    test_res = []

    for epoch in range(epochs):
        print(f'EPOCH [{epoch + 1} / {epochs}]')

        for m in models: m.train()

        loss, accuracies = train_loop(modalities, models, train_dataset, losses, optimizers)
        
        print(f'train loss {round(loss, 3)}')

        for (m, writer) in zip(modalities, writers):
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/accuracy', accuracies[m], epoch)
            print(f'train accuracy {m} {round(accuracies[m], 3)* 100}%')
 
        for m in models: m.eval()

        log = test(modalities, models, test_dataset, losses)

        for (m, writer) in zip(modalities, writers):
            writer.add_scalar(f'test/loss', log[f'loss'], epoch)
            writer.add_scalar(f'test/accuracy', log[f'accuracy_{m}'], epoch)
            writer.add_scalar(f'test/precision', log[f'precision_{m}'], epoch)
            writer.add_scalar(f'test/recall', log[f'recall_{m}'], epoch)
            writer.add_scalar(f'test/f1', log[f'f1_{m}'], epoch)

        print(f"test loss {round(log['loss'], 3)}")
        for ext in modalities:
            print(f"test accuracy {ext} {round(log[f'accuracy_{ext}'], 3)* 100}%")
        
        test_res.append(log)
    
    return test_res

