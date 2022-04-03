import sys
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import json
from load_data import create_dataloaders
from model import make_models
from train import train

if __name__ == '__main__':

    # usage python3 main.py <epoch_no> <dropout_p> <weight_decay>
    assert len(sys.argv) >= 2
    epoch_no = int(sys.argv[1])
    dropout_p = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    wd = float(sys.argv[3]) if len(sys.argv) >= 4 else 0

    # Load the dataset
    data_train = '../data/dataset_with_audio/features_valid.json'
    data_valid = '../data/dataset_with_audio/features_train.json'

    # Create the model
    train_dataloader, test_dataloader = create_dataloaders(data_train, data_valid)
    print(f'Loaded data from {data_train} and {data_valid} into memory')

    qmodes = ['vision', 'vision', 'audio']
    kmodes = ['vision', 'audio',  'vision']

    models = make_models(qmodes, kmodes, N=2)

    modalities = ['vv', 'va', 'av', 'va_av', 'vv_va', 'vv_av', 'vv_va_av']
    #modalities = ['vv', 'va', 'va_va']
    writers = [ SummaryWriter() for _ in modalities ]

    # Train the model

    losses = [nn.CrossEntropyLoss() for _ in models]
    optimizers = [ torch.optim.Adam(m_i.parameters(), lr=1e-6, weight_decay=wd) for m_i in models ]

    results = train(modalities, models, train_dataloader, test_dataloader, losses, optimizers, writers, epochs=epoch_no)
    json.dump(results, open('result_log.json', 'w'))

    for writer in writers: writer.close()

    models[0].save('models/model_vv.pt')
    models[1].save('models/model_va.pt')
    models[2].save('models/move_av.pt')

