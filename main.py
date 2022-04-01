import sys
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from load_data import create_dataloaders
from model import make_models
from train import train

if __name__ == '__main__':

    # usage python3 main.py <epoch_no> <dropout_p> <weight_decay>
    assert len(sys.argv) >= 2
    epoch_no = int(sys.argv[1])
    dropout_p = float(sys.argv[2]) if len(sys.argv) >= 3 else 0
    weight_decay = float(sys.argv[3]) if len(sys.argv) >= 4 else 0

    # Load the dataset
    data_train = '../data/video_dataset/features_set1_train.json'
    data_valid = '../data/video_dataset/features_set2_valid.json'

    # Create the model
    train_dataloader, test_dataloader = create_dataloaders(data_train, data_valid)
    print(f'Loaded data from {data_train} and {data_valid} into memory')

    qmodes = ['vision', 'vision']
    kmodes = ['vision', 'audio']
    models = make_models(qmodes, kmodes, N=2)

    writer = SummaryWriter()

    # Train the model

    losses = [(nn.CrossEntropyLoss(), nn.CrossEntropyLoss()) for _ in models]
    optimizers = [(
        torch.optim.Adam(m_a.parameters(), lr=1e-7, weight_decay=weight_decay),
        torch.optim.Adam(m_i.parameters(), lr=1e-7, weight_decay=weight_decay)) 
            for (m_a, m_i) in models]

    train(models, train_dataloader, test_dataloader, losses, optimizers, writer, epochs=100)

    writer.close()

    models[0][0].save('models/TC_action.pt')
    models[0][1].save('models/TC_interaction.pt')
    models[1][0].save('models/CrossTC_action.pt')
    models[1][1].save('models/CrossTC_interaction.pt')

