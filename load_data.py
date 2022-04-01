import json
from multiprocessing.sharedctypes import Value
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, instances, shuffle=False) -> None:
        super().__init__()
        self._instances = instances
        if shuffle:
            random.shuffle(self._instances)
        self._index = -1

    def __len__(self):
        return len(self._instances)

    def __getitem__(self, index):
        return self._instances[index]


def load_tracks(tracks_path):
    """
    Loads the tracks from the given path.
    """
    return json.load(open(tracks_path, 'r'))


def load_labels(labels_path):
    """
    Loads the labels from the given path.
    """
    return json.load(open(labels_path, 'r'))
    

def create_datasets(path_train, path_valid, batch_size: int = 16):

    data_train, data_valid =  json.load(open(path_train)), json.load(open(path_valid))]
    max_audio = 5

    train_pos, train_neg = [], []
    
    for (feature1, feature2, audios, label_y) in data_train:
    
        f1 = F.normalize(torch.tensor(feature1), p=2, dim=0)
        f2 = F.normalize(torch.tensor(feature2), p=2, dim=0)

        audio = [F.normalize(torch.tensor(au), p=2, dim=0) for au in audios]
        pad_aud = torch.cat((torch.zeros(len(audio)), torch.ones(max_audio - len(audio)) ))
        audio += [torch.zeros(128) for _  in range(max_audio - len(audio))]
        audio = torch.stack(audio)

        if label_y == 0:
            train_neg.append((f1, f2, audio, pad_aud, torch.tensor(label_y)))
            train_neg.append((f2, f1, audio, pad_aud, torch.tensor(label_y)))
        else:
            train_pos.append((f1, f2, audio, pad_aud, torch.tensor(label_y)))
            train_pos.append((f2, f1, audio, pad_aud, torch.tensor(label_y)))

    test = []

    for (feature1, feature2, audios, label_y) in data_valid:
    
        f1 = F.normalize(torch.tensor(feature1), p=2, dim=0)
        f2 = F.normalize(torch.tensor(feature2), p=2, dim=0)

        audio = [F.normalize(torch.tensor(au), p=2, dim=0) for au in audios]
        pad_aud = torch.cat((torch.zeros(len(audio)), torch.ones(max_audio - len(audio)) ))
        audio += [torch.zeros(128) for _  in range(max_audio - len(audio))]
        audio = torch.stack(audio)

        test.append((f1, f2, audio, pad_aud, torch.tensor(label_y)))
        test.append((f2, f1, audio, pad_aud, torch.tensor(label_y)))

    return DataLoader(CustomDataset(train_pos + train_neg, shuffle=True), batch_size=batch_size), DataLoader(test, batch_size=batch_size)


def create_dataloaders(path_train, path_valid, batch_size=16):
    """
    Creates dataloaders from the given paths.
    """
    
    return create_datasets(path_train, path_valid, batch_size)
