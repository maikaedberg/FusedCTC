import json
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


    data = [json.load(open(path_train)), json.load(open(path_valid))]
    dataset = [[], []]
    max_audio = 29

    for i in range(2):

        train_pos, train_neg = [], []

        for (feature1, feature2, audio, label) in data[i]:

            f1 = F.normalize(torch.tensor(feature1), p=2, dim=0)
            f2 = F.normalize(torch.tensor(feature2), p=2, dim=0)
            
            if audio == []: continue

            audio = [F.normalize(torch.tensor(i), p=2, dim=0) for i in audio]
            pad_aud = torch.cat((torch.zeros(len(audio)), torch.zeros(max_audio - len(audio)) ))
            audio += [torch.zeros(128) for _  in range(max_audio - len(audio))]
            audio = torch.stack(audio)
            
            label = torch.tensor(label)

            if label == 0 and i == 0:
                train_neg.append((f1, f2, audio, pad_aud, label))
                train_neg.append((f2, f1, audio, pad_aud, label))
            else:
                train_pos.append((f1, f2, audio, pad_aud, label))
                train_pos.append((f2, f1, audio, pad_aud, label))                
  
        #if i == 0:
        #   train_neg = random.sample(train_neg, min(len(train_pos), len(train_neg)))
        #   train_pos = random.sample(train_pos, min(len(train_pos), len(train_neg)))

        dataset[i] = CustomDataset(train_neg + train_pos, shuffle=True)

    return DataLoader(dataset[0], batch_size=batch_size), DataLoader(dataset[1], batch_size=batch_size)


def create_dataloaders(path_train, path_valid, batch_size=16):
    """
    Creates dataloaders from the given paths.
    """
    
    return create_datasets(path_train, path_valid, batch_size)
