# Adapted from
# https://github.com/tensorflow/models/tree/master/research/rebar and
# https://github.com/duvenaud/relax/blob/master/datasets.py

import logging
import numpy as np
import os
import urllib.request
import torch.utils.data
import scipy.io

OMNIGLOT_URL = "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat"
OMNIGLOT_PATH = "/Users/tuananhle/Documents/research/datasets/omniglot/omniglot.mat"
OMNIGLOT_PATH_CC = "/home/tuananh/projects/def-fwood/tuananh/datasets/omniglot/omniglot.mat"
OMNIGLOT_PATH_OM = "/om2/user/tuananh/datasets/omniglot/omniglot.mat"
OMNIGLOT_PATH_SATORI = "/nobackup/users/tuananh/datasets/omniglot/omniglot.mat"
OMNIGLOT_PATH_SATORI_LUKE = "/nobackup/users/lbh/datasets/omniglot/omniglot.mat"


def download_omniglot(path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        logging.info("Downloaded {} to {}".format(url, path))


def load_binarized_omniglot(location=None, path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        z_train,
        z_valid,
        z_test,
    ) = load_binarized_omniglot_with_everything(location, path, url)

    return x_train, x_valid, x_test


def load_binarized_omniglot_with_targets(location=None, path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        z_train,
        z_valid,
        z_test,
    ) = load_binarized_omniglot_with_everything(location, path, url)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def load_binarized_omniglot_with_characters(location=None, path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        z_train,
        z_valid,
        z_test,
    ) = load_binarized_omniglot_with_everything(location, path, url)

    return x_train, x_valid, x_test, z_train, z_valid, z_test


def load_binarized_omniglot_with_everything(location=None, path=OMNIGLOT_PATH, url=OMNIGLOT_URL):
    if location is not None:
        if location == "local":
            path = OMNIGLOT_PATH
        elif location == "cc_cedar":
            path = OMNIGLOT_PATH_CC
        elif location == "om":
            path = OMNIGLOT_PATH_OM
        elif location == "satori":
            path = OMNIGLOT_PATH_SATORI
        elif location == "satori-luke":
            path = OMNIGLOT_PATH_SATORI_LUKE

    download_omniglot(path, url)
    n_validation = 1345

    def reshape_data(data):
        return data.reshape((-1, 28, 28))

    omni_raw = scipy.io.loadmat(path)
    logging.info("Loaded {}".format(path))

    train_data = reshape_data(omni_raw["data"].T.astype("float32"))
    test_data = reshape_data(omni_raw["testdata"].T.astype("float32"))
    train_target = omni_raw["target"].T.astype("float32")
    test_target = omni_raw["testtarget"].T.astype("float32")
    train_targetchar = omni_raw["targetchar"].T.astype("float32")
    test_targetchar = omni_raw["testtargetchar"].T.astype("float32")

    # Binarize the data with a fixed seed
    np.random.seed(5)
    train_data = (0.5 < train_data).astype(float)
    test_data = (0.5 < test_data).astype(float)

    shuffle_seed = 123
    permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
    train_data = train_data[permutation]
    train_target = train_target[permutation]
    train_targetchar = train_targetchar[permutation]

    x_train = train_data[:-n_validation]
    x_valid = train_data[-n_validation:]
    x_test = test_data

    y_train = train_target[:-n_validation]
    y_valid = train_target[-n_validation:]
    y_test = test_target

    alphabet_train = y_train.nonzero()[1]
    alphabet_valid = y_valid.nonzero()[1]
    alphabet_test = y_test.nonzero()[1]

    targetchar_train = train_targetchar[:-n_validation][:, 0]
    targetchar_valid = train_targetchar[-n_validation:][:, 0]
    targetchar_test = test_targetchar[:, 0]

    all_alphabets = np.concatenate([alphabet_train, alphabet_valid, alphabet_test])
    all_targetchars = np.concatenate([targetchar_train, targetchar_valid, targetchar_test])

    l = list(zip(all_alphabets, all_targetchars))
    get_character = {
        (a, tc): i for (i, (a, tc)) in enumerate(set(zip(all_alphabets, all_targetchars)))
    }

    z_train = np.array(
        [get_character[(a, tc)] for a, tc in zip(alphabet_train, targetchar_train)], dtype=np.int
    )
    z_valid = np.array(
        [get_character[(a, tc)] for a, tc in zip(alphabet_valid, targetchar_valid)], dtype=np.int
    )
    z_test = np.array(
        [get_character[(a, tc)] for a, tc in zip(alphabet_test, targetchar_test)], dtype=np.int
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test, z_train, z_valid, z_test


def get_data_loader(np_array, batch_size, device, ids=False, id_offset=0, shuffle=True):
    """Args:
        np_array: shape [num_data, dim_1, ..., dim_N]
        batch_size: int
        device: torch.device object
        ids: bool. if True then each batch consists of
            a tuple (data_ids, data) where
                data_ids: tensor [batch_size]
                data: tensor [batch_size, dim_1, ..., dim_N]

    Returns: torch.utils.data.DataLoader object
    """
    if device == torch.device("cpu"):
        kwargs = {"num_workers": 4, "pin_memory": True}
    else:
        kwargs = {}

    if ids:
        dataset = torch.utils.data.TensorDataset(
            torch.arange(len(np_array), device=device) + id_offset,
            torch.tensor(np_array, dtype=torch.float, device=device),
        )
    else:
        dataset = torch.tensor(np_array, dtype=torch.float, device=device)

    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
    )


class OmniglotDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, device, ids=False, id_offset=0, target_type=torch.float):
        self.data = torch.tensor(data, dtype=torch.float, device=device)
        self.target = torch.tensor(target, dtype=target_type, device=device)
        self.ids = ids
        if ids:
            self.arange = torch.arange(id_offset, len(self.data) + id_offset, device=device)

    def __getitem__(self, index):
        if self.ids:
            return self.arange[index], (self.data[index], self.target[index])
        else:
            return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def get_conditional_omniglot_data_loader(data, target, batch_size, device, ids=False, id_offset=0):
    return torch.utils.data.DataLoader(
        dataset=OmniglotDataset(data, target, device, ids, id_offset=id_offset),
        batch_size=batch_size,
        shuffle=True,
    )


def get_characters_omniglot_data_loader(data, target, batch_size, device, ids=False, id_offset=0):
    return torch.utils.data.DataLoader(
        dataset=OmniglotDataset(
            data, target, device, ids, id_offset=id_offset, target_type=torch.long
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def split_data_by_target(data, target, num_data_per_target):
    """
    Args:
        data: np.array [num_data, *data_dims]
        target: np.array [num_data, num_targets]
            target[i] is a one hot
        num_data_per_target: int

    Returns:
        result_data: np.array [num_data_per_target * num_targets, *data_dims]
        result_target: np.array
            [num_data_per_target * num_targets, num_targets]
    """
    num_unique_targets = len(np.unique(target, axis=0))
    target_numeric = np.dot(target, np.arange(num_unique_targets))
    result_data = []
    result_target = []
    for target_id in range(num_unique_targets):
        result_data.append(data[target_numeric == target_id][:num_data_per_target])
        result_target.append(target[target_numeric == target_id][:num_data_per_target])

    return np.concatenate(result_data), np.concatenate(result_target)


if __name__ == "__main__":
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_binarized_omniglot_with_targets()
    conditional_omniglot_data_loader = get_conditional_omniglot_data_loader(
        xtrain, ytrain, 5, "cpu", True
    )
    obs_id, (obs, alphabet) = next(iter(conditional_omniglot_data_loader))
    print(obs_id)
    print(obs.shape)
    print(alphabet.shape)
