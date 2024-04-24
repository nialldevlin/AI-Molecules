import numpy as np
from torch.utils.data import Dataset
import os
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data


class MoleculeFrames:
    def __init__(self, dir, k,  header_sep='Direct'):
        self.header_sep = header_sep
        self.precision = 16

        # Get data from the file
        raw_data = []
        self.headers = []
        for file in os.listdir(dir):
            d = []
            header = ""
            with open(dir + '/' + file, 'r') as f:
                notHeader = False
                for line in f:
                    if notHeader:
                        d.append([float(x) for x in line.split()])
                    elif line.startswith(self.header_sep):
                        notHeader = True
                        header += line
                    elif not notHeader:
                        header += line
            self.headers.append(header)
            raw_data.append(np.array(d))

        # Pack with zeros
        # max_length = max(len(lst) for lst in raw_data)
        # self.data = np.zeros((len(raw_data), max_length), dtype=int)

        # for i, lst in enumerate(raw_data):
        #     self.data[i, :len(lst)] = lst

        # self.headers = np.array(self.headers)

        # # Normalize
        # self.mean = self.data.mean()
        # self.std = self.data.std()
        # self.data = (self.data - self.mean) / self.std
        self.data = raw_data

        self.knn_data = []
        for d in self.data:
            t = torch.tensor(d, dtype=torch.float)
            edge_index = knn_graph(t, k, loop=False)

            # Create a PyTorch Geometric Data object
            knn = Data(x=t, edge_index=edge_index)
            self.knn_data.append(knn)

    def unNormalize(self, features=None):
        # Reshape the data and un normalize it
        if features is None:
            newFeatures = self.data
        else:
            newFeatures = features

        newFeatures = newFeatures * self.std + self.mean
        return newFeatures

    def _format_float(self, f):
        # Format the float with the correct number decimal places
        formatted = "{:.{}f}".format(f, self.precision)
        if not formatted.startswith('-'):
            formatted = " " + formatted
        return formatted

    def toPOSCAR(self, filename, header, data):
        with open(filename, 'w') as f:
            f.write(header)
            for row in data:
                row_fmt = map(self._format_float, row)
                line = " " + " ".join(row_fmt) + "\n"
                f.write(line)


class MoleculeDataset(Dataset):
    def __init__(self, data, sequence_len):
        self.s_len = sequence_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.s_len

    def __getitem__(self, idx):
        return self.data[idx:idx + self.s_len], self.data[idx + self.s_len]


class VAEMoleculeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
