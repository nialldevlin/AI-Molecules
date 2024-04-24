import numpy as np
from torch.utils.data import Dataset
import os
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data


class MoleculeFrames:
    def __init__(self, dir, header_sep='Direct'):
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
            raw_data.append(d)

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
