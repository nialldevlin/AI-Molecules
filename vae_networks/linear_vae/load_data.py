import numpy as np
from torch.utils.data import Dataset

class XDATCAR:
    def __init__(self, filename, separator="Direct configuration="):
        self.separator = separator
        self.filename = filename

        data = []
        header = ""
        with open(self.filename, 'r') as f:
            section = []
            notHeader = False  # Flag if we are in the header
            for line in f:
                if notHeader:
                    # Every time you see the separator, a new section starts
                    if line.startswith(separator):
                        data.append(section)
                        section = []
                    else:
                        # Add each line to current section
                        section.append([float(x) for x in line.split()])
                elif line.startswith(separator):
                    notHeader = True  # Set the flag to indicate header is over
                elif not notHeader:
                    header += line
            data.append(section)

        # Make it numpy array
        self.data = np.array(data)
        self.header = header

class DataHandler:
    def __init__(self, filename, split, flatten=True, transpose=False):
        self.header = ""
        self.precision = 8

        # Get data from the file

        # Normalize
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.data = (self.data - self.mean) / self.std

        # # Get velocity with time derivative
        # self.vel = np.diff(self.data, n=1, axis=0, prepend=np.nan)
        # self.vel[np.isnan(self.vel)] = 0

        # # Get acceleration with time derivative
        # self.acc = np.diff(self.vel, n=1, axis=0, prepend=np.nan)
        # self.acc[np.isnan(self.acc)] = 0

        # # Stack it all in one array
        # self.data = np.stack([self.data, self.vel, self.acc], axis=-1)

        self.shape = self.data.shape

        # Transpose each feature
        self.transpose = transpose
        if self.transpose:
            self.data = np.transpose(self.data, (0, 2, 1))

        # Flatten each feature
        self.flatten = flatten
        if self.flatten:
            self.data = self.data.reshape(self.shape[0], -1)

        # Split into training and test data
        # If split is 100%, test data is empty
        self.split = split
        if split == 1:
            self.train_data = self.data
            self.test_data = np.array([])
        else:
            split_idx = int(self.split * len(self.data))
            self.train_data = self.data[:split_idx]
            self.test_data = self.data[split_idx:]

    def unNormalizeFeatures(self, features):
        # Reshape the data and un normalize it
        newFeatures = features
        if self.transpose:
            newFeatures = np.transpose(newFeatures, (0, 2, 1))
        if self.flatten:
            newFeatures = newFeatures.reshape((-1,) + self.shape[1:])
        # newFeatures = newFeatures[:,:,:,0]
        newFeatures = newFeatures * self.std + self.mean
        return newFeatures

    def unNormalizeFeature(self, feature):
        # Reshape the data and un normalize it
        newFeature = feature
        if self.transpose:
            newFeature = np.transpose(newFeature, (1, 0))
        if self.flatten:
            newFeature = newFeature.reshape((-1,) + self.shape[1:])
        # newFeatures = newFeatures[:,:,:,0]
        newFeature = newFeature * self.std + self.mean
        return newFeature

    def unNormalize(self):
        return self.unNormalizeFeatures(self.data)

    def _format_float(self, f):
        # Format the float with 8 decimal places
        formatted = f"{f:.8f}"
        # Adjust for negative numbers: if the number is not negative, prepend a space for alignment
        if not formatted.startswith('-'):
            formatted = " " + formatted
        return formatted

    def toXDATCAR(self, data, filename):
        with open(filename, 'w') as f:
            f.write(self.header)

            for i in range(data.shape[0]):
                index = str(i + 1)
                spaces = 6 - len(index)
                f.write(self.separator + (' ' * spaces) + index + "\n")

                for row in data[i]:
                    row_fmt = map(self._format_float, row)
                    line = "  " + " ".join(row_fmt) + "\n"
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
