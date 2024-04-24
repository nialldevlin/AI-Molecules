import numpy as np
from torch.utils.data import Dataset


def loadXDATCAR(filename, separator="Direct configuration="):
    """Reads an XDATCAR file and converts it into a 3D NumPy array.

    This function parses an XDATCAR file extracting the atomic configurations
    stored over different time frames. Each frame is separated in the file by
    a specified separator string. The function also collects the file's header
    information, which is typically present before the first occurrence of
    the separator. The numerical data representing atomic positions or other
    variables in each frame are converted into floating point values and stored
    in a 3D NumPy array.

    Args:
        filename (str): The path to the XDATCAR file.
        separator (str): A string used to identify the start of each new frame
                         within the file. Defaults to "Direct configuration=".

    Returns:
        tuple: A tuple containing two elements:
            - numpy.ndarray: A 3D array where each sub-array represents one
                             frame of data from the XDATCAR file.
            - str: A string containing the header of the file, i.e., all
                   content before the first separator.
    """
    data = []
    header = ""
    with open(filename, 'r') as f:
        section = []
        notHeader = False
        for line in f:
            if notHeader:
                if line.startswith(separator):
                    data.append(section)
                    section = []
                else:
                    section.append([float(x) for x in line.split()])
            elif line.startswith(separator):
                notHeader = True
            elif not notHeader:
                header += line
        data.append(section)

    data = np.array(data)
    return (data, header)


def _format_float(self, f):
    """Formats a float to a string with 8 decimal places.

    This internal method ensures that floats are formatted consistently with
    8 decimal places. It also aligns the numbers by prepending a space to
    positive numbers to align with the space taken up by the negative sign
    in negative numbers.

    Args:
        f (float): The float number to format.

    Returns:
        str: The formatted float as a string with 8 decimal places. Positive numbers
             are prepended with a space for alignment purposes.
    """
    formatted = f"{f:.8f}"
    if not formatted.startswith('-'):
        formatted = " " + formatted
    return formatted


def toXDATCAR(filename, data, header, separator="Direct configuration="):
    """Writes a 3D NumPy array into an XDATCAR file format.

    This function takes a 3D NumPy array where each sub-array represents one frame
    of data, and writes it to a file in the XDATCAR format, commonly used in VASP
    materials simulations. Each frame is preceded by a separator line indicating
    the frame number, starting from 1.

    Args:
        filename (str): The path where the XDATCAR file will be written.
        data (numpy.ndarray): The 3D array containing the data to be written. Each
                              sub-array corresponds to a frame.
        header (str): The header content to be written at the top of the file.
        separator (str): A string used to separate frames in the file. Defaults to
                         "Direct configuration=".
    """
    with open(filename, 'w') as f:
        f.write(header)

        for i in range(data.shape[0]):
            index = str(i + 1)
            spaces = 6 - len(index)
            f.write(separator + (' ' * spaces) + index + "\n")

            for row in data[i]:
                row_fmt = map(_format_float, row)
                line = "  " + " ".join(row_fmt) + "\n"
                f.write(line)


class LinearVAEMoleculeDataset(Dataset):
    """
    A dataset class for a Pytorch Linear Variational Autoencoder that handles molecule data,
    providing normalization and reshaping utilities to prepare data for modeling.

    This class normalizes the input data by subtracting the mean and dividing by the
    standard deviation of each feature. It also reshapes the data into a flat structure
    suitable for input into linear models. Functions are provided to revert these
    transformations.

    Attributes:
        data (numpy.ndarray): The normalized and flattened dataset.
        shape (tuple): Original shape of the data before flattening.
        mean (numpy.ndarray): Mean of each feature across the dataset.
        std (numpy.ndarray): Standard deviation of each feature across the dataset.

    Args:
        data (numpy.ndarray): The original data array, where each row represents a
                              data point and each column a feature.

    Methods:
        __len__(): Returns the number of items in the dataset.
        __getitem__(idx): Retrieves the normalized and flattened data at index `idx`.
        unNormalizeFeatures(features): Reverts normalization on a flat array and reshapes it.
        unNormalize(): Reverts normalization on the entire dataset.
    """

    def __init__(self, data):
        """
        Initializes the dataset instance, normalizes the data, and flattens it.

        Args:
            data (numpy.ndarray): The raw data to be processed.
        """
        self.data = data
        self.shape = self.data.shape

        # Normalize
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.data = (self.data - self.mean) / self.std

        # Flatten
        self.data = self.data.reshape(self.shape[0], -1)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the dataset item at the specified index after normalization and flattening.

        Args:
            idx (int): The index of the item.

        Returns:
            numpy.ndarray: The normalized and flattened data at the specified index.
        """
        return self.data[idx]

    def unNormalizeFeatures(self, features):
        """
        Reverses the normalization and reshaping process for a set of features.

        Args:
            features (numpy.ndarray): The normalized and flattened features to be un-normalized.

        Returns:
            numpy.ndarray: The original shape and scale of the features.
        """
        return features.reshape((-1,) + self.shape[1:]) * self.std + self.mean

    def unNormalize(self):
        """
        Reverses the normalization for the entire dataset, returning it to its original scale and shape.

        Returns:
            numpy.ndarray: The dataset in its original shape and scale.
        """
        return self.unNormalizeFeatures(self.data)
