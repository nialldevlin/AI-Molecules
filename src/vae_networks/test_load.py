from load_poscar import *
import os

mfs = []
dir = 'MG_data'
for file in os.listdir(dir):
    d = MoleculeFrame(dir + '/' + file, 1)
    mfs.append(VAEMoleculeDataset(d.data))
print(mfs[0][0])
