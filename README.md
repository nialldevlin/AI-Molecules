# AI Molecules

## Structure

The structure of this project is as follows

### data

This folder stores training data

### src

Where the goods are stored

### README.md

See README.md for more information

## Networks
### Linear VAE

A variational autoencoder using linear layers. Run `train_linear_vae.py` to
train the network and verify it by predicting the first sample. To make it
useful, change the filename to the location of the XDATCAR containing your
training data, and use the generate function at the bottom of the file to
generate new data. There are two networks in this folder, a pure linear vae,
`linear_vae.py` and a convolutional vae, 'conv_vae.py'. To switch between them,
change the import at the top from linear_vae to conv_vae. Change the model save
path as well, and the line that creates the model, `model = ConvVae(device,
750)`. Depending on the useage, this file will save the original sample to
actual/XDATCAR, the predicted sample to predicted/XDATCAR, and generated samples
to generated/XDATCAR

### Graph VAE

These autoencoders use Pytorch Geometric and process the XDATCAR as a pointcloud.
Run `train_graph_vae.py` to train and test the network. It will then display the
3d structure of one of the original data points and it's prediction.

### predictive_networks

This folder contains several types of networks to predict new molecular data
in a time series.

### vae_networks

This folder contians older attempts at the graph vae. They may still be useful
for reference.



