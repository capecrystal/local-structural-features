
The following code accompanies the "Local structural features elucidate crystallization of complex structures" (https://arxiv.org/abs/2401.13765) preprint and utilizes the descriptor/methods from Spellings and Glotzer (https://doi.org/10.1002/aic.16157).

The following python libraries will need to be installed in order to run this code:

numpy
scipy
pythia
freud
gsd
gtar
sklearn
pickle
loky
matplotlib


This code can be used in conjunction with simulation data deposited to the Materials Data Facility (MDF) at the following link: <>. Hyperparameters used for each structure are included as a training.json file.

The local_Ylms.py file contains all the functions for creating local descriptors and performing unsupervised learning to build local order parameters by clustering particles. It also contains additional functions for plotting coordination number histograms as well as performing t-distributed stochastic neighbor embedding (tSNE) on clustered data.

The train_order_params.py file runs the functions in the local_Ylms.py file and applies them to the MDF data. This script will automatically generate folders grouped by descriptor type (Steinhardt parameter and Spellings' spherical harmonics descriptor) as well as by number of Gaussian mixture model (GMM) clusters.

For more information on the method for crystal identification (i.e., building global order parameters) by Spellings and Glotzer, look at:
- pythia GitHub: https://github.com/glotzerlab/pythia
- jupyter notebook: https://aiche.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Faic.16157&file=aic16157-sup-0001-suppinfo1.html




