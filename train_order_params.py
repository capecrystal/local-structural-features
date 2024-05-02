import pythia
import numpy as np
import numpy.ma as ma
import gsd, gsd.hoomd, gsd.pygsd
import sklearn.decomposition
import sklearn.mixture
import sklearn.manifold
import pickle
import os
import json
import loky
import freud
from collections import namedtuple
import matplotlib, matplotlib.pyplot as pp
import matplotlib.ticker as ticker
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from local_Ylms import *

data_dir = '/Users/mayamartirossyan/Documents/research/manuscripts/crystal growth/Code/Data_test' 
# put in your Data directory as data_dir
# needs to be a global path

structures = os.listdir(data_dir)
try:
    structures.remove('.DS_Store')

for struc in structures:

    struc_dir = '{}/{}'.format(data_dir, struc)

    dump_file = struc_dir+'/dump.gsd'
    training_file = struc_dir+'/training.json'

    with open(training_file, 'r') as f:
        training_params = json.load(f)

        frame_indices = training_params['training_frames']
        frame_list = training_params['training_frames']
        #frame_list = list(range(0,getFrameCount(dump_dir)))
        
        descriptor_params = training_params['descriptor_params']
        neigh_max = descriptor_params['neigh_max']
        lmax = descriptor_params['lmax']

        clusters = training_params['gmm_clusters']

        f.close()

    outputs_folder = struc_dir+'/model_outputs' #change if performing multiple training rounds

    if not os.path.isdir(outputs_folder):
        os.mkdir(outputs_folder)
    os.chdir(outputs_folder)

    params = getParams(neigh_max, lmax)
    NPCA = 64
    cluster_range = True
    NN = True
    tSNE = True

    for mode in params:

        mode_folder = outputs_folder+'/{}'.format(mode)
        if not os.path.isdir(mode_folder):
            os.mkdir(mode_folder)
        os.chdir(mode_folder)

        dataset = makeDataset(params, mode, dump_file, frame_indices, use_cache=True)
        (train_input, train_output) = dataset
        pca, tf = get_pca(train_input, mode, NPCA, plot=True, use_cache=True)

        if tSNE:
            tsne_tf = tsne_transform(tf, mode, use_cache=True)

        if NN:
            os.chdir(outputs_folder)
            nn_radius = rdf(dump_file)
            NN_label = NNcounter(dump_file, frame_list, nn_radius, use_cache=True)
            os.chdir(mode_folder)

        # creating GMM model, generating labels for particles, applying color to simulations
        if cluster_range:
            ## getGMMs executes gmmTrainer over range of cluster sizes, from 1 to clusters
            Ncomps, gmms, _ = getGMMs(tf, mode, clusters, icType='bic', plot=True, use_cache=True, multiprocessing=True)

            for cluster_num in range(2, clusters+1):

                cluster_folder = mode_folder+'/gmm{}'.format(cluster_num)
                if not os.path.isdir(cluster_folder):
                    os.mkdir(cluster_folder)
                os.chdir(cluster_folder)

                gmm = gmms[cluster_num-1]
                gmm_label = labeling_particles(params, mode, pca, gmm, cluster_num, dump_file, frame_list, use_cache=True)
                fname = 'gmm{}_colored_frames_{}.tar'.format(cluster_num, mode)
                if not os.path.isfile(fname):
                    sim_coloring_gtar(gmm_label, frame_list, fname, dump_file, centering=True)
                fname = 'gmm{}_colored_frames_{}.gsd'.format(cluster_num, mode)
                if not os.path.isfile(fname):
                    sim_coloring_gsd(gmm_label, frame_list, fname, dump_file, centering=True)

                if NN:
                    NN_coloring(mode, gmm_label, NN_label, dump_file)

                if tSNE:
                    TSNE_plot(tsne_tf, train_output, gmm_label, frame_list)

        else:

            cluster_folder = mode_folder+'/gmm{}'.format(clusters)
            if not os.path.isdir(cluster_folder):
                os.mkdir(cluster_folder)
            os.chdir(cluster_folder)

            ## gmmTrainer executes gmm code for only one cluster size = clusters
            loadName = 'pythia_gmm{}_{}.pkl'.format(clusters, mode)
            if not os.path.exists(loadName):
                (Ncomps, gmm, ics) = gmmTrainer(tf, clusters, 'bic') # comment line below if uncommenting this one
                Ncomps = np.asarray([Ncomps]).astype(np.int32)
                with open(loadName, 'wb') as f:
                    dumped = dict(ics=ics, gmm=gmm, Ncomps=Ncomps)
                    pickle.dump(dumped, f)
            else:
                with open(loadName, 'rb') as f:
                    loaded = pickle.load(f)
                    gmm = loaded['gmm']
                    Ncomps = loaded['Ncomps']

            gmm_label = labeling_particles(params, mode, pca, gmm, clusters, dump_file, frame_list, use_cache=True)
            fname = 'gmm{}_colored_frames_{}.tar'.format(clusters, mode)
            if not os.path.isfile(fname):
                sim_coloring_gtar(gmm_label, frame_list, fname, dump_file, centering=True)
            fname = 'gmm{}_colored_frames_{}.gsd'.format(clusters, mode)
            if not os.path.isfile(fname):
                sim_coloring_gsd(gmm_label, frame_list, fname, dump_file, centering=True)

            if NN:
                NN_coloring(mode, gmm_label, NN_label, dump_file)

            if tSNE:
                TSNE_plot(tsne_tf, train_output, gmm_label, frame_list)


        # saving analysis parameters
        os.chdir(mode_folder)
        loadName = 'analysis_params_{}.json'.format(mode)
        if not os.path.isfile(loadName):

            analysis_dict = {}

            analysis_dict['descriptor_params'] = params[mode]
            analysis_dict['training_frames'] = frame_indices
            analysis_dict['dataset_dimensions'] = dataset[0].shape
            analysis_dict['NPCA'] = NPCA
            analysis_dict['gmm_ncomps'] = np.ndarray.tolist(Ncomps)
            analysis_dict['gmm_clusters'] = clusters
            analysis_dict['coloring_frames'] = frame_list

            analysis_dict['cluster_range'] = cluster_range
            analysis_dict['NN'] = NN
            analysis_dict['tSNE'] = tSNE

            with open(loadName, 'w') as json_file:
                json.dump(analysis_dict, json_file)

