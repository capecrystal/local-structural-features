
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


def getFrameCount(fname):
    import gsd.hoomd

    with gsd.hoomd.open(fname, 'rb') as traj:
        return len(traj)


def getParams(neigh_max, lmax=12):
    params = dict(
        amean = dict(neigh_max=neigh_max, lmax=lmax),
        steinhardt_q = dict(neighbors=neigh_max, lmax=lmax), 
        )
    return params


def computeParticleDescriptors(box, positions, params, mode, seed=13):
    import pythia
    import numpy as np

    kwargs = params[mode]
    if mode == 'amean':
        descriptors = pythia.spherical_harmonics.neighbor_average(
            box, positions, negative_m=False, **kwargs)
    elif mode == 'steinhardt_q':
        descriptors = pythia.spherical_harmonics.steinhardt_q(box, positions, **kwargs)
        descriptors[np.isnan(descriptors)] = 0 #0 neighbors will throw a NaN
    else:
        raise NotImplementedError('Unknown descriptor mode {}'.format(mode))
    
    result = descriptors

    if mode == 'amean':
        result = np.abs(result)

    return np.asarray(result, dtype=np.float32)


def globalDescriptors(params, mode, filename, frame_index):
    import gsd.hoomd, gsd.pygsd
    from collections import namedtuple

    FakeBox = namedtuple('FakeBox', ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz'])
    with gsd.hoomd.open(filename, 'rb') as traj:
        frame = traj[frame_index]
        positions = frame.particles.position
        box = frame.configuration.box
    
    box = FakeBox(*box)
   
    return computeParticleDescriptors(box, positions, params, mode)


def makeDataset(params, mode, filename, frame_indices, use_cache=False):

    import os
    import numpy as np
    import pickle
    
    loadName = 'datasets_{}.pkl'.format(mode)
    if os.path.exists(loadName) and use_cache:
        with open(loadName, 'rb') as f:
            return pickle.load(f)
    
    train_inputs = []
    train_outputs = []
    
    for frame_index in frame_indices:
        descriptors = globalDescriptors(params, mode, filename, frame_index)
        train_inputs.append(descriptors)
        train_outputs.append(np.ones((len(descriptors),), dtype=np.uint32)*frame_indices.index(frame_index))

    train_inputs = np.concatenate(train_inputs, axis=0)
    train_outputs = np.concatenate(train_outputs)

    #print(train_inputs.shape, train_outputs.shape)
    
    with open(loadName, 'wb') as f:
        pickle.dump((train_inputs, train_outputs), f)
    
    return (train_inputs, train_outputs)


def get_pca(train_inputs, mode, NPCA=64, plot=True, use_cache=False):

    import sklearn.decomposition
    import os
    import pickle
    import matplotlib, matplotlib.pyplot as pp
    import numpy as np

    loadName = 'pythia_pca_{}.pkl'.format(mode)
    
    if os.path.exists(loadName) and use_cache:
        with open(loadName, 'rb') as f:
            loaded = pickle.load(f)
        pca = loaded['pca']
        tf = loaded['tf']
        return pca, tf
    
    if NPCA is not None and train_inputs.shape[1] > NPCA:
        pca = sklearn.decomposition.PCA(NPCA) #this holds the pca information
        #tf = pca.transform(train_inputs)
        tf = pca.fit_transform(train_inputs)

        if plot:
            pp.figure()
            pp.plot(np.cumsum(pca.explained_variance_ratio_))
            #pp.yscale('log')
            pp.title(mode)
            saveName = 'pca_{}_plot.png'.format(mode)
            pp.savefig(saveName)
    else:
        pca = None
        tf = train_inputs

    with open(loadName, 'wb') as f:
        dumped = dict(pca=pca, tf=tf)
        pickle.dump(dumped, f)

    return pca, tf


def gmmTrainer(tf, Ncomp, icType):

    import sklearn.mixture

    best_ic = np.inf
    best_gmm = None

    for cv_type in ['spherical', 'tied', 'full', 'diag']:
        gmm_ = sklearn.mixture.GaussianMixture(Ncomp, covariance_type=cv_type, n_init=3, init_params='random')
        try:
            gmm_.fit(tf)
        except ValueError as e:
            if str(e).startswith('Fitting the mixture model failed because some components'):
                continue
            else:
                raise

        if icType == 'bic':
            ic = gmm_.bic(tf)
        elif icType == 'aic':
            ic = gmm_.aic(tf)
        else:
            raise RuntimeError('Unknown ic type')

        if ic < best_ic:
            best_gmm = gmm_
            best_ic = ic

    return (Ncomp, best_gmm, best_ic)

def getGMMs(tf, mode, ncomp_max, icType='bic', plot=True, use_cache=False, multiprocessing=True):

    import numpy as np
    import matplotlib, matplotlib.pyplot as pp
    import os
    import pickle
    import loky

    Ncomps = np.round(np.linspace(1, np.sqrt(ncomp_max), 32)**2).astype(np.int32)
    Ncomps = np.unique(Ncomps)
    
    loadName = 'pythia_gmms_{}.pkl'.format(mode)
    
    if os.path.exists(loadName) and use_cache:
        with open(loadName, 'rb') as f:
            loaded = pickle.load(f)
        ics = loaded['ics']
        all_gmms = loaded['all_gmms']
        Ncomps = loaded['Ncomps']
        return Ncomps, all_gmms, ics

    ics = {}

    if multiprocessing:
        with loky.get_reusable_executor(max_workers=4) as p:
            thunks = []
            all_gmms = []
            for N in Ncomps:
                thunks.append(p.submit(gmmTrainer, tf, N, icType))
            for thunk in thunks:
                (k, gmm_, ic) = thunk.result()
                all_gmms.append(gmm_)
                ics[k] = ic
    else:
        all_gmms = []
        for N in Ncomps:
            (k, gmm_, ic) = gmmTrainer(tf, N, icType)
            all_gmms.append(gmm_)
            ics[k] = ic
            
    with open(loadName, 'wb') as f:
        dumped = dict(ics=ics, all_gmms=all_gmms, Ncomps=Ncomps)
        pickle.dump(dumped, f)
            
    if plot:
        pp.figure()
        xs = list(sorted(ics))
        ys = [ics[k] for k in xs]
        pp.plot(xs, ys)
        pp.xlabel('components')
        pp.ylabel(icType)
        saveName = 'gmm_{}_{}_plot.png'.format(mode, icType)
        pp.savefig(saveName)
            
    return Ncomps, all_gmms, ics 


def labeling_particles(params, mode, pca, gmm, clusters, filename_2, frame_list, use_cache=False):

    # filename_2 can but is not required to be the same as filename
    # frame_list can but is not required to be the same as frame_indices

    import pickle
    import os
    import sklearn.decomposition
    import gsd.hoomd, gsd.pygsd
    import numpy as np

    loadName = 'gmm{}_label_{}.pkl'.format(clusters, mode)
    if os.path.exists(loadName) and use_cache:
        with open(loadName,'rb') as f:
            return pickle.load(f)

    gmm_label = []
    for frame_index in frame_list:
        descriptors = globalDescriptors(params, mode, filename_2, frame_index)

        if pca is not None:
            descriptors = pca.transform(descriptors)
        labels = gmm.predict(descriptors)
        gmm_label.append(labels)

    gmm_label = np.array(gmm_label)

    with open(loadName, 'wb') as f:
        pickle.dump(gmm_label, f)

    return gmm_label


def centercrystal(simulationbox, particlepositions):
    """
    Recentering the positions of all particles relative to COM, because otherwise they are split across the "box"
    Algorithm projects points along 3 orthogonal cylinders, finds COM, projects back into real space
    Useful resource w/ similar algorithm: https://www.cs.drexel.edu/~david/Papers/Bai_JGT.pdf
    
    inputs:
    simulationbox= 6x1 array of the box dimensions [Lx Ly Lz xy xz yz]
    particlepositions= the positions of the particles from the snapshot, #particles x 3 dimensions (xyz) array
    
    """    
    import numpy as np

    simbox = [simulationbox[3]-simulationbox[0], simulationbox[4]-simulationbox[1], simulationbox[5]-simulationbox[2]]
    theta = (particlepositions / simbox + .5) * 2 * np.pi
    sums = np.sum(np.exp(1j * theta), axis = 0)
    fractions = np.angle(sums) / 2 / np.pi
    fractions %= 1.
    fractions -= .5
    delta = fractions * simbox
    pos_CM = np.copy(particlepositions)
    pos_CM[:] -= delta[np.newaxis, :]
    
    # wrap particles back into box
    pos_CM[pos_CM[:, 0] > simulationbox[0]/2] -= [simulationbox[0], 0, 0]
    pos_CM[pos_CM[:, 1] > simulationbox[1]/2] -= [0, simulationbox[1], 0]
    pos_CM[pos_CM[:, 2] > simulationbox[2]/2] -= [0, 0, simulationbox[2]]
    pos_CM[pos_CM[:, 0] < -simulationbox[0]/2] += [simulationbox[0], 0, 0]
    pos_CM[pos_CM[:, 1] < -simulationbox[1]/2] += [0, simulationbox[1], 0]
    pos_CM[pos_CM[:, 2] < -simulationbox[2]/2] += [0, 0, simulationbox[2]]
    
    return pos_CM


def sim_coloring_gtar(gmm_label, frame_list, fname, filename, centering=True):
    """
    INPUTS:
    gmm - gmm labels for frames in frame_list
    frame_list - list of frames for which gmm values were computed for each particle
    fname - name of tar file to be created
    filename - directory + name of original simulation file
    """
    import gtar
    import gsd, gsd.hoomd

    with gtar.GTAR(fname, 'w') as gtar_traj:
        for frame_index in frame_list:
            # to color by component (best for 3 or more components)

            labels = gmm_label[frame_list.index(frame_index)]

            with gsd.hoomd.open(filename, 'rb') as traj:
                frame = traj[frame_index]
                box = frame.configuration.box
                if centering:
                    positions = centercrystal(box, frame.particles.position)
                else:
                    positions = frame.particles.position


            colors = np.ones((len(positions), 4))

            #assigning fixed colors for 9 clusters
            for (i,x) in enumerate(labels):
                if x==0:
                    colors[i, :3] = np.float32(np.divide([246, 100, 100], 255)) #red
                if x==1:
                    colors[i, :3] = np.float32(np.divide([184, 165, 239], 255)) #lilac
                if x==2:
                    colors[i, :3] = np.float32(np.divide([48, 116, 233], 255)) #blue
                if x==3:
                    colors[i, :3] = np.float32(np.divide([246, 173, 100], 255)) #orange
                if x==4:
                    colors[i, :3] = np.float32(np.divide([79, 166, 114], 255)) #green                
                if x==5:
                    colors[i, :3] = np.float32(np.divide([148, 148, 148], 255)) #gray 
                if x==6:
                    colors[i, :3] = np.float32(np.divide([171, 43, 161], 255)) #magenta
                if x==7:
                    colors[i, :3] = np.float32(np.divide([204, 204, 0], 255)) #yellow
                if x==8:
                    colors[i, :3] = np.float32(np.divide([147, 255, 245], 255)) #cyan

            colors = np.clip(colors, 0, 1)

            #writing positions, box, and colors to gtar file
            frame_num = frame_list.index(frame_index)
            gtar_traj.writePath('frames/{}/position.f32.ind'.format(frame_num), positions)
            gtar_traj.writePath('frames/{}/box.f32.uni'.format(frame_num), box)
            gtar_traj.writePath('frames/{}/color.f32.ind'.format(frame_num), colors)
            gtar_traj.writePath('frames/{}/cluster.u32.ind'.format(frame_num), x)

def sim_coloring_gsd(gmm_label, frame_list, fname, filename, centering=True):
    """
    INPUTS:
    gmm - gmm labels for frames in frame_list
    frame_list - list of frames for which gmm values were computed for each particle
    fname - name of tar file to be created
    filename - directory + name of original simulation file
    """
    import gsd, gsd.hoomd

    t = gsd.hoomd.open(fname, 'wb')
    types = np.arange(0, np.amax(gmm_label)+1)

    for frame_index in frame_list:

        labels = gmm_label[frame_list.index(frame_index)]

        with gsd.hoomd.open(filename, 'rb') as traj:
            snap = traj[frame_index]
            box = snap.configuration.box

            if centering:
                positions = centercrystal(box, snap.particles.position)
                snap.particles.position = positions

        snap.particles.typeid = labels
        snap.particles.types = [str(i) for i in types]

        t.append(snap)

def rdf(filename, smoothing=2, rdf_frame=-1):

    import gsd.hoomd
    import gsd.pygsd
    
    f = gsd.pygsd.GSDFile(open(filename, 'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)
    snap = t[rdf_frame]
    pos = snap.particles.position
    box = snap.configuration.box

    import numpy as np
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d

    bins = 200
    rdf_rmax = 4.0
    dr = 0.025
    
    import freud
    rdf = freud.density.RDF(bins, rdf_rmax, dr)
    rdf = rdf.compute(system=[box,pos])
    rdf_rdf = rdf.rdf
    rdf_r = np.linspace(dr, rdf_rmax, bins)
    yi = gaussian_filter1d(rdf.rdf, smoothing)

    #extracting minima
    min_i = signal.find_peaks(-yi)
    max_i = signal.find_peaks(yi)
    min_i = min_i[0]
    max_i = max_i[0]

    nn_radius = rdf_r[min_i[np.argmax(min_i > np.amin(max_i))]]
  
    return nn_radius

def NNcounter(filename, frame_list, nn_radius, use_cache=False):
    # inputs:
    # filename: str of simulation name
    # frame_list: list of simulation frames to calculate NN's for
    # use_cache: True if want to load in previous data
    
    # outputs:
    # NN_all_frames: 2D array for NN by particle by frame
    
    import os
    import gsd.hoomd
    import numpy as np
    import freud
    import pickle

    loadName = 'NN_data.pkl'
    if os.path.exists(loadName) and use_cache:
        with open(loadName,'rb') as f:
            return pickle.load(f)
    
    # initialize our output files
    NN_label = []
    
    # loop through frames
    for frame_index in frame_list:
        with gsd.hoomd.open(filename, 'rb') as traj:
            frame = traj[frame_index]
            box = frame.configuration.box
            pos = frame.particles.position
        
        # further initialization of our data structures
        num_particles = len(pos)
        nn_matrix = [[None for _ in range(num_particles)] for _ in range(num_particles)]
        
        # calculate neighbor distances using freud
        q = freud.AABBQuery(box, pos)
        qr = q.query(pos, {'r_max': nn_radius, 'exclude_ii': True})
        for (p1, p2, dist) in qr:
            nn_matrix[p1][p2] = dist
            
        # calculate number of nearest neighbors for each particle
        NN = np.zeros(num_particles)
        for p in range(num_particles):
            NN[p] = len(np.nonzero(nn_matrix[p])[0])
        NN_label.append(NN)

    NN_label = np.array(NN_label)

    # saving data
    with open(loadName, 'wb') as f:
        pickle.dump(NN_label, f)
    
    return NN_label


def NN_coloring(mode, gmm_label, NN_label, filename):
    # generating plots for each cluster
    import numpy as np
    import numpy.ma as ma
    import matplotlib, matplotlib.pyplot as pp
    import matplotlib.ticker as ticker

    fontsize = 25

    clusters = np.amax(gmm_label) + 1

    for gmm_val in range(clusters):
        counts = np.ndarray(shape=(NN_label.shape[0], int(np.amax(NN_label)+1)), dtype='int64')
        for frame in range(NN_label.shape[0]):
            NN_filt = ma.masked_array(NN_label[frame], mask = gmm_label[frame] != gmm_val)
            NN_filt = ma.compressed(NN_filt)
            NN_filt = np.array(NN_filt, dtype='int64')
            counts_ = np.bincount(NN_filt, minlength = int(np.amax(NN_label)+1))
            counts[frame] = counts_

        counts = np.transpose(counts)

        fig,ax = pp.subplots(figsize=np.shape(counts), dpi = 200)

        if gmm_val ==0: #red
            plot = ax.imshow(counts,cmap="Reds",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')
        if gmm_val ==1: #purple
            plot = ax.imshow(counts,cmap="Purples",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')
        if gmm_val ==2: #blue
            plot = ax.imshow(counts,cmap="Blues",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')
        if gmm_val ==3: #orange
            plot = ax.imshow(counts,cmap="Oranges",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')
        if gmm_val ==4: #green
            plot = ax.imshow(counts,cmap="Greens",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')
        if gmm_val ==5: #gray
            plot = ax.imshow(counts,cmap="Greys",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')        
        if gmm_val ==6: #magenta
            plot = ax.imshow(counts,cmap="RdPu",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')        
        if gmm_val ==7: #yellow
            plot = ax.imshow(counts,cmap="YlOrBr",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')        
        if gmm_val ==8: #cyan
            plot = ax.imshow(counts,cmap="BuGn",origin='lower',aspect=np.shape(counts)[1]/np.shape(counts)[0],extent=(0, np.shape(counts)[1], 0, np.shape(counts)[0]), interpolation = 'nearest')        


        cbar = fig.colorbar(plot, ax=ax, extend='neither', fraction=0.046, pad=0.05, ticks=[])

        pp.xlabel('Time', fontsize=fontsize)
        pp.ylabel('Coordination number', fontsize=fontsize)
        pp.tick_params(axis='both', which='major', labelsize=fontsize)
        pp.tick_params(axis='both', which='minor', labelsize=fontsize)

        #setting x ticks
        pp.xticks(np.arange(0, np.shape(counts)[1], step=1))
        ax.xaxis.set_major_formatter(ticker.NullFormatter())

        #setting y ticks
        pp.yticks(np.arange(0, np.shape(counts)[0], step=1))
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(np.shape(counts)[0]) + 0.5))
        y_labels = list(map(str, range(np.shape(counts)[0])))
        ax.yaxis.set_minor_formatter(ticker.FixedFormatter(y_labels))
        for label in ax.yaxis.get_minorticklabels()[1::2]:
            label.set_visible(False)

        saveName = 'NN_cluster_{}_{}.pdf'.format(gmm_val, mode)
        pp.savefig(saveName,format='pdf', transparent=True, bbox_inches='tight')


def tsne_transform(tf, mode, use_cache=False):
    #inputs:
    # tf: pca transformed data
    # mode: descriptor parameter, ex: 'amean'
    # use_cache: set to True to load in saved data
    #
    #outputs: tsne transformed data, which has array shape len(descriptors) x 2, since it is a 2D visualization

    import sklearn.manifold
    import pickle
    import os

    loadName = 'tsne_{}.pkl'.format(mode)
    
    if os.path.exists(loadName) and use_cache:
        with open(loadName, 'rb') as f:
            return pickle.load(f)
            
    tsne = sklearn.manifold.TSNE(init='pca')
    tsne_tf = tsne.fit_transform(tf)
    
    with open(loadName, 'wb') as f:
        pickle.dump(tsne_tf, f)
    
    return tsne_tf


def TSNE_plot(tsne_tf, train_outputs, gmm_label, frame_list):

    import matplotlib, matplotlib.pyplot as pp

    fig_scale = 1.

    # setting axes
    buff = 5
    l = min(tsne_tf[:,0]) - buff
    r = max(tsne_tf[:,0]) + buff
    b = min(tsne_tf[:,1]) - buff
    t = max(tsne_tf[:,1]) + buff

    color = [np.float32(np.divide([246, 100, 100], 255)), #red
             np.float32(np.divide([184, 165, 239], 255)), #lilac
             np.float32(np.divide([48, 116, 233], 255)), #blue
             np.float32(np.divide([246, 173, 100], 255)), #orange
             np.float32(np.divide([79, 166, 114], 255)), #green
             np.float32(np.divide([148, 148, 148], 255)), #gray 
             np.float32(np.divide([171, 43, 161], 255)), #magenta
             np.float32(np.divide([204, 204, 0], 255)), #yellow
             np.float32(np.divide([147, 255, 245], 255)), #cyan
             np.float32(np.divide([0, 0, 0], 255)) #black
            ]

    fontsize = 16
    fig_scale = 1.
    fig = pp.figure(figsize=(8*fig_scale, 6*fig_scale), dpi=200)

    clusters = np.amax(gmm_label) + 1
    
    for frame in frame_list:
        #coloring TSNE data
        filt = train_outputs == frame_list.index(frame)
        x = tsne_tf[filt, 0]
        y = tsne_tf[filt, 1]
        for cluster in range(clusters):
            gmm_ = gmm_label[frame_list.index(frame)]
            gmm_filt = gmm_ == cluster
            pp.scatter(x[gmm_filt], y[gmm_filt], color=color[cluster], alpha=.45)

    axes = pp.gca()
    axes.set_xlim(left=l, right=r)
    axes.set_ylim(bottom=b, top=t)
    pp.title('T-SNE')
    saveName = 'tsne_clusters_{}.pdf'.format(clusters)
    pp.savefig(saveName, format='pdf', transparent=True, bbox_inches='tight')



