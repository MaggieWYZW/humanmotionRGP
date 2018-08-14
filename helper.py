import sys
sys.path.insert(0, '/home/yw440/MXGPY/mxgpy')

import numpy as np
from mxgpy.rgp.rgp import RGP 
from mxgpy.kernel.rbf import RBF
import mxnet as mx
ctx = mx.gpu()
mx.Context.default_ctx = ctx

import scipy.io
from pylab import *

def extract_independent_bones(data, labels):
    def convert_data_to_list(data, labels ):
        return [data[np.where(labels[:,i]==1)[0]] for i in range(labels.shape[1])]

    ## This is the correct order
    bones = ['root', 'lfemur', 'ltibia', 'lfoot', 'ltoes', 'rfemur', 'rtibia', 'rfoot', 'rtoes', 
             'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head', 
             'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb']
    idxs = [6, 3, 1, 2, 1, 3, 1, 2, 1,
            3, 3, 3, 3, 3, 3, 
            2, 3, 1, 1, 2, 1, 2, 
            2, 3, 1, 1, 2, 1, 2 ]
    
    data_list = []
    for i in range(np.shape(bones)[0]):
        begin = int(sum(idxs[:i]))
        end = int(begin + idxs[i])
        data_list.append(convert_data_to_list(data[:,begin:end], labels))
    
    return data_list


def normalise_data(data_list):
    data_out_list, data_mean_list, data_std_list = [], [], []
    for data in data_list:
        data_mean_list.append(data.mean(0))
        data_std_list.append(data.std(0))
        data_out_list.append( (data-data.mean(0))/(data.std(0)+1e-5))
        
    return data_out_list, data_mean_list, data_std_list

def reverse_normalise_data(norm_data, data_mean, data_std):
    return norm_data*data_std+data_mean

def convert_data_to_list(data, labels ):
    return [data[np.where(labels[:,i]==1)[0]] for i in range(labels.shape[1])]


# Function to write .amc motion file
def write_amc(predictions, filename):
    import numpy as np
    
    f = open(filename+'.amc', 'w')
    f.write(r"#!OML:ASF F:\VICON\USERDATA\INSTALL\rory3\rory3.ASF"+'\n')
    f.write(':FULLY-SPECIFIED\n')
    f.write(':DEGREES\n')
    
    ## This is the correct order
    bones = ['root', 'lfemur', 'ltibia', 'lfoot', 'ltoes', 'rfemur', 'rtibia', 'rfoot', 'rtoes', 
             'lowerback', 'upperback', 'thorax', 'lowerneck', 'upperneck', 'head', 
             'lclavicle', 'lhumerus', 'lradius', 'lwrist', 'lhand', 'lfingers', 'lthumb',
            'rclavicle', 'rhumerus', 'rradius', 'rwrist', 'rhand', 'rfingers', 'rthumb']
    idxs = [6, 3, 1, 2, 1, 3, 1, 2, 1,
            3, 3, 3, 3, 3, 3, 
            2, 3, 1, 1, 2, 1, 2, 
            2, 3, 1, 1, 2, 1, 2 ]
    
    for i in range(predictions.shape[0]):
        f.write(str(i+1)+'\n')
        for j,bone in enumerate(bones):
            begin = int(sum(idxs[:j]))
            end = int(begin + idxs[j])
            temp = list(map(lambda x: str(x), predictions[i,begin:end]))
            f.write(bone+' '+' '.join(temp)+'\n')
    
    f.close()
    return 


# Function to show 3D image of a human given the data as an array
def gen_frames(data, data_mean, data_std, skel, imgpath):
    import os
    import GPy
    a = np.zeros((62,))
    fig = plt.figure(figsize=(8,10))
    ax = fig.add_subplot(111, projection='3d',aspect='equal')
    ax.view_init(elev=20., azim=65)
    fig.tight_layout()
    a[3:] = (data[0])*data_std+data_mean
    p = GPy.plotting.matplot_dep.visualize.skeleton_show(a, skel ,axes=ax)
    for i in range(data.shape[0]):
        a[3:] = (data[i])*data_std+data_mean
        p.modify(a)
        fig.savefig(os.path.join(imgpath,'%05d'%i+'.png'))

## Function to show 3D motion sequence of a human
def save_trail(data, data_mean, data_std, skel, filename):
    import os
    import GPy
    a = np.zeros((62,))
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0., azim=20)
    ax.set_ylim3d([-10, 130])
    ax.set_zlim3d([-20, 20])
    ax.set_xlim3d([-15, 15])
    for i in range(13,26):
        a[3:] = (data[i*3])*data_std+data_mean
        a[2] = (i-15)*13
        p = GPy.plotting.matplot_dep.visualize.skeleton_show(a, skel ,axes=ax)
    plt.show()
    fig.savefig(filename)

    
def reshape(data_list):
    '''
    This function stack all separete feature lists into a big list.
    (29,4) -> (4,)
    So that in each list there are 62 feature dimensions -> (_,62)
    '''
    import numpy as np
    n = np.shape(data_list)[-1]
    out_list = [[] for _ in range(n)]
    for i, bone in enumerate(data_list):
        for seq, data in enumerate(bone):
            if i==0:
                out_list[seq] = data
            else:
                out_list[seq] = np.hstack((out_list[seq],data))
    return out_list


def test_run(model, seq, win_out, win_in, test_data, control_signal, path, MEAN_PRED=True, with_control=True):
    if with_control:
        YD = np.shape(test_data[0])[-1]; print('Observationa dimension:', YD)
        UD = 1 if model is 'root' else np.shape(control_signal[0])[-1] ; print('Control signal dimension:', UD)
    
        m = RGP(wins=[win_out], with_control=True, X_dims=[YD], U_dim=UD, U_win=win_in, 
            num_inducing=400, kernels=[RBF(win_out*YD+win_in*UD, ARD=True)], 
            ctx=ctx, dtype=np.float64)
        #'./models/control_signal/fully_observed/'
        m.load_params(path+model)
    
        y_pd = m.layers[0].freerun(test_data[seq].shape[0]-win_out, init_X=test_data[seq][:win_out,:],
                               U=control_signal[seq], mean_predict=False, nSamples=100)
        
    else:
        D = np.shape(test_data[0])[-1]
        m = RGP(wins=[win_out], with_control=False, X_dims=[D], num_inducing=400, kernels=[RBF(win_out*D, ARD=True)], ctx=ctx, dtype=np.float64)
        m.load_params('./base_models/'+model)
        y_pd = m.layers[0].freerun(test_data[seq].shape[0]-win_out, init_X=test_data[seq][:win_out,:], mean_predict=False, nSamples=100)
    
    
    y_pd_mean = y_pd.mean(0) # the first dimention of the matrix is number of samples
    y_pd_std = y_pd.std(0)
    
    if MEAN_PRED:
        return y_pd_mean
    else:
        return y_pd_mean + y_pd_std*np.random.normal(0,0.5,size=(y_pd_mean.shape[0], y_pd_mean.shape[1]))
    
    
def plot_pred(seq, win_out, model, data_in, dim, path):
    D = np.shape(data_in[0])[-1]; print('Dimensions: ', D)
    m = RGP(wins=[win_out], with_control=False, X_dims=[D], num_inducing=400, kernels=[RBF(win_out*D, ARD=True)], ctx=ctx, dtype=np.float64)
    m.load_params(path+model)
    y_pd = m.layers[0].freerun(data_in[seq].shape[0]-win_out, init_X=data_in[seq][:win_out,:], mean_predict=False, nSamples=100)
    pred_mean, = plot(y_pd[:,:,dim].mean(0),'b', label='prediction-mean')
    pred_var, = plot(y_pd[:,:,dim].mean(0)-y_pd[:,:,dim].std(0)*2, 'b--', label='prediction-variance')
    plot(y_pd[:,:,dim].mean(0)+y_pd[:,:,dim].std(0)*2, 'b--')
    ground_truth, =plot(data_in[seq][:,dim],'r',label='ground-truth')
    ylabel('Normalised Value')
    xlabel('Samples')
    title(model+': Dimension-'+str(dim+1))
    legend(handles=[pred_mean, pred_var, ground_truth], prop={'size':7})
    savefig('./images/'+model+str(dim))
    
    
def compute_delta(data_list):
       
    def construct_M(seq_len):
        import numpy as np
        block = np.array([-1,1])
        padded_block = np.pad(block, (0, seq_len-2), mode='constant')
        M = np.concatenate( list(map(lambda x: np.roll([padded_block], x, axis=1), range(seq_len-1))) )
        return np.vstack([np.zeros(seq_len), M])

    out_list = []
    for data in data_list:
        M = construct_M(data.shape[0])
        out_list.append(M.dot(data))
    return out_list



def compute_MSE(y_test_list, pds):
    # merge the individual joints as a complete matrix
    target = [[] for _ in range(np.shape(y_test_list)[-1])]
    for i, joint in enumerate(y_test_list):
        for j, seq in enumerate(joint):
            if i==0:
                target[j] = seq
            else:
                target[j] = np.hstack([target[j], seq])
      
    MSE = []
    for i, pd in enumerate(pds):
        tmp = pd - target[i]
        MSE.append( tmp.dot(tmp.T).diagonal().mean()/62 )
    return MSE


    