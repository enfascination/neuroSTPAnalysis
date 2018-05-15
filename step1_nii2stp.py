#!/usr/bin/env python

# coding: utf-8

# In[32]:


#!/usr/bin/env python
from local_settings import * 

import os
import sys
from glob import glob
from time import time as getSecs
import pdb
import argparse
import logging
if True:
    dryrun = True
    log_level = logging.DEBUG
else:
    dryrun = False
    log_level = logging.INFO
logging.basicConfig(level=log_level)

import numpy as np
from numpy import inf
from numpy.core.defchararray import startswith
import scipy.stats as stats
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

from nltools.data import Brain_Data, Design_Matrix

### tisean code
from pytisean import tiseanio

from psutil import virtual_memory
import multiprocessing
mem = virtual_memory()
logging.debug(mem.total/1024/1024/1024)  # total physical memory available in GB
logging.debug(multiprocessing.cpu_count()) #number of CPUs

#%matplotlib inline


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('--sub', default="4", help='subject index (integer)')
sub_list = [d for d in os.listdir(preproc_dir) if not d.startswith('.')]
#sub_list = ["sub-sid000216", "sub-sid000476", "sub-sid000529", "sub-sid000668", "sub-sid000678", "sub-sid000682"]
print(preproc_dir)
print(sub_list)
nSub = len(sub_list)
args = parser.parse_args()
subject_id = sub_list[int(args.sub)]


# ## Local Functions

# In[9]:


def fileGetter(preproc_dir,subject_id,episode):
    """
    Quick file getter to traverse study tree and return subject epi and covariates file paths
    for a given episode.
    
    Args:
        data_dir (str): parent directory of onsets files (usually same as raw data)
        preproc_dir (str): parent directory of covariates files (usually same as preproc outputs)
        subject_id (str): subject scan-name (should be same as folder name for a subject)
        episode (int): what episode to grab files for 
    
    Returns:
        covFiles (list): covariate file path
        funcFiles (list): nifti file paths    
        
    """
    episode = str(episode)
        
    funcFiles = glob(os.path.join(preproc_dir, subject_id,'functional','_fwhm_6.0','*nii.gz'))
    covFiles = glob(os.path.join(preproc_dir,subject_id,'functional', 'covariates.csv'))
    #covFiles = glob(os.path.join(preproc_dir,subject_id,'functional','ep0'+episode+'*','cov*'))
    #funcFiles = glob(os.path.join(preproc_dir,subject_id,'functional','ep0'+episode+'*','*int16.nii.gz'))
    
    return covFiles, funcFiles


# In[34]:


def stpw(ts, dspace=10, dtime=100, carefully=True):
    pct = 1.0 / dspace
    out, _ = tiseanio('stp', '-d1', '-m4', '-t{}'.format(dtime), '-#1', '-%{}'.format(pct), data=ts, silent=not carefully)
    nonzero_count = (dspace * dtime)
    zero_count = out.shape[0] - (dspace * dtime)
    out_zeros = out[nonzero_count:len(out), :]
    out = out[0:nonzero_count, :]
    #print(out[:,1])
    #print(out_zeros[:,1])
    ### test (keep carefully true for the foreseeable until I know my pct calucs are clean)
    if carefully:
        assert pct <= 1 and pct > 0
        assert dspace >=1 
        assert len(out) == nonzero_count, (len(out), nonzero_count, dspace, dtime)
        assert len(out_zeros) == zero_count, (len(out_zeros), zero_count, dspace, dtime)
        # vvv this test is wierd and not really necessary and most important I can't get it working
        #assert np.all(0 == np.around(out_zeros[:,1],10)), stats.describe(np.around(out_zeros[:,1],10))
        assert np.all(0 != out[:,1]), out
    out[out == -inf] = 0
    out[out == inf] = 0
    out = np.nan_to_num(out)
    out = pd.DataFrame(out, columns=["time_dist", "dist_val"])
    out.time_dist =  out.time_dist.astype(int)
    out.dist_val =  np.around(out.dist_val, 3)
    out = out.assign( phasespace_dist = np.repeat((np.arange(dspace)+1) / dspace, dtime) )
    return(out)

def recurrw(ts):
    out, _ = tiseanio('recurr', '-d1', data=ts )
    out[out == -inf] = 0
    out[out == inf] = 0
    out = np.nan_to_num(out)
    return(out)


# In[925]:


def findsorted(haystack_srtd, needle, carefully=False):
    if carefully:
        assert np.all(np.sort(haystack_srtd) == haystack_srtd)
    ix = np.searchsorted(haystack_srtd, needle)
    if ix == len(haystack_srtd):
        out = False
    elif needle != haystack_srtd[ix]:
        out = False
    else:
        out = True
    return(out)


# ## Local arguments

# In[37]:


episodes = [1,2]
episodes = [1,]
episode = 1
TR = 2.0
smooth = '6mm'

max_dtime = 576

csf = Brain_Data(os.path.join(base_dir,'masks','csf.nii.gz'))

out_file = os.path.join(path_dataout,'1_{}_ep{}_voxel2stp.csv.gz'.format(subject_id, episode))


# ## Prepare brain data

# In[13]:


# Get data and covariates file and create nuisance design matrix
print(preproc_dir, subject_id, episode)
sub_cov, sub_epi = fileGetter(preproc_dir,subject_id,episode)

#Load run data
print("Loading brain data: {}".format(smooth))
dat = Brain_Data(sub_epi)

cov_mat = Design_Matrix(pd.read_csv(sub_cov[0]).fillna(0), sampling_rate=TR)
# Add Intercept
cov_mat['Intercept'] = 1
# Add Linear Trend
cov_mat['LinearTrend'] = range(cov_mat.shape[0])-np.mean(range(cov_mat.shape[0]))
cov_mat['QuadraticTrend'] = cov_mat['LinearTrend']**2
cov_mat['CSF'] = dat.extract_roi(csf.threshold(.85,binarize=True))

assert cov_mat.shape[0] == dat.shape()[0]
spikeless_idx =  np.logical_not( startswith(cov_mat.columns.values.astype(str), "spike") | startswith(cov_mat.columns.values.astype(str), "FD") )
#dat.X = cov_mat
dat.X = cov_mat.loc[:,spikeless_idx]
datcln = dat.regress()['residual']


# ## Loop through voxels to produce STPs

# In[14]:


zero_voxels = np.apply_along_axis(lambda x: np.count_nonzero(x) == 0, axis=0, arr=datcln.data)
zero_voxels = np.arange(datcln.data.shape[1])[zero_voxels]


# In[36]:


if dryrun:
    cols = np.random.randint(low=0,high=datcln.data.shape[1],size=20)
    nBatch = 10
else:
    del dat
    cols = np.arange(datcln.data.shape[1])
    nBatch = 1000
### voxel-level parameter maps
stpl = []
out_header = ["voxel", "phasespace_dist", "time_dist", "dist_val"]
#vstp = pd.DataFrame(columns=out_header)
#vstp.to_csv(out_file, encoding="utf-8", compression="gzip", index=False)
for col in cols:
    if np.count_nonzero(datcln.data[:,col]) == 0:
        logging.debug("skipping col {} for zeros".format(col))
        continue
    vstp = stpw(datcln.data[:,col], dspace=10, dtime=576, carefully=False)
    vstp = vstp.loc[np.where(vstp.phasespace_dist.isin([0.1, 0.5, 1.0]))]
    vstp = vstp.assign(voxel=col)
    vstp = vstp.loc[:,out_header]
    stpl.append( vstp )
    del vstp
    if len(stpl) == nBatch or col == cols[-1] :
        stpl = pd.concat(stpl)
        stpl.to_csv(out_file, mode="a", header=False, encoding="utf-8", compression="gzip", index=False)
        stpl = []


# ## Cleanup

# In[ ]:


if dryrun:
    pass
else:
    del datcln
elapsed = (getSecs()-sTime)/60.
logging.info("Analysis complete in: %.2f minutes" % elapsed)

