import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

np.set_printoptions(suppress=True, precision=3)
from datetime import datetime
from itertools import chain

import os
import os.path
import ast
import random

import scipy
from scipy import stats
from scipy.io import loadmat
from scipy.io import savemat
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy.stats import pearsonr
from scipy.optimize import least_squares
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.stats import circmean
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import circmean
from scipy.stats import linregress
from scipy.spatial.distance import cosine
from scipy.linalg import svd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp, kruskal, mannwhitneyu
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from scipy import signal
from scipy.io import loadmat
from scipy.ndimage import median_filter
from scipy import signal
from scipy.io import loadmat
from scipy.stats import sem
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform

# from sklearn.manifold import MDS
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import normalize
# from scipy.cluster.hierarchy import linkage, fcluster
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.manifold import Isomap
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import pairwise_distances
# from sklearn.metrics import jaccard_score
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import StandardScaler

import seaborn as sns
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import umap
# import hdbscan
from collections import defaultdict, Counter
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture

import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests
import itertools
from joblib import load, dump
import imageio
import cv2
import random
import math
import json
from joblib import dump, load
from types import SimpleNamespace
import colorednoise as cn
from PIL import Image
from tqdm import tqdm
import re

# from facemap import process
from glob import glob
from rastermap import Rastermap, utils
import skimage
import skimage.transform as transform
from skimage.measure import block_reduce
import skimage.transform as transform
from skvideo.io import vreader
import pandas as pd
from itertools import combinations
import umap.umap_ as umap
from datetime import datetime, timedelta
from scipy.signal import find_peaks, peak_widths
from scipy.stats import f_oneway