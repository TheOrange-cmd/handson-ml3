def create_flower_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c = y[0, :], s = 80, edgecolors='black', cmap=plt.cm.Spectral)

def dummy_dataset_1():
    return np.random.rand(3, 256), 8, np.random.rand(2, 256)

def dummy_dataset_2():
    return np.random.rand(5, 432), 7, np.random.rand(1, 432)

def forward_propagation_test():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)

    parameters = {'WM1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'WM2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'BV1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'BV2': np.array([[ 0.]])}

    return X_assess, parameters

def gt_grads():
    return {'dW1': np.array([[ 0.00209634, -0.0005974 ],
       [ 0.00179617, -0.00051044],
       [-0.00109224,  0.00031134],
       [-0.00453992,  0.00129391]]), 'db1': np.array([[ 0.00152135],
       [ 0.0013039 ],
       [-0.00079283],
       [-0.00329514]]), 'dW2': np.array([[ 0.00079406,  0.00515362,  0.00307798, -0.00169948]]), 'db2': np.array([[-0.14380585]])}

def bp_values():
    params = {'WM1': np.array([[-0.00416758, -0.00056267],
       [-0.02136196,  0.01640271],
       [-0.01793436, -0.00841747],
       [ 0.00502881, -0.01245288]]), 'WM2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]), 'BV1': np.array([[0.],
       [0.],
       [0.],
       [0.]]), 'BV2': np.array([[0.]])}

    cache = {'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
       [-0.05229879,  0.02726335, -0.02646869],
       [-0.02009991,  0.00368692,  0.02884556],
       [ 0.02153007, -0.01385322,  0.02600471]]), 'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
       [-0.05225116,  0.02725659, -0.02646251],
       [-0.02009721,  0.0036869 ,  0.02883756],
       [ 0.02152675, -0.01385234,  0.02599885]]), 'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]]), 'A2': np.array([[0.5002307 , 0.49985831, 0.50023963]])}

    X_assess = np.array([[ 1.62434536, -0.61175641, -0.52817175],
                             [-1.07296862,  0.86540763, -2.3015387 ]])

    Y_assess = np.array([[0.88654505, 0.61002528, 0.43517586]])
    return params, cache, X_assess, Y_assess

def get_dummy_images():
    X_dummy = np.random.randn(3456, 343)
    y_dummy = [0, 1, 1, 2, 1, 1, 4,3, 2, 2, 3, 5, 6]
    return X_dummy, y_dummy

import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os, glob

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import ipywidgets
from IPython.display import display

from collections import OrderedDict

import scipy
import skimage
import skimage.transform
import skimage.util

import glob

def list_images(image_dir, filename_expression='*.jpg'):
    filenames = glob.glob(os.path.join(image_dir, filename_expression))
    filenames = sorted(filenames) # important for cross-platform compatiblity
    print(f'Found {len(filenames)} image files in the directory "{image_dir}"')
    return filenames

# the size of the patch in pixels
WIN_SIZE = (100, 100, 3)

# for convenience, half the window
HALF_WIN_SIZE = (WIN_SIZE[0] // 2, WIN_SIZE[1] // 2, WIN_SIZE[2])

def get_patch_at_point(I, p):
    ### BEGIN SOLUTION
    p = p.astype(int)
    P = I[(p[1]-HALF_WIN_SIZE[1]):(p[1]+HALF_WIN_SIZE[1]), (p[0]-HALF_WIN_SIZE[0]):(p[0]+HALF_WIN_SIZE[0]),:].copy()
    ### END SOLUTION

    return P
def sample_points_grid(I):
    # window centers
    W = I.shape[1]
    H = I.shape[0]

    step_size = (WIN_SIZE[0]//2, WIN_SIZE[1]//2)
    min_ys = range(0, H-WIN_SIZE[0]+1, step_size[0])
    min_xs = range(0, W-WIN_SIZE[1]+1, step_size[1])
    center_ys = range(HALF_WIN_SIZE[0], H-HALF_WIN_SIZE[0]+1, step_size[0])
    center_xs = range(HALF_WIN_SIZE[1], W-HALF_WIN_SIZE[1]+1, step_size[1])
    centers = np.array(np.meshgrid(center_xs, center_ys))
    centers = centers.reshape(2,-1).T
    centers = centers.astype(float)

    # add a bit of random offset
    centers += np.random.rand(*centers.shape) * 10

    # discard points close to border where we can't extract patches
    centers = remove_points_near_border(I, centers)

    return centers

def sample_points_around_pen(I, p1, p2):
    Nu = 100 # uniform samples (will mostly be background, and some non-background)
    Nt = 50 # samples at target locations, i.e. near start, end, and middle of pen

    target_std_dev = np.array(HALF_WIN_SIZE[:2])/3 # variance to add to locations

    upoints = sample_points_grid(I)
    idxs = np.random.choice(upoints.shape[0], Nu)
    upoints = upoints[idxs,:]


    # sample around target locations
    tpoints1 = np.random.randn(Nt,2)
    tpoints1 = tpoints1 * target_std_dev + p1

    tpoints2 = np.random.randn(Nt,2)
    tpoints2 = tpoints2 * target_std_dev + p2

    # sample over length pen
    alpha = np.random.rand(Nt)
    tpoints3 = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None])
    tpoints3 = tpoints3 + np.random.randn(Nt,2) * target_std_dev

    # merge all points
    points = np.vstack((upoints, tpoints1, tpoints2, tpoints3))

    # discard points close to border where we can't extract patches
    points = remove_points_near_border(I, points)

    return points

def remove_points_near_border(I, points):
    W = I.shape[1]
    H = I.shape[0]

    # discard points that are too close to border
    points = points[points[:,0] > HALF_WIN_SIZE[1],:]
    points = points[points[:,1] > HALF_WIN_SIZE[0],:]
    points = points[points[:,0] < W - HALF_WIN_SIZE[1],:]
    points = points[points[:,1] < H - HALF_WIN_SIZE[0],:]

    return points

CLASS_NAMES = [
    'background', # class 0
    'tip',        # class 1
    'end',        # class 2
    'middle'      # class 3
]

def make_labels_for_points(I, p1, p2, points):
    """ Determine the class label (as an integer) on point distance to different parts of the pen """
    num_points = points.shape[0]

    # for all points ....

    # ... determine their distance to tip of the pen
    dist1 = points - p1
    dist1 = np.sqrt(np.sum(dist1 * dist1, axis=1))

    # ... determine their distance to end of the pen
    dist2 = points - p2
    dist2 = np.sqrt(np.sum(dist2 * dist2, axis=1))

    # ... determine distance to pen middle
    alpha = np.linspace(0.2, 0.8, 100)
    midpoints = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None])
    dist3 = scipy.spatial.distance_matrix(midpoints, points)
    dist3 = np.min(dist3, axis=0)

    # the class label of a point will be determined by which distance is smallest
    #    and if that distance is at least below `dist_thresh`, otherwise it is background
    dist_thresh = WIN_SIZE[0] * 2./3.

    # store distance to closest point in each class in columns
    class_dist = np.zeros((num_points, 4))
    class_dist[:,0] = dist_thresh
    class_dist[:,1] = dist1
    class_dist[:,2] = dist2
    class_dist[:,3] = dist3

    # the class label is now the column with the lowest number
    labels = np.argmin(class_dist, axis=1)

    return labels

def plot_labeled_points(points, labels):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'r.', label=CLASS_NAMES[0])
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'g.', label=CLASS_NAMES[1])
    plt.plot(points[labels == 2, 0], points[labels == 2, 1], 'b.', label=CLASS_NAMES[2])
    plt.plot(points[labels == 3, 0], points[labels == 3, 1], 'y.', label=CLASS_NAMES[3])

def patch_to_vec(P, FEAT_SIZE):

    ### BEGIN SOLUTION
    P = skimage.transform.resize(P, FEAT_SIZE, anti_aliasing=False)
    if len(FEAT_SIZE) < 3: # flatten the color space
        #P = rgb2gray(P)
        P = np.mean(P, axis=2) # faster than rgb2gray
    x = P.flatten() # make vector
    ### END SOLUTION

    return x

def extract_patches(I, p1, p2, FEAT_SIZE):

    points = sample_points_around_pen(I, p1, p2)
    labels = make_labels_for_points(I, p1, p2, points)

    xs = []
    for p in points:
        P = get_patch_at_point(I, p)
        x = patch_to_vec(P, FEAT_SIZE)
        xs.append(x)
    X = np.array(xs)

    return X, labels, points

def extract_multiple_images(Is, idxs, annots, FEAT_SIZE, print_output=False):
    Xs = []
    ys = []
    points = []
    imgids = []

    for step, idx in enumerate(idxs):
        I = Is[idx]
        I_X, I_y, I_points = extract_patches(I, annots[idx,:2], annots[idx,2:], FEAT_SIZE)

        classcounts = np.bincount(I_y)
        if print_output:
            print(f'image {idx}, class count = {classcounts}')

        Xs.append(I_X)
        ys.append(I_y)
        points.append(I_points)
        imgids.append(np.ones(len(I_y),dtype=int)*idx)

    Xs = np.vstack(Xs)
    ys = np.hstack(ys)
    points = np.vstack(points)
    imgids = np.hstack(imgids)
    print("done.")

    return Xs, ys, points, imgids

def plot_samples(Ps, labels, FEAT_SIZE, nsamples):
    uls = np.unique(labels)
    nclasses = len(uls)

    plt.figure(figsize=(10,4))

    for lidx, label in enumerate(uls):
        idxs = np.where(labels == label)[0]
        idxs = np.random.choice(idxs, nsamples, replace=False)

        for j, idx in enumerate(idxs):
            P = Ps[idx,:]
            P = P.reshape(FEAT_SIZE)

            plt.subplot(nclasses, nsamples, lidx*nsamples+j+1)
            plt.imshow(P, clim=(0,1))
            plt.axis('off')
            plt.title('label: %d' % label)

    plt.show()
