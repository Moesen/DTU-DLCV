import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
#from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from projects.utils import get_project2_root

import numpy as np
import pandas as pd
import seaborn as sns



def get_latent_direction(X, y):
    """
    X should be [n_data_points, feature_dim]
    y should be [n_data_points]
    both numpy arrays
    """

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(X, y)

    return clf.coef_[0]


def plot_latent_direction(X,y,mag=100,ref=[-2,-2],labels=["class1","class2"]):

    scaler = StandardScaler()

    scaler = scaler.fit(X)
    x_scaled = scaler.transform(X)

    #x_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(x_scaled)

    # Create dataframe
    pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC1', 'PC2'])
    
    # map target names to PCA features   
    target_names = {
    0:labels[0],
    1:labels[1], 
    }
    
    pca_df['class'] = y
    pca_df['class'] = pca_df['class'].map(target_names)

    sns.set()
    
    sns.lmplot(
        x='PC1', 
        y='PC2', 
        data=pca_df, 
        hue='class', 
        fit_reg=False, 
        legend=True
        )
    
    w = get_latent_direction(X, y)
    w = w[np.newaxis,:]
    #w = StandardScaler().fit_transform(w)
    w = scaler.transform(w)
    w = pca.transform(w).squeeze()

    xx = np.array([ref[0],mag*w[0]])
    yy = np.array([ref[1],mag*w[1]])

    #plt.plot(xx,yy)
    plt.arrow(xx[0], yy[0], xx[1], yy[1], length_includes_head=True,
          head_width=0.08, head_length=0.5, width=0.02,edgecolor='black')
    plt.text(xx[0]+xx[1], yy[0]+yy[1], "w", fontsize=12)

    plt.title('PCA projected classes')

    PROJECT_ROOT = get_project2_root()
    save_path =  PROJECT_ROOT / "reports/figures/latent_direction_PCA.png"

    plt.savefig(save_path)

    # fit the model, don't regularize for illustration purposes
    """clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(X, y)

    #plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    #ax = plt.gca()
    #ax.scatter(X[:,0],X[:,1])

    xx = np.array([0,0])
    y = clf.coef_[0]

    mag = 100

    v = [-10,-10]
    xx = np.array([v[0], v[0] + mag*y[0]])
    yy = np.array([v[1], v[1] + mag*y[1]])


    xxx = np.array([-10,15])
    yyy = -(clf.intercept_[0] + xxx*clf.coef_[0][0]) / clf.coef_[0][-1]

    ax.plot( xx, yy)
    ax.plot(xxx,yyy)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect('equal')

    print( np.dot( np.array([y[0],y[1]]), np.array([xxx[1]-xxx[0],yyy[1]-yyy[0]]) ) )

    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.show()
    #plt.savefig()"""





"""X, y = make_blobs(n_samples=100, centers=2, random_state=6)

X = np.array([[1,1],
             [0.5,0.5],
             [0,1],
             [2,2],
             [3,3],
             [3,2]])

y = np.array([0,0,0,1,1,1])


X = np.array([[1,1,1],
             [0.5,0.5,1],
             [0,1,2],
             [-1,1,4],
             [2,-1,3],
             [2,2,3],
             [3,3,4],
             [3,2,9],
             [4,1,5],
             [5,3,2]])

y = np.array([0,0,0,0,0,1,1,1,1,1])

plot_latent_direction(X,y,mag=-3,ref=[-2,-1])"""
