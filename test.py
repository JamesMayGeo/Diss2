# Import of relevent sktime and sktime module information

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sktime.transformations.panel.pca import PCATransformer
from sktime.utils.data_processing import from_2d_array_to_nested
from sktime.utils.data_processing import from_3d_numpy_to_nested
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.transformations.panel.rocket import Rocket
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.transformations.panel.dictionary_based import SFA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numpy.fft import fft
import math
from scipy.fft import ifftn
from sktime.transformations.series.acf import AutoCorrelationTransformer
from sklearn.preprocessing import PowerTransformer
from sktime.transformations.panel.reduce import Tabularizer
from sklearn.neighbors import KNeighborsClassifier
import sktime
import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "C:\\Users\\James\\Documents\\uni\\CompMasters\\Diss\\Univariate_ts")

datasets_trainA = ["GunPoint", "ArrowHead","Beef","Lightning2","Lightning7","ECG200","Adiac","FaceFour","FiftyWords","CBF","Fish"]
datasets_testA = datasets_trainA

datasets_train = []
datasets_test = []

for i in range(len(datasets_trainA)):
    datasets_train.append(datasets_trainA[i] + str("\\") + datasets_trainA[i] + str("_TRAIN.ts"))
for i in range(len(datasets_testA)):
    datasets_test.append(datasets_testA[i] + str("\\") + datasets_testA[i] + str("_TEST.ts"))

classifier_Knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean")
classifier_DTW = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")

pipeline_pst_knn = Pipeline(
    [
        (
           "pst",
            PowerTransformer())
        ,
        ("knn", KNeighborsClassifier(n_neighbors = 3)),
    ]
)

pipeline_act_knn = Pipeline(
    [
        #("act", AutoCorrelationTransformer()),
       ("knn", KNeighborsClassifier(n_neighbors = 3)),
    ]
)

pipeline_pca_knn = Pipeline(
    [
        ("ss", StandardScaler()),
        ("pca", PCA()),
        ("knn", KNeighborsClassifier(n_neighbors = 3))
    ]
)
act = AutoCorrelationTransformer()

#bundle up the classifiers
clfs = [ classifier_Knn, classifier_DTW, pipeline_pst_knn, pipeline_act_knn, pipeline_pca_knn]
names = ["Euclidean", "DTW", "powerknn",  "actknn",  "pcaknn"]
table = {'Classifier':names}
table = pd.DataFrame(table)


for k, j  in zip(datasets_train, datasets_test):
    train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, k))
    test_x, test_y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, j))
    pcas = []
    
    #fit the classifiers with the data
    for clf in clfs:
        if clf in [pipeline_pst_knn, pipeline_pca_knn]:
            xs = from_nested_to_2d_array(train_x)       
            clf.fit(xs,train_y)

        elif clf in [pipeline_act_knn]:
            xs = from_nested_to_2d_array(train_x)
            xs = np.asarray(xs, dtype='float64')
            x_hat = [act.fit_transform(xs[i],train_y) for i in range(len(train_x))]
            clf.fit(x_hat,train_y)
        else:    
            clf.fit(train_x, train_y)
    

    #get the predictions for each classifier
    clf_preds = []
    for clf in clfs:
        if clf in [pipeline_pst_knn, pipeline_pca_knn]:
            xt = from_nested_to_2d_array(test_x)
            clf_preds.append([clf.predict(xt)])
        elif clf in [pipeline_act_knn]:
            xt = from_nested_to_2d_array(test_x)
            xt = np.asarray(xt, dtype='float64')
            x_hat = [act.transform(xt[i],test_y) for i in range(len(test_x))]
            clf_preds.append([clf.predict(x_hat)])
        else:
            clf_preds.append([clf.predict(test_x)])

    #get the accuracy score for each classifer
    accs = []

    for clf_p in clf_preds:

        accs.append([accuracy_score(test_y, clf_p[0])])

    #get the index of the best accuracy
    index = np.argmax(accs)

    #get the index of the best accuracy
    index = np.argmax(accs)

    #print("best classifier is:", names[index], " with an accuracy of: ", accs[index])
    np.sort(accs)
    data = {'classifier':names, 'accuracy':accs}
    data = pd.DataFrame(data)
    data["Rank"] = len(names) - data["accuracy"].rank() + 1
    table[f'Accuracy {k[0:10]}'] = data["accuracy"]
    table[f'Rank {k[0:10]}'] = data["Rank"]
a = table.iloc[:,::2]
a = a.iloc[: , 1:]
table['AVG Rank'] = a.mean(axis=1)
print(table)
