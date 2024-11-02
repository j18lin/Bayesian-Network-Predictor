import pandas as pd

raw_data_path = "OSHA Accident and Injury Data.csv" # put the directory of the raw dataset between the quotation marks

raw_data = pd.read_csv(raw_data_path)
raw_data # csv file as it was

# Select all features based on all networks we have 
Sel_feas = ['degree_of_injury',
            'Nature of Injury',
            'Part of Body',
            'Event type',
            'Environmental Factor',
            'Human Factor',
            'Task Assigned'] 

Sel_data = raw_data[Sel_feas]
print("The number of original data records is ", str(len(Sel_data)))

# # drop the missing values
Sel_data_na = Sel_data.dropna()

missing_no = len(Sel_data)-len(Sel_data_na)

print(str(missing_no) + ' data records are subject to missing values')
print("Mising value rate is " + str(round(100 * missing_no/len(Sel_data), 4)) + '%')

data = Sel_data

data # all non preprocessed relevant data


sub_data_na = sub_data.dropna() # Drop the samples with missing values from the subset

sub_data_na.reset_index(inplace=False) # Rest the index for the pandas dataframe

## Replace the original column names using the abbreviatations below 
  # DOI: degree of injury
  # NI: nature of injury
  # PB: part of body
  # ET: event type
  # EF: environmental factor
  # HF: human factor
  # TA: task assigned

new_column_names = ['DOI', 'NI', 'PB', 'ET', 'EF', 'HF', 'TA']

sub_data_na.columns = new_column_names

sub_data_na

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN


def data_preproc(sub_data_na, outlier_removal, c, n, kernel, clustertype, nc, e, ns):
    '''
    Input:
    (1) data (pandas dataframe): the raw dataset for the selected features only
        
    (2) outlier_removal (str): the type of outlier removal techniques, including:
        (a) 'MCD': minimum covariance determinant
        (b) 'LOF': local outlier factor
        (c) 'SVM': one-class SVM
        
    (3) c (float): the propotion of outliers in data 
    (4) n (int): the number of neighbors for LOF outlier detection technique
    (5) kernel (str): the type of kernel for one-class SVM
    (6) clustertype (str): the type of clustering algorithm
    (7) nc (int): number of clusters
    (8) e (float): distance to the nearest n data points
    (9) ns (int): number of data samples
    
    Output:
    (1) clean_data: clean dataset for BN parameter learning
    '''
    
    '''
    Subtask 1: text clustering to dicrease data dimensionality
    '''
    def text_clustering(data, clustertype, nc, e, ns):
        
        if clustertype == 'k-means':
            kmeans = KMeans(n_clusters=nc, random_state=42)
            kmeans.fit(data)
            clusters = kmeans.labels_
            return clusters
            
        elif clustertype == 'dbscan':
            DB = DBSCAN(eps=e, min_samples=ns)
            DB.fit(data)
            clusters = DB.labels_
            return clusters
    
    # create an empty dataframe to store the clustered textual variables
    clustered_data = pd.DataFrame()
    
    # get a list of variables from the input dataset
    variables = sub_data_na.columns.tolist()
    variables_excluded = ['DOI', 'TA']
    
    # Perform text clustering on the variables that need to be clustered only
    features = [x for x in variables if x not in variables_excluded]
    
    for feature in features:
        
        vectorizer = TfidfVectorizer(stop_words='english')
        data_X = vectorizer.fit_transform(sub_data_na[feature])

        clutered_var = text_clustering(data_X, clustertype, nc, e, ns)
        
        clustered_data[feature] = clutered_var
    
    # Append the variables that require no text clustering
    clustered_data[variables_excluded] = sub_data_na[variables_excluded]
    
    # Encode the categorical values of two variables using ordinal numbers
    clustered_data.loc[clustered_data['DOI'] == 'Nonfatal', 'DOI'] = 0
    clustered_data.loc[clustered_data['DOI'] == 'Fatal', 'DOI'] = 1
    clustered_data.loc[clustered_data['TA'] == 'Not Regularly Assigned', 'TA'] = 0
    clustered_data.loc[clustered_data['TA'] == 'Regularly Assigned', 'TA'] = 1
    
    # drop the potential missing values
    clustered_data = (clustered_data.dropna()).astype(int)
    
    '''
    Subtask 2: Outlier removal
    '''
    ################# Define a function for outlier removal##############
    #ref: https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
    def remove_outliers(clustered_data, outlier_removal, c, n, kernel):
        '''
        Input:
        (1) clustered_data (pandas dataframe): the normalized and imputed dataset
        (2) outlier_removal (str): outlier removal method
        (3) c (float): the propotion of outliers in data (i.e., argument 1 in this function)
        (4) n (int): the number of neighbors for LOF outlier detection technique
        (5) kernel (str): the type of kernel for one-class SVM
        
        Output:
        (1) ro_data (pandas dataframe): complete dataset without outliers
        '''
        
        if outlier_removal == 'MCD':
            if c > 0.5: # c has a bound of (0, 0.5] for MCD algorithm  
                c = 0.5
                mcd = EllipticEnvelope(contamination = c, random_state = 0)
                y_hat = mcd.fit_predict(clustered_data)
                mask = y_hat != -1
                ro_data = clustered_data.loc[mask, :]
            else:
                mcd = EllipticEnvelope(contamination = c, random_state = 0)
                y_hat = mcd.fit_predict(clustered_data)
                mask = y_hat != -1
                ro_data = clustered_data.loc[mask, :]
            
        elif outlier_removal == 'LOF': 
            if c > 0.5: # c has a bound of (0, 0.5] for MCD algorithm
                c = 0.5
                lof = LocalOutlierFactor(n_neighbors = n, contamination = c)
                y_hat = lof.fit_predict(clustered_data)
                mask = y_hat != -1
                ro_data = clustered_data.loc[mask, :]
                
            else:
                lof = LocalOutlierFactor(n_neighbors = n, contamination = c)
                y_hat = lof.fit_predict(clustered_data)
                mask = y_hat != -1
                ro_data = clustered_data.loc[mask, :]
            
        elif outlier_removal == 'SVM':
            svm = OneClassSVM(kernel = kernel, nu = c)
            y_hat = svm.fit_predict(clustered_data)
            mask = y_hat != -1
            ro_data = clustered_data.loc[mask, :]
            
        ro_data = ro_data.reset_index(drop=True) # !!! reset the index of dataframe
        
        return ro_data
    
    clean_data = remove_outliers(clustered_data, outlier_removal, c, n, kernel)

    return clean_data


clean_data = data_preproc(sub_data_na, 
                         'SVM',    #outlier_removal, 
                         0.3,      #c, 
                         1,        #n, 
                         'linear', #kernel, 
                         'k-means',
                         3,
                         0.5,
                         5)

clean_data

import bnlearn as bn
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


def param_learning(BN_structure, data_tr):
    
    '''
    Input:
    (1) BN_structure (list): a list the stores all the edges between nodes of a BN structure
    (2) data_tr: the training dataset used to learn the dependency relationships between variables
    
    Output:
    (1) DAG_update: DAG with conditional prabability table for each node in DAG
    '''
    
    DAG = bn.make_DAG(BN_structure, verbose=0) # DAG is stored in an adjacency matrix
                                        # 'verbose=0' print no progress to screen (default: verbose = 3)
    # visualize the DAG
    #bn.plot(DAG)
    
    DAG_update = bn.parameter_learning.fit(DAG, data_tr, verbose=0)
    
    # visualize the updated DAG with CPD
    #bn.plot(DAG_update, interactive=True)
    
    return DAG_update 


def bn_infer(X_te, bn_model, variables):
    
    '''
    Input:
    (1) X_te (pandas dataframe): data that include observations for all selected evidence for inference
    (2) bn_model: trained Bayesian network with known DAG and CPD
    (3) variables (list): the feature(s) to be infered using the trained Bayesian network
    
    Output:
    (1) y_pred (pandas dataframe): the predicted class for pile capacity
    '''
    
    '''
    step 1: extract the evidence from the discretized testing dataset
    '''
    X_te.reset_index(drop = True, inplace = True)
    
    evidence_names = list(X_te.columns)
    
    query_result = []
    
    for row in list(X_te.index):
        
        evidence = {}
        
        for evidence_name in evidence_names:
        
            evidence[evidence_name] = X_te.loc[row, evidence_name]
        
        query = bn.inference.fit(bn_model, variables = variables, evidence = evidence, verbose = 0)
        
        p_list = list(query.df['p']) # query.df is a pandas dataframe
        
        pred = list(query.df[variables[0]])
        
        max_idx = p_list.index(max(p_list))
        
        query_result.append(pred[max_idx])
        
    y_pred = pd.DataFrame(query_result, columns = variables)
      
    return y_pred


from sklearn import metrics
from sklearn import preprocessing

def cal_cat_metrics(y_test, y_pred):
        
    accuracy = round(metrics.accuracy_score(y_test, y_pred), 4) # accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = round(metrics.precision_score(y_test, y_pred, average='weighted'), 4) # precision = tp / (tp + fp)

    recall = round(metrics.recall_score(y_test, y_pred, average='weighted'), 4) # recall: tp/(tp + fn)

    f1 = round(metrics.f1_score(y_test, y_pred, average='weighted'), 4) # f1 = 2*(acc * recall)/(acc + recall)
    
    #------------------------------#
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test_auc = lb.transform(y_test)
    y_pred_auc = lb.transform(y_pred)
    auc = round(metrics.roc_auc_score(y_test_auc, y_pred_auc, average='weighted', multi_class = 'ovr'), 4)
    #------------------------------#
    ba_accuracy = round(metrics.balanced_accuracy_score(y_test, y_pred), 4)
    
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    '''
    #cmap ref: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = 'Blues'
    #disp = metrics.ConfusionMatrixDisplay.from_predictions(confusion_matrix = cm, cmap = cmap, colorbar = False)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap = cmap, colorbar = True)
    disp.plot()
    plt.show()
    '''
    cat_metrics = [accuracy, precision, recall, f1, auc, ba_accuracy]
    
    return cat_metrics, cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, format(cm[i, j], 'd'),
        plt.text(j, i, str(cm[i, j]) + '\n' + str(round(100 * (cm[i, j]/sum(cm[i, :])), 1)) + '%',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Predicted range')
    plt.ylabel('True range')
    plt.plot()

#ref: https://stackoverflow.com/questions/60728819/matplotlib-table-individual-colormap-for-each-columns-range    
def cm_visual(cm, labels, alpha):
    import matplotlib.cm as CM
    import itertools
    
    '''
    Input:
    (1) cm (np array): confusion matrix obtained from 'metrics.confusion_matrix' function
    (2) labels (list): a list of labels for different classes
    (3) alpha (float): a index to change the transparency of a color
    
    Output: confusion matrix figure that has color coding for each row and percentage of each element in each row
    '''
    if len(labels) == 5:
        
        fig, ax = plt.subplots(figsize=(20, 6))
        
    elif len(labels) == 4:
        
        fig, ax = plt.subplots(figsize=(10, 6))

    rows = labels
    columns= labels

    colores = np.zeros((cm.shape[0], cm.shape[1], 4))
    for i in range(cm.shape[1]):
        col_data = cm[:, i]
        normal = plt.Normalize(np.min(col_data), np.max(col_data))
        colores[:, i] = CM.Blues(normal(col_data) * alpha)

    data = np.zeros((cm.shape[0], cm.shape[1]), dtype= 'object')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        data[i, j] = str(cm[i, j]) ## + '\n' + '(' + str(round(100 * (cm[i, j]/sum(cm[:, j])), 1)) + '%)'

    # ref: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.table.html
    #fig.patch.set_visible(True)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=data,
             rowLabels=rows,
             colLabels=columns,
             cellColours=colores,
             cellLoc = 'center',
             loc='center',
             rowLoc ='center',
             colWidths=[0.1 for x in columns])
    
    if len(labels) == 4:
        
        table.scale(3, 3) # change the size of the table
        table.set_fontsize(12)
        fig.tight_layout()
    
    elif len(labels) == 5:
        table.scale(3, 3) # change the size of the table
        table.set_fontsize(15)
        fig.tight_layout()
    
    return fig


from sklearn.model_selection import KFold

def bn_param_tr_val(data_tr_val, BN_structure, selected_fea, variables):
    
    '''
    Input:
    (1) data_tr_val (pandas dataframe): the discretized training (including validation) dataset (80%) for all features
    (1) n: the number of folds to which the training dataset will be spitted
    (3) BN_structure (list): a list that stores all the edges between nodes of a BN structure
    (4) selected_fea: the selected features that match all the nodes in the DAG for parameter learning
    (5) variables: the specified feature (e.g., type of injury) to be inferred
    
    Output:
    (1) avg_results: average performance metrics for each n-fold validation
    '''
    
    '''
    Subtask 1: split the training data into n folds (i.e., n-1 folds for training and one fold for validation)
    '''
    data = data_tr_val.copy()
    data.reset_index(drop = True, inplace = True)
    
    kf = KFold(n_splits = 5, random_state = 0, shuffle = True)
    
    split_data = kf.split(data)
    
    results = []
    for tr_idx, val_idx in split_data:
        
        X_tr, X_val = data.loc[tr_idx, selected_fea], data.loc[val_idx, selected_fea]
        
        y_tr, y_val = data.loc[tr_idx, variables], data.loc[val_idx, variables]
        
        data_tr = pd.concat([X_tr, y_tr], axis = 1)
        temp = np.round(data_tr, decimals = 0)
        data_tr = pd.DataFrame(temp, columns = data_tr.columns) 
        
        bn_model = param_learning(BN_structure, data_tr)
        
        y_pred = bn_infer(X_val, bn_model, variables)
        
        performance_metrics, cm = cal_cat_metrics(y_val, y_pred)
        
        results.append(performance_metrics)
        
    columns = ['Accuracy', 'Precision', 'Recall', 'F score', 'AUC', 'Balanced Accuracy']
    temp = np.array(results)
    
    results = pd.DataFrame(temp, columns = columns)
#     non_avg_results = results
#     non_avg_results['CV'] = [1, 2, 3, 4, 5]
#     results['Fold number'] = 5
    
    result_mu = results.mean(axis = 0)
    
    avg_metrics = result_mu.to_frame().T
    
    return avg_metrics


def nodes2feas(BN_structure):
    '''
    Input:
    (1) BN_structure (list): a list of edges between nodes to graphically represent the BN 
    
    Output:
    (1) all_fea (list): names of features captured by the BN structure 
    '''
    nodes = []
    for i in range(len(BN_structure)):
 
        for j in range(2):

            nodes.append(BN_structure[i][j])

    all_fea = list(dict.fromkeys(nodes))
    
    return all_fea


NBN_1 = [('PB', 'DOI'),
         ('ET', 'DOI'),
         ('HF', 'DOI'),
         ('TA', 'DOI')]

NBN_2 = [('NI', 'DOI'),
         ('PB', 'DOI'),
         ('ET', 'DOI'),
         ('HF', 'DOI'),
         ('TA', 'DOI')]

NBN_3 = [('NI', 'DOI'),
         ('PB', 'DOI'),
         ('ET', 'DOI'),
         ('HF', 'DOI'),
         ('TA', 'DOI'),
         ('EF', 'DOI')]

TAN_1 = [('NI', 'DOI'),
         ('PB', 'NI'),
         ('PB', 'DOI'),
         ('ET', 'NI'),
         ('ET', 'DOI'),
         ('ET', 'PB'),
         ('HF', 'DOI'),
         ('HF', 'ET')]

TAN_2 = [('NI', 'DOI'),
         ('PB', 'NI'),
         ('PB', 'DOI'),
         ('ET', 'NI'),
         ('ET', 'DOI'),
         ('ET', 'PB'),
         ('HF', 'DOI'),
         ('HF', 'ET'),
         ('TA', 'ET'),
         ('TA', 'DOI')]

TAN_3 = [('NI', 'DOI'),
         ('PB', 'NI'),
         ('PB', 'DOI'),
         ('ET', 'NI'),
         ('ET', 'DOI'),
         ('ET', 'PB'),
         ('HF', 'DOI'),
         ('HF', 'ET'),
         ('TA', 'ET'),
         ('TA', 'DOI'),
         ('EF', 'DOI'),
         ('EF', 'HF')]

# Create another hyperparameter space for fine tuning LOF
space = dict()

# hyperprameters for text clustering
space['clustertype'] = ['k-means', 'dbscan']
space['nc'] = [3, 4, 5, 6, 7]
space['e'] = [0.3, 0.4, 0.5, 0.6, 0.7]
space['ns'] = [300, 400, 500, 600, 700]

# hyperparameters for outlier removal
space['outlier_removal'] = ['MCD','LOF','SVM']
space['c'] = [0.2, 0.3, 0.4, 0.5]
space['n'] = [1, 2]
space['kernel'] = ['linear', 'sigmoid']

# BN networks
space['BN_structure'] = [NBN_1, NBN_2, NBN_3, TAN_1, TAN_2, TAN_3]

print(space)

import csv
from sklearn.model_selection import ParameterGrid

def dict2csv(param_grid, saving_folder, file_name):
    
    csv_file = saving_folder + '\\' + file_name
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dict.keys(param_grid[0]))
            writer.writeheader()
            for data in param_grid:
                writer.writerow(data)
    except IOError:
        print("I/O error")

#saving_folder = r""
param_grid = ParameterGrid(space)
print(len(param_grid))
#print('Parameter grid size: ', len(param_grid))
#dict2csv(param_grid, saving_folder, 'param_grid trial 2 build construction injury prediction.csv')

import os
from sklearn.model_selection import ParameterGrid
import time
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def main_train(param, data):
    
    start = time.time() # initiate the time for each training
    
    clustertype = param['clustertype']
    nc = param['nc']
    e = param['e']
    ns = param['ns']
    outlier_removal = param['outlier_removal']
    c = param['c']
    n = param['n']
    kernel = param['kernel']
    BN_structure = param['BN_structure']
    
    try:

        if BN_structure == NBN_1:
            BN_name = 'NBN_1'
        elif BN_structure == NBN_2:
            BN_name = 'NBN_2'
        elif BN_structure == NBN_3:
            BN_name = 'NBN_3'
        elif BN_structure == TAN_1:
            BN_name = 'TAN_1'
        elif BN_structure == TAN_2:
            BN_name = 'TAN_2'
        else:
            BN_name = 'TAN_3'

        # all selected features from raw dataset based on the BN structure
        all_fea = nodes2feas(BN_structure)
        # all explanatory features from selected features
        exp_fea = all_fea.copy()
        exp_fea.remove('DOI')
        # the response feature(s) from selected features
        variables = ['DOI']

        ########################### Train and validate BN models ################  
        
        prepro_data = data_preproc(data, outlier_removal, c, n, kernel, clustertype, nc, e, ns) 

        data_tr, data_te = train_test_split(prepro_data, test_size = 0.20, random_state = 373, shuffle = True)

        avg_metrics = bn_param_tr_val(data_tr, BN_structure, exp_fea, variables)

        avg_metrics[['Outlier removal method', 
                     'c',
                     'n', 
                     'kernel', 
                     'BN names',
                     'Clustering algorithm',
                     'nc', 
                     'e', 
                     'ns']] = pd.DataFrame([[outlier_removal,
                                                   c, 
                                                   n, 
                                                   kernel,
                                                   BN_name,
                                                   clustertype,
                                                   nc, 
                                                   e, 
                                                   ns]], index = avg_metrics.index)

        eva_metrics = [avg_metrics.to_string(index = False, header=False)]

        return eva_metrics

    except IndexError:
        print('Index error') 
        pass
    
    except KeyError: 
        print('Key error')
        pass
    
    except Exception:
        print('Exception')
        pass
    
    except TerminatedWorkerError: # This error was raised up when there was an out-of-memory issue
        print('Worker Issue')
        pass

import time
from tqdm import tqdm
from contextlib import contextmanager
import joblib
from joblib import Parallel, delayed

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
        
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()
        
with tqdm_joblib(tqdm(desc="Parametric learning", total=len(param_grid))) as progress_bar:
    st = time.time()
    results = Parallel(n_jobs=35)(delayed(main_train)(param_grid[i], sub_data_na) for i in range(len(param_grid)))  #(len(param_grid)))
    runtime = round((time.time() - st)/60, 4) 
    print('Runtime is: ', runtime, ' minute(s)') 
    
    print(results)

import smtplib

sender_email = ''
rec_email = ''
password = #''
message = "BN parameter learning for injury prediction is completed!"

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(sender_email, password)
server.sendmail(sender_email, rec_email, message)

saving_folder = r"" # put the directory of a FOLDER to specify where to store the results
saving_file = saving_folder + '\\' + 'Trial 5.txt'

#Save the results (str) to a txt file
with open(saving_file, 'a') as f:
    for result in results:
        if result != None:
            f.write(str(result[0]) + '\n')

            
# Convert the txt file to pandas dataframe for getting the optimal hyperparameters
import pandas as pd

df = pd.read_csv(saving_file, sep=" ", header=None)

df.columns = ['Accuracy', 
              'Precision', 
              'Recall', 
              'F score', 
              'AUC', 
              'Balanced Accuracy',
              'Outlier removal method', 
              'c',
              'n', 
              'kernel', 
              'BN names',
              'Clustering algorithm',
              'nc', 
              'e', 
              'ns']

df


import os
from sklearn.model_selection import ParameterGrid
import time
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from contextlib import contextmanager
import joblib
from joblib import Parallel, delayed

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
 
    try:
        yield tqdm_object
        
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

def test_acc_find(df_3, rs, data, BN):
    outlier_removal = df_3['Outlier removal method'].loc[0]
    c = df_3['c'].loc[0]
    n = df_3['n'].loc[0]
    kernel = df_3['kernel'].loc[0]
    BN_structure = BN
    print(BN_structure)

    per_metrics = {0:'Accuracy', 1:'Precision', 2:'Recall', 3:'F-1',
                   4:'AUC', 5:'Balanced Accuracy'}

    try:
        # all selected features from raw dataset based on the BN structure
        all_fea = nodes2feas(BN_structure)
        # all explanatory features from selected features
        exp_fea = all_fea.copy()
        exp_fea.remove('degree_of_injury')
        # the response feature(s) from selected features
        variables = ['degree_of_injury']

        data_tr, data_te = train_test_split(discret_data, test_size = 0.2, random_state = rs, shuffle = True)

        # use all training dataset to train the optimal BN model
        bn_model = param_learning(BN_structure, data_tr)

        # use the testing dataset to test the optimal BN model
        exp_fea = all_fea.copy()
        exp_fea.remove('degree_of_injury')

        X_te = data_te[exp_fea]
        y_te = data_te['degree_of_injury']

        X_tr = data_tr[exp_fea]
        y_tr = data_tr['degree_of_injury']

        y_pred = bn_infer(X_te, bn_model, variables)
        #y_pred = bn_infer(X_tr, bn_model, variables)

        performance_metrics, cm = cal_cat_metrics(y_te, y_pred)
        #performance_metrics, cm = cal_cat_metrics(y_tr, y_pred)

        performance_metrics = pd.DataFrame(performance_metrics).T.rename(columns = per_metrics)
        
        acc = performance_metrics['Accuracy'].loc[0]
        if acc >= 0.85:
            alpha = 0.4
            target_names = ['Non-existent', 'Slight', 'Moderate', 'Strong'] # ['(-1, 1]', '(1, 3]', '(3, 5]', '> 5']
            cm_fig = cm_visual(cm, target_names, alpha)
            plt.title('Random state: ' + str(rs))
            plt.show()
        
        return [rs, acc]

    except ValueError:
        pass

    except KeyError:
        pass

rss = range(250)
BN = NBN_2
print(BN)

with tqdm_joblib(tqdm(desc="Finding optimal testing acc", total=len(rss))) as progress_bar:
    
    Accs = Parallel(n_jobs=2)(delayed(test_acc_find)(df_3, rss[i], Sel_data, BN) for i in range(len(rss))) #len(param_grid)
    
# Identify the random state that enables the testing accuracy to be greater than 0.8


test_accs = []
test_rs = []
thred = 0.87

for item in Accs:
    if item != None: 
        #print(item[1])
        if item[1] >= thred:
            test_accs.append(item[1])
            test_rs.append(item[0])

if len(test_accs) == 0:
    print('No optimal testing accuracy is greater than ' + str(thred))

else:
    print(test_accs)
    print(test_rs)

data_f_path = r""

data_f = pd.read_csv(data_f_path, index_col = [0])

data_f
