

import tensorflow as tf
import os
import re

savedir = "models"

networks = ['cnn']

r = [2,4,6,8]
l = [2,3,4]
f = [0]

image_filepath = os.path.join("doc","results","tmp","images")
table_filepath = os.path.join("doc","results","tmp","tables")

if not os.path.exists(image_filepath):
    os.makedirs(image_filepath)
    
if not os.path.exists(table_filepath):
    os.makedirs(table_filepath)

dataset = "test"
rolling_mean_window = 10

def params2name(l,r,f):
    return "{}l{}r50d{}f".format(l,r,f)

def name2param(name):
    l,r,d,f = re.findall('\d+', name)
    return l,r,d,f

runs = [] # expected runs
for r_ in r:
    for l_ in l:
        for f_ in f:
            runs.append(params2name(l_,r_,f_))

def fix_typos(classes):
    classes[3] = classes[3].replace("_"," ") # cloud_shadow -> cloud shadow
    classes[22] = "sugar beet" # org. sugar beets
    return classes

print ("ok")

#extraer datos 
def extract_from_eventfile(eventfile_path, tag='train/cross_entropy'):
    steps = []
    values = []

    for e in tf.train.summary_iterator(eventfile_path):
        for v in e.summary.value:
            if v.tag == tag:
                steps.append(e.step)
                values.append(v.simple_value)

    return steps, values

def extract_from_all_eventfiles(path, tag='train/cross_entropy'):
    """ 
    appends values from all event files in one folder
    if path does not exist: returns empty list
    """
    steps = []
    values = []
    
    if os.path.exists(path):
        eventfiles = os.listdir(path)
    else:
        return steps, values # empty

    for eventfile in eventfiles:
        steps_,values_ = extract_from_eventfile(os.path.join(path,eventfile), tag=tag)
        steps.extend(steps_)
        values.extend(values_)
        
    return steps, values

def gather_data_from_multiple_runs(folder_path, runs, dataset="test", tag='train/cross_entropy'):
    """
    look through all save folders defined by runs, extract eventfiles from runs
    and append everything to pandas dataframe
    """

    series = []
    for run in runs:
        path = os.path.join(folder_path,run,dataset)
        steps,values = extract_from_all_eventfiles(path, tag)
        print("run: {} extracted {} values".format(run,len(values)))   

        if len(values) > 0:

            s = pd.Series(data=values, index=steps,name=run).sort_index()
            # drop duplicates
            s = s[~s.index.duplicated(keep='last')]

            #.drop_duplicates(keep='last')
            series.append(s)
            
    return pd.concat(series,axis=1,join="outer")


import pandas as pd
import numpy as np

def smooth_and_interpolate(data, rolling_mean_window = 10):
    data.interpolate(axis=1,inplace=True)
    return data.rolling(window=rolling_mean_window,axis=0).mean()

def get_best_run(data,max_is_better=False):
    scores = []
    for col in data.columns: 
        s = data[col]
        scores.append(s.loc[s.last_valid_index()])
    
    if max_is_better:
        return data.columns[np.array(scores).argmax()]
    else:
        return data.columns[np.array(scores).argmin()]

datasets = []
best_runs = []

# cross entropy
max_is_better = False
tag='train/cross_entropy'
# for lstm/rnn/cnn
for network in networks: 
    print
    print network
    d = gather_data_from_multiple_runs(os.path.join(savedir,network),runs,dataset=dataset,tag=tag)
    best_runs.append(get_best_run(d, max_is_better))
    d = smooth_and_interpolate(d,rolling_mean_window)
    datasets.append(d)



for best_run, network in zip(best_runs, networks):
    print "Network {}: best run {}".format(network, best_run)
print datasets




import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import rgb2hex
#get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import numpy as np

sns.set(context='notebook', style='whitegrid', palette='deep', font='Times', font_scale=1, color_codes=False, rc=None)

tumivory = rgb2hex((218./255, 215./255, 203./255))

tumblue = rgb2hex((0., 101./255, 189./255))
tumgreen = rgb2hex((162./255, 173./255, 0))
tumorange = rgb2hex((227./255, 114./255, 34./255))

tumbluelight=rgb2hex((152./255, 198./255, 234./255))
tumbluemedium=rgb2hex((100./255, 160./255, 200./255))
tumbluedark=rgb2hex((0, 82./255, 147./255))

figsize=(11,9)
xlim = (0,7.5e6)

pdf_filepath = os.path.join(image_filepath,"shadedplot.pdf")
tikz_filepath = os.path.join(image_filepath,"shadedplot.tikz")


def plot_network_runs(data, ax=None, best_run = None, col="#cccccc", std_alpha=0.5, label="LSTM"):
    if ax is None:
        f, ax = plt.subplots()
    
    std = data.std(axis=1)
    
    patch = mpatches.Patch(color=col, label=label)
    
    mean = data.mean(axis=1)
    p_mean, = ax.plot(mean, color=col, linestyle="dashed", label="mean")
    
    #median = data.median(axis=1)
    #ax.plot(median,label="median", color=col, linestyle="dotted")
    
    p_best, = ax.plot(data[best_run], color=col, label="best")

    p_std = ax.fill_between(mean.index, mean-std, mean+std, where=mean+std >= mean-std, 
                            interpolate=True, color=col, alpha=std_alpha, label="std")
    
    
    
    # xlabels
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.set_xlim(*xlim)
    
    return ax, [patch,p_best,p_mean,p_std], [mean, std, data[best_run]]
    
f, ax = plt.subplots(figsize=figsize)
ax, handles2, dat_cnn = plot_network_runs(datasets[0], 
                       ax=ax, 
                       best_run=best_runs[0], 
                       col=tumgreen, 
                       label="CNN")

ax.legend(handles=handles2,ncol=len(networks))
ax.set_xlabel("Caracteristicas entrenadas")
ax.set_ylabel("cross entropy")

f.savefig(pdf_filepath,transparent=True)


dat_filepath_cnn = os.path.join(image_filepath,"shadedplot_cnn.dat")

cnn = pd.DataFrame(dat_cnn, index=["mean", "std", "best"]).transpose()


cnn.dropna().to_csv(dat_filepath_cnn, sep=' ', header=True)
plt.show()

# ### all networks
# 
# plot data from all networks

# In[53]:

def plot_all_runs(data, ax=None, col="#cccccc", std_alpha=0.5, label_std=r"1$\sigma$ std", label="best run", title="title"):
    if ax is None:
        f, ax = plt.subplots()
    
    std = data.std(axis=1)
    mean = data.mean(axis=1)

    runs = data.columns
    for run in runs:
        ax.plot(data[run],label=run)

    ax.fill_between(mean.index, mean-std, mean+std, where=mean+std >= mean-std, interpolate=True, facecolor=col, alpha=std_alpha, label=label_std)
    
    # xlabels
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.set_xlim(0,1e7)
    ax.set_title(title)
    
    return ax

figsize=(15,10)
for data, network in zip(datasets, networks):
    f, ax = plt.subplots(figsize=figsize)
    plot_all_runs(data, ax=ax, col="#cccccc", std_alpha=0.5, label_std=r"1$\sigma$ std", label="best run", title=network)
    ax.legend()

plt.show()




from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support

def gather_accuracy_values_per_class(classes,targets,scores):
    """
    Gather per class a variety of accuracy metrics from targets and scores
    """
    y_pred = np.argmax(scores,axis=1)
    y_true = np.argmax(targets,axis=1)
    
    precision_, recall_, fscore_, support_ = precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None)

    fscore = pd.Series(index=classes, data=fscore_, name="f-score")
    precision = pd.Series(index=classes, data=precision_, name="precision")
    recall = pd.Series(index=classes, data=recall_, name="recall")
    support = pd.Series(index=classes, data=support_, name="support")
    
    s = [fscore,precision,recall, support]
    names = [el.name for el in s]
    return pd.DataFrame(zip(*s), columns=names, index=recall.index).T




import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, cohen_kappa_score, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
 
def gather_mean_accuracies(classes, scores, targets, average='weighted', label="label", b=4):
    """
    calculate a series for mean accuracy values for all, covered (class id < b) and fields (class id > b)
    """
    metrics = []
    
    y_pred = np.argmax(scores,axis=1)
    y_true = np.argmax(targets,axis=1)
    
    all_mask = np.ones(y_true.shape)
    covered_mask = y_true<b
    field_mask = y_true>=b
    
    # class weighted average accuracy
    w_all = np.ones(y_true.shape[0])
    for idx, i in enumerate(np.bincount(y_true)):
        w_all[y_true == idx] *= (i/float(y_true.shape[0]))
        
    w_cov = np.ones(y_true[covered_mask].shape[0])
    for idx, i in enumerate(np.bincount(y_true[covered_mask])):
        w_cov[y_true[covered_mask] == idx] *= (i/float(y_true[covered_mask].shape[0]))
        
    w_field = np.ones(y_true[field_mask].shape[0])
    for idx, i in enumerate(np.bincount(y_true[field_mask])):
        w_field[y_true[field_mask] == idx] *= (i/float(y_true[field_mask].shape[0]))
        
    w_acc = accuracy_score(y_true, y_pred, sample_weight=w_all)
    w_acc_cov = accuracy_score(y_true[covered_mask], y_pred[covered_mask], sample_weight=w_cov)
    w_acc_field = accuracy_score(y_true[field_mask], y_pred[field_mask], sample_weight=w_field)
    
    metrics.append(pd.Series(data=[w_acc, w_acc_cov, w_acc_field], dtype=float, name="accuracy"))
    
    # AUC
    try:
        # if AUC not possible skip
        auc = roc_auc_score(targets, scores, average=average)
        auc_cov = roc_auc_score(targets[covered_mask,:b], scores[covered_mask,:b], average=average)
        auc_field = roc_auc_score(targets[field_mask,b:], scores[field_mask,b:], average=average)

        metrics.append(pd.Series(data=[auc, auc_cov, auc_field], dtype=float, name="AUC"))
    except:
        print "no AUC calculated"
        pass
    
    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    kappa_cov = cohen_kappa_score(y_true[covered_mask], y_pred[covered_mask])
    kappa_field = cohen_kappa_score(y_true[field_mask], y_pred[field_mask])
    
    metrics.append(pd.Series(data=[kappa, kappa_cov, kappa_field], dtype=float, name="kappa"))
    
    # Precision, Recall, F1, support
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, beta=1, average=average)
    prec_cov, rec_cov, f1_cov, support_cov = precision_recall_fscore_support(y_true[covered_mask], y_pred[covered_mask], beta=1, average=average)
    prec_field, rec_field, f1_field, support_field = precision_recall_fscore_support(y_true[field_mask], y_pred[field_mask], beta=1, average=average)
    
    metrics.append(pd.Series(data=[prec, prec_cov, prec_field], dtype=float, name="precision"))
    metrics.append(pd.Series(data=[rec, rec_cov, rec_field], dtype=float, name="recall"))
    metrics.append(pd.Series(data=[f1, f1_cov, f1_field], dtype=float, name="fscore"))
    #sup_ = pd.Series(data=[support, support_cov, support_field], dtype=int, name="support")
        
    df_ = pd.DataFrame(metrics).T
    if label is not None:
        df_.index = [[label,label,label],["all","cov","fields"]]
    else:
        df_.index = ["all","cov","fields"]
    
    return df_



from sklearn.metrics import confusion_matrix

# border in the classes between field classes and coverage
b = 4

obs_file = "eval_observations.npy"
probs_file = "eval_probabilities.npy"
targets_file = "eval_targets.npy"
conf_mat_file = "eval_confusion_matrix.npy"
class_file = "classes.npy"

# drop fc for now:
#networks = [networks[0], networks[2]]
#best_runs = [best_runs[0], best_runs[2]]

networklabels = ["CNN"]

#over_accuracy_label = "ov. accuracy2"

# ignore <obs_limit> first observations
obs_limit = 0
mean_df=[]
acc=[]
mean = []
for best_run, network, label_ in zip(best_runs,networks,networklabels):
    print network
   
    path = os.path.join(savedir,network,best_run)
    #obs = np.load(os.path.join(path,obs_file))
    scores = np.load(os.path.join(path,probs_file))
    targets = np.load(os.path.join(path,targets_file))
    #if os.path.exists(os.path.join(path,conf_mat_file)):
    #    cm = np.load(os.path.join(path,conf_mat_file))
    #else:
    y_pred = np.argmax(scores,axis=1)
    y_true = np.argmax(targets,axis=1)
    cm = confusion_matrix(y_true,y_pred)


    classes = fix_typos(
                    list(np.load(os.path.join(path,class_file)))
            )
    
    #df_, a_ = acc_mean_accuracies(cm, classes, label_, b, scores,targets)
    df_ = gather_mean_accuracies(classes, scores, targets, b=b, label=label_)
    
    mean.append(df_)
    
mean_df = pd.concat(mean)
print mean_df


# In[59]:

# CNN
gather_accuracy_values_per_class(classes,targets,scores)


# In[60]:

best_runs, networks
table=[]


# ### SVM
# 
# evaluation of SVM baseline. 
# SVM baseline is calculated at ```SVM.py```. Results are saved as ```.npy```

# In[61]:

from sklearn.metrics import roc_curve
# load from SVM Evaluation.ipynb

# cm = cm_SVM = np.load("svm/confusion_matrix.npy")
# scores = np.load("svm/scores.npy")
# targets = np.load("svm/targets.npy")
# pred = np.load("svm/predicted.npy")

#df = gather_mean_accuracies(classes, scores, targets, b=b, label="CNN")


# In[62]:

# drop previous SVM if exists
#table_df=[]
#table_df = pd.concat([mean_df,df])
table=mean_df
table = table.round(3) * 100


# In[63]:

# put overall accuracy first
#overall_acc = table[over_accuracy_label]
#new = table.drop([over_accuracy_label],axis=1)
#new.insert(0,over_accuracy_label,overall_acc)
#table = new.transpose()
table = table.transpose()
print table






