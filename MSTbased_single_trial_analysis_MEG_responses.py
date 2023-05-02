import math

import numpy as np
import nx as nx
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn import manifold
from matplotlib import pyplot as plt
from scipy import signal
from scipy.spatial.distance import pdist
import networkx as nx
import HelperMethods

mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\stimulation\\stim_times.mat')
stim_times = np.array(mat['stim_times'])

mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\stimulation\\subj3_left.mat')
subj3_left = np.array(mat['subj3_left'])

mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\stimulation\\subj3_right.mat')
subj3_right = np.array(mat['subj3_right'])

mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\control\\subj3_control_right.mat')
subj3_control_right = np.array(mat['subj3_control_right'])

mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\control\\subj3_control_left.mat')
subj3_control_left = np.array(mat['subj3_control_left'])

Fs=625;

left_trials,t = HelperMethods.signal_to_trials(subj3_left,stim_times,100,300)
right_trials,t = HelperMethods.signal_to_trials(subj3_right,stim_times,100,300)

Wn=np.array([3/(Fs/2) ,20/(Fs/2)])
b, a = signal.butter(5, Wn, btype='pass')
filt_left_trials = signal.filtfilt(b, a, left_trials)
filt_right_trials = signal.filtfilt(b, a, right_trials)

figure, axis = plt.subplots(2,2)

axis[0,0].plot(t*(1/Fs),filt_right_trials.T)
axis[0,0].set_title("single-trial M100 responses")
axis[0,0].set_ylabel("")
axis[0,0].set_xlabel("sec")

axis[1,0].plot(t*(1/Fs),np.mean(filt_right_trials,axis=0),c='red')
axis[1,0].set_title("single-trial M100 responses (averaged)")
axis[1,0].set_ylabel("")
axis[1,0].set_xlabel("time (s)")

axis[0,1].plot(t,filt_right_trials.T)
axis[0,1].set_title("single-trial M100 responses")
axis[0,1].set_ylabel("")
axis[0,1].set_xlabel("sample no")


axis[1,1].plot(t,np.mean(filt_right_trials,axis=0),c='red')
axis[1,1].plot([t[i] for i in range(0,len(t),10)],np.zeros(len([t[i] for i in range(0,len(t),10)])),c='black',marker='.')
axis[1,1].set_title("single-trial M100 responses (averaged)")
axis[1,1].set_ylabel("")
axis[1,1].set_xlabel("sample no")

plt.show()

ave = np.mean(filt_right_trials,axis=0)
latencies=[x for x in range(251,300)] #141 188
DataMatrix=filt_right_trials[:,latencies]

figure, axis = plt.subplots(2,1)

axis[0].plot([x for x in range(len(ave))],ave)
axis[0].plot(latencies,ave[latencies],c='red',marker='.')
axis[0].set_title("the latencies of interest")
axis[0].set_ylabel("")
axis[0].set_xlabel("")

axis[1].plot([x for x in range(len(ave))],filt_right_trials.T)
axis[1].plot(latencies,filt_right_trials[:,latencies].T,color='red')
axis[1].set_title("the extracted ST-segmens")
axis[1].set_ylabel("")
axis[1].set_xlabel("")

plt.show()

Ntrials = DataMatrix.shape[0]
dimensionality = DataMatrix.shape[1]
print("size of DataMatrix: "+str((Ntrials,dimensionality)))

D = scipy.spatial.distance.squareform(pdist(DataMatrix))
print('size of distance matrix: '+str(D.shape))

MST = minimum_spanning_tree(D,False)
MST = scipy.sparse.lil_matrix(MST).toarray()

rows, cols = np.where(MST > 0)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
labeldict = {}
for i in range(0, len(D)):
    labeldict[i] = str(i+1)
T=nx.minimum_spanning_tree(gr)
nx.draw(T, node_size=100, with_labels=True, labels=labeldict)
plt.show()

extremeties = []
node_degrees = []
for node, degree in T.degree():
  node_degrees.append(degree)
  if degree == 1:
      extremeties.append(node)

print('The extremeties :'+str(extremeties))
print('number of nodes with the smallest degree: '+str(len(extremeties)))
print('The node degrees: '+str(node_degrees))
print('Highest degree: '+str(max(node_degrees)))

selected=[87,68,55,60,62,95]

selected_colors = ['red','blue','green','orange','black','purple']
print('selected trials from the MDS map: '+str(selected))
padding = 0
current_color = 0

for i in range(len(filt_right_trials)):
    if i in selected:
        plt.plot(t*(1/Fs),filt_right_trials[i,:].T+padding,color=selected_colors[current_color])
        padding += 2
        current_color += 1

plt.title('Selected Single Trials')

plt.show()