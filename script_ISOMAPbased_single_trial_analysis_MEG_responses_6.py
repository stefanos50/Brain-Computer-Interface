import numpy as np
import scipy
from sklearn import manifold
from matplotlib import pyplot as plt
from scipy import signal
from scipy.spatial.distance import pdist
from sklearn.manifold import Isomap
import HelperMethods

Fs = 625;
def run_script(side='left'):
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
    results = []


    left_trials,t = HelperMethods.signal_to_trials(subj3_left,stim_times,100,300)
    right_trials,t = HelperMethods.signal_to_trials(subj3_right,stim_times,100,300)
    results.append(t)
    print("stimtimes shape: "+str(stim_times.shape))
    print("subj3_left shape: "+str(subj3_left.shape))
    print("t shape: "+str(t.shape))
    print("left trials shape: "+str(left_trials.shape))

    Wn=np.array([3/(Fs/2) ,20/(Fs/2)])
    b, a = signal.butter(5, Wn, btype='pass')
    filt_left_trials = signal.filtfilt(b, a, left_trials)
    filt_right_trials = signal.filtfilt(b, a, right_trials)

    if side=="right":
        filt_side_trials = filt_right_trials
    else:
        filt_side_trials = filt_left_trials

    figure, axis = plt.subplots(2,2)

    axis[0,0].plot(t*(1/Fs),filt_side_trials.T)
    axis[0,0].set_title("single-trial M100 responses")
    axis[0,0].set_ylabel("")
    axis[0,0].set_xlabel("sec")

    axis[1,0].plot(t*(1/Fs),np.mean(filt_side_trials,axis=0),c='red')
    axis[1,0].set_title("single-trial M100 responses (averaged)")
    axis[1,0].set_ylabel("")
    axis[1,0].set_xlabel("time (s)")

    axis[0,1].plot(t,filt_side_trials.T)
    axis[0,1].set_title("single-trial M100 responses")
    axis[0,1].set_ylabel("")
    axis[0,1].set_xlabel("sample no")


    axis[1,1].plot(t,np.mean(filt_side_trials,axis=0),c='red')
    axis[1,1].plot([t[i] for i in range(0,len(t),10)],np.zeros(len([t[i] for i in range(0,len(t),10)])),c='black',marker='.')
    axis[1,1].set_title("single-trial M100 responses (averaged)")
    axis[1,1].set_ylabel("")
    axis[1,1].set_xlabel("sample no")

    plt.show()
    results.append(np.mean(filt_side_trials,axis=0))

    ave = np.mean(filt_side_trials,axis=0)
    latencies=[x for x in range(141,188)]
    DataMatrix=filt_side_trials[:,latencies]

    figure, axis = plt.subplots(2,1)

    axis[0].plot([x for x in range(len(ave))],ave)
    axis[0].plot(latencies,ave[latencies],c='red',marker='.')
    axis[0].set_title("the latencies of interest")
    axis[0].set_ylabel("")
    axis[0].set_xlabel("")

    axis[1].plot([x for x in range(len(ave))],filt_side_trials.T)
    axis[1].plot(latencies,filt_side_trials[:,latencies].T,color='red')
    axis[1].set_title("the extracted ST-segmens")
    axis[1].set_ylabel("")
    axis[1].set_xlabel("")

    plt.show()

    Ntrials = DataMatrix.shape[0]
    dimensionality = DataMatrix.shape[1]
    print("size of DataMatrix: "+str((Ntrials,dimensionality)))

    D = scipy.spatial.distance.squareform(pdist(DataMatrix))
    print('size of distance matrix: '+str(D.shape))

    embedding = Isomap(n_components=2,n_neighbors=19)
    X_transformed = embedding.fit(D)
    A = X_transformed.dist_matrix_


    mds = manifold.MDS(2,random_state=0,dissimilarity="precomputed")
    M = mds.fit_transform(A)
    Projected_trials = M[:,0:2]

    results.append(Projected_trials)

    figure, axis = plt.subplots(1,2)

    axis[0].imshow(A)
    axis[0].set_title("")
    axis[0].set_ylabel("trial no")
    axis[0].set_xlabel("trial no")

    for i in range(len(Projected_trials)):
        axis[1].plot(Projected_trials[i,0],Projected_trials[i,1],color='red',marker='.')
    for i in range(len(Projected_trials)):
        axis[1].annotate(str(i+1), (Projected_trials[i,0],Projected_trials[i,1]))
    axis[1].set_title("reduced dimensionality map")
    axis[1].set_ylabel("2nd dimension")
    axis[1].set_xlabel("1st dimension")

    plt.show()


    from sklearn.cluster import KMeans
    no_groups = 5
    kmeans = KMeans(n_clusters=no_groups)
    label = kmeans.fit_predict(Projected_trials)
    u_labels = np.unique(label)

    Prototypes=[]
    for ii in range(no_groups):
        list=(label==ii)
        Prototypes.append(np.mean(filt_side_trials[list,:],axis=0))
    Prototypes = np.array(Prototypes)


    figure, axis = plt.subplots(1,2)
    selected_colors = ['red','blue','green','orange','black','purple']
    current_color = 0
    for i in u_labels:
        axis[0].scatter(Projected_trials[label == i, 0], Projected_trials[label == i, 1], label=i,color=selected_colors[current_color])
        current_color+=1
    axis[0].legend()
    axis[0].set_title("ISOMAP-kMeansClustering")
    axis[0].set_ylabel("2nd Dimension")
    axis[0].set_xlabel("1st Dimension")

    current_color = 0
    padding = 0
    for i in range(len(Prototypes)):
        axis[1].plot(t*(1/Fs),Prototypes[i,:].T+padding,color=selected_colors[current_color])
        results.append(Prototypes[i,:].T+padding)
        padding += 2
        current_color += 1
    axis[1].axvline(0.1, color="blue", alpha=0.5)
    axis[1].set_title("prototypical (within-group averaged)  responses")
    axis[1].set_ylabel("")
    axis[1].set_xlabel("time (s)")

    plt.show()
    return results

left_res = run_script('left')
right_res = run_script('right')

figure, axis = plt.subplots(2, 3)
axis[0,0].plot(left_res[0]*(1/Fs),left_res[1],color='black',label='left')
axis[0,0].plot(right_res[0]*(1/Fs),right_res[1],color='red',label='right')
axis[0,0].axhline(y = 0.0, color = 'black', linestyle = 'dotted',linewidth=3)
axis[0,0].set_title("single-trial M100 responses (averaged) (left vs right)")
axis[0,0].axvline(0.1, color="blue", alpha=0.5)
axis[0,0].legend(loc="upper left")
axis[0,0].set_xlabel("time (s)")
axis[0,1].scatter(left_res[2][:,0], left_res[2][:,1],color='black',label='left')
axis[0,1].scatter(right_res[2][:,0], right_res[2][:,1],color='red',label='right')
axis[0,1].set_title("reduced dimensionality map (left vs right)")
axis[0,1].legend(loc="upper left")
axis[0,1].set_xlabel("time (s)")
for i in range(3,len(left_res)):
    if i==3:
        axis[0,2].plot(left_res[0] * (1 / Fs), left_res[i], color='black',label='left')
        axis[0,2].plot(right_res[0] * (1 / Fs), right_res[i], color='red',label='right')
    else:
        axis[0,2].plot(left_res[0] * (1 / Fs), left_res[i], color='black')
        axis[0,2].plot(right_res[0] * (1 / Fs), right_res[i], color='red')
    axis[0,2].axvline(0.1, color="blue", alpha=0.5)
axis[0,2].set_title("prototypical (within-group averaged)  responses (left vs right)")
axis[0,2].legend(loc="upper left")

padding = 0
selected_colors = ['red', 'blue', 'green', 'orange', 'black', 'purple']
axis[1,0].plot(left_res[0]*(1/Fs),left_res[1],color='dimgray',linewidth = '7',label="M100 responses (averaged)")
for i in range(3,len(left_res)):
    axis[1,0].plot(left_res[0] * (1 / Fs), left_res[i]+padding, color=selected_colors[i-3],label="Sub-Average group "+str(i-3+1))
    padding -= 2
axis[1,0].axvline(0.1, color="navy", alpha=0.5,linewidth = '5')
axis[1,0].legend(loc="upper right")
axis[1,0].set_title("Averaging & Subaverages comparison Left")
axis[1,0].set_xlabel("time (s)")
padding = 0
selected_colors = ['red', 'blue', 'green', 'orange', 'black', 'purple']
axis[1,1].plot(right_res[0]*(1/Fs),right_res[1],color='dimgray',linewidth = '7',label="M100 responses (averaged)")
for i in range(3,len(right_res)):
    axis[1,1].plot(right_res[0] * (1 / Fs), right_res[i]+padding, color=selected_colors[i-3],label="Sub-Average group "+str(i-3+1))
    padding -= 2
axis[1,1].axvline(0.1, color="navy", alpha=0.5,linewidth = '5')
axis[1,1].legend(loc="upper right")
axis[1,1].set_title("Averaging & Subaverages comparison Right")
axis[1,1].set_xlabel("time (s)")
figure.delaxes(axis.flatten()[5])
plt.show()
