import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
from sklearn.covariance import MinCovDet, EmpiricalCovariance

import HelperMethods
from sklearn import manifold

#Load the data
mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\VEP_responses__worked_examples\\VEPS.MAT')
veps = mat['veps']
veps = np.array(veps)

num_trials,num_samples=veps.shape
Fs=1000
t=[x for x in range(1,num_samples+1)]
t = np.array(t)

HelperMethods.create_plot((2,1),[t,t*(1/Fs)],[np.mean(veps,axis=0),np.mean(veps,axis=0)],['averaged response',''],[0,1],['sample no','time (s)'],['a.u.(volts)',''])

VEPS = signal.detrend(veps, type='linear', bp=0)

#create_plot((2,1),[t,t*(1/Fs)],[np.mean(VEPS,axis=0),np.mean(VEPS,axis=0)],['averaged response after detrend',''],[0,1],['sample no','time (s)'],['a.u.(volts)',''])


np,sp = HelperMethods.snr_sample(VEPS)
trial_SNR=sp/np
ave_SNR=num_trials*trial_SNR

print('signal-to-noise ratio per trial: '+str(trial_SNR))
print('SNR of the averaged response: '+str(ave_SNR))

Dmatrix = scipy.spatial.distance.squareform(pdist(VEPS))
print('size of distance matrix: '+str(Dmatrix.shape))


mds = manifold.MDS(2,random_state=0,dissimilarity="precomputed")
M = mds.fit_transform(Dmatrix)
Projected_trials = M[:,0:2]
#print(Projected_trials)

# Import libraries.
from sklearn.covariance import EllipticEnvelope
# Fit detector.
detector = EllipticEnvelope(contamination=0.113, random_state=0)
is_outlier = detector.fit(Projected_trials).predict(Projected_trials)
outlier_indexes = []
kept_indexes = []

for i in range(len(is_outlier)):
    if is_outlier[i] == -1:
        outlier_indexes.append(i)
    else:
        kept_indexes.append(i)
print("The outlier candidates are: "+str(outlier_indexes))



# Placing the plots in the plane
plot1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
plot2 = plt.subplot2grid((3, 3), (0, 2), rowspan=3, colspan=2)
plot3 = plt.subplot2grid((3, 3), (1, 0), rowspan=2)

for i in range(len(VEPS)):
    plot1.plot(t, VEPS[i])
plot1.set_title("single-trial waveforms")
plot1.set_xlabel("time (sample no)")
plot1.set_ylabel("")

plot2.imshow(Dmatrix)
plot3.set_title("Distance Matrix")
plot2.set_ylabel("trial no")
plot2.set_xlabel("trial no")

Dist_Score= sum(Dmatrix)

plot3.plot(Projected_trials[:,0],Projected_trials[:,1], 'o')
plot3.set_title("")
plot3.set_ylabel("Aggregade distance")
plot3.set_xlabel("trial no")

for i in range(1,len(Projected_trials)+1):
    plot3.annotate(str(i), (Projected_trials[i-1][0],Projected_trials[i-1][1]))

plt.show()

plt.plot(Projected_trials[kept_indexes,0], Projected_trials[kept_indexes,1],'.',c='black',markersize=5)
plt.plot(Projected_trials[outlier_indexes,0], Projected_trials[outlier_indexes,1],'D',c='red',markersize=6)
for i in range(len(outlier_indexes)):
    plt.annotate(str(outlier_indexes[i]+1), (Projected_trials[outlier_indexes[i]][0],Projected_trials[outlier_indexes[i]][1]))
plt.show()


figure, axis = plt.subplots(1,2)

for i in range(len(kept_indexes)):
    axis[0].plot([x for x in range(len(VEPS[i]))],VEPS[kept_indexes[i]],c='black')
for i in range(len(outlier_indexes)):
    axis[0].plot([x for x in range(len(VEPS[i]))], VEPS[outlier_indexes[i]], c='red')
axis[0].set_title("single times waveforms")
axis[0].set_ylabel("")
axis[0].set_xlabel("time (sample no)")

mean_kept = VEPS[kept_indexes,:].mean(axis=0)
mean = VEPS.mean(axis=0)


axis[1].plot([x for x in range(len(mean))],mean,c='red',label='ensemble-averaging')
axis[1].plot([x for x in range(len(mean_kept))], mean_kept, c='blue',label='selective-averaging')
axis[1].set_title("")
axis[1].set_ylabel("")
axis[1].set_xlabel("time (sample no)")

plt.legend(loc='best')
plt.show()