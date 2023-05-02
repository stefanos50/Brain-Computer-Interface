import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist
import HelperMethods

#Load the data
mat = scipy.io.loadmat('K:\\DataLearningSignalAnalytics\\VEP_responses__worked_examples\\VEPS.MAT')
veps = mat['veps']
veps = np.array(veps)
#veps = veps[0:50,:]



num_trials,num_samples=veps.shape
Fs=1000
t=[x for x in range(1,num_samples+1)]
t = np.array(t)

HelperMethods.create_plot((2,1),[t,t*(1/Fs)],[np.mean(veps,axis=0),np.mean(veps,axis=0)],['averaged response',''],[0,1],['sample no','time (s)'],['a.u.(volts)',''])

VEPS = signal.detrend(veps, type='linear', bp=0)

#create_plot((2,1),[t,t*(1/Fs)],[np.mean(VEPS,axis=0),np.mean(VEPS,axis=0)],['averaged response after detrend',''],[0,1],['sample no','time (s)'],['a.u.(volts)',''])


figure, axis = plt.subplots(1, 2)
# plot lines
for i in range(len(veps)):
    axis[0].plot(t, veps[i])
axis[0].plot(t, np.mean(veps,axis=0), linewidth = '5',c="black",label = "averaged response original")
axis[0].set_title("single trials responses - original")
axis[0].set_xlabel("sample no")
axis[0].set_ylabel("a.u. (volts)")

for i in range(len(VEPS)):
    axis[1].plot(t, VEPS[i])
axis[1].plot(t, np.mean(VEPS,axis=0), linewidth = '5',c="black",label = "averaged response after dc-offset")
axis[1].set_title("after applying dc-offset")
axis[1].set_xlabel("sample no")
axis[1].set_ylabel("a.u. (volts)")
plt.show()


np,sp = HelperMethods.snr_sample(VEPS)
trial_SNR=sp/np
ave_SNR=num_trials*trial_SNR

print('signal-to-noise ratio per trial: '+str(trial_SNR))
print('SNR of the averaged response: '+str(ave_SNR))

Dmatrix = scipy.spatial.distance.squareform(pdist(VEPS))
print('size of distance matrix: '+str(Dmatrix.shape))

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

plot3.stem(Dist_Score)
plot3.set_title("")
plot3.set_ylabel("Aggregade distance")
plot3.set_xlabel("trial no")

plt.show()


Ranked_Dist_Score = sorted(Dist_Score)

sorted_list = sorted(
    range(len(Dist_Score)),
    key=lambda index: Dist_Score[index]
)


out_to_remove = 6

outlier_list = sorted_list[-out_to_remove:]
print("The candidate outliers: "+str(outlier_list))

figure, axis = plt.subplots(1,2)
axis[0].plot(Ranked_Dist_Score, linestyle='-', marker='o', color='red', label='line with marker')
for i in range(len(Ranked_Dist_Score[-out_to_remove:])):
    axis[0].annotate(sorted_list[-out_to_remove:][i], (Ranked_Dist_Score.index(Ranked_Dist_Score[-out_to_remove:][i]),Ranked_Dist_Score[-out_to_remove:][i]))

axis[1].imshow(VEPS[sorted_list,:])
plot3.set_title("sorted ST-waveforms")
plot3.set_ylabel("ranked-waveforms")
plot3.set_xlabel("time (sample no)")
plt.show()

indices_all = [x for x in range(num_trials)]
kept_list = []
for i in range(len(indices_all)):
    if indices_all[i] in sorted_list[-out_to_remove:]:
        continue
    kept_list.append(indices_all[i])

figure, axis = plt.subplots(1,2)

for i in range(len(kept_list)):
    axis[0].plot([x for x in range(len(VEPS[i]))],VEPS[kept_list[i]],c='black')
for i in range(len(outlier_list)):
    axis[0].plot([x for x in range(len(VEPS[i]))], VEPS[outlier_list[i]], c='red')
axis[0].set_title("single times waveforms")
axis[0].set_ylabel("")
axis[0].set_xlabel("time (sample no)")


veps_mean_kept = VEPS[kept_list,:].mean(axis=0)
veps_mean = VEPS.mean(axis=0)


axis[1].plot([x for x in range(len(veps_mean))],veps_mean,c='red',label='ensemble-averaging')
axis[1].plot([x for x in range(len(veps_mean_kept))], veps_mean_kept, c='blue',label='selective-averaging')
axis[1].set_title("")
axis[1].set_ylabel("")
axis[1].set_xlabel("time (sample no)")

plt.legend(loc='best')
plt.show()


np,sp = HelperMethods.snr_sample(VEPS[kept_list,:])
selective_trial_SNR=sp/np

print('relative increase in trial-level SNR without removing decimals: '+str((selective_trial_SNR-trial_SNR)/trial_SNR))




