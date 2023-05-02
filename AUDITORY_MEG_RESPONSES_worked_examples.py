

import matplotlib
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import signal
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


print("sampling frequency: "+str(Fs))
print('length of continuous MEG time-series (#samples): '+str(subj3_left.shape[0]))
print('length of continuous MEG time-series (in seconds): '+str(subj3_left.shape[0]/Fs))
print('number of stimuli: '+str(stim_times[0:stim_times.shape[0]-1].shape[0]))


figure, axis = plt.subplots(2,1)
axis[0].plot([x for x in range(len(subj3_left))],subj3_left,label='left')
axis[0].plot([x for x in range(len(subj3_right))],subj3_right+5,label='right')
axis[0].plot(stim_times,np.ones(len(stim_times))*-2,label='stim onset' ,marker='o', color='black')
axis[0].set_title("stimulation")
axis[0].set_ylabel("")
axis[0].set_xlabel("time (sample no)")

axis[1].plot([x for x in range(len(subj3_control_left))],subj3_control_left,label='left')
axis[1].plot([x for x in range(len(subj3_control_right))],subj3_control_right+5,label='right')
axis[1].set_title("spontaneous activity")
axis[1].set_ylabel("")
axis[1].set_xlabel("time (sample no)")

axis[0].legend(loc='upper left')
plt.show()


figure, axis = plt.subplots(2,2)
the_window = scipy.signal.get_window('hamming', 1024)


axis[0,0].psd(subj3_left.flatten(), NFFT=1024, Fs=Fs, Fc=None, detrend=None, window=the_window, noverlap=500, pad_to=None, sides=None, scale_by_freq=None, return_line=None, data=None)
axis[0,0].set_title("PSD stimulation: Left")
axis[0,0].set_ylabel("")
axis[0,0].set_xlabel("time (sample no)")

axis[0,1].psd(subj3_right.flatten(), NFFT=1024, Fs=Fs, Fc=None, detrend=None, window=the_window, noverlap=500, pad_to=None, sides=None, scale_by_freq=None, return_line=None, data=None)
axis[0,1].set_title("PSD stimulation: Right")
axis[0,1].set_ylabel("")
axis[0,1].set_xlabel("time (sample no)")

axis[1,0].psd(subj3_control_left.flatten(), NFFT=1024, Fs=Fs, Fc=None, detrend=None, window=the_window, noverlap=500, pad_to=None, sides=None, scale_by_freq=None, return_line=None, data=None)
axis[1,0].set_title("PSD spontaneus: Left")
axis[1,0].set_ylabel("")
axis[1,0].set_xlabel("time (sample no)")

axis[1,1].psd(subj3_control_right.flatten(), NFFT=1024, Fs=Fs, Fc=None, detrend=None, window=the_window, noverlap=500, pad_to=None, sides=None, scale_by_freq=None, return_line=None, data=None)
axis[1,1].set_title("PSD spontaneus: Right")
axis[1,1].set_ylabel("")
axis[1,1].set_xlabel("time (sample no)")

plt.show()


left_trials,t = HelperMethods.signal_to_trials(subj3_left,stim_times,100,300)
right_trials,t = HelperMethods.signal_to_trials(subj3_right,stim_times,100,300)



figure, axis = plt.subplots(2,2)
axis[0,0].plot(t*(1/Fs),HelperMethods.get_matrix_with_step(left_trials,10).T,c='blue')
axis[0,0].set_title("left-hemisphere single-trials")
axis[0,0].set_ylabel("")
axis[0,0].set_xlabel("time (s)")

axis[0,1].plot(t*(1/Fs),HelperMethods.get_matrix_with_step(right_trials,10).T,c='red')
axis[0,1].set_title("right-hemisphere single-trials")
axis[0,1].set_ylabel("")
axis[0,1].set_xlabel("time (s)")

axis[1,0].plot(t*(1/Fs),np.mean(left_trials,axis=0),c='blue')
axis[1,0].set_title("M100 averaged response Left")
axis[1,0].set_ylabel("")
axis[1,0].set_xlabel("time (s)")

axis[1,1].plot(t*(1/Fs),np.mean(right_trials,axis=0),c='red')
axis[1,1].set_title("M100 averaged response Right")
axis[1,1].set_ylabel("")
axis[1,1].set_xlabel("time (s)")

plt.show()

figure, axis = plt.subplots(1,2)

axis[0].imshow(left_trials,cmap = plt.cm.get_cmap("jet"))
axis[0].set_title("right-hemisphere single-trials")
axis[0].set_ylabel("")
axis[0].set_xlabel("time (sample no)")

axis[1].imshow(right_trials,cmap = plt.cm.get_cmap("jet"))
axis[1].set_title("right-hemisphere single-trials")
axis[1].set_ylabel("")
axis[1].set_xlabel("time (sample no)")
plt.show()

Wn=np.array([3/(Fs/2) ,20/(Fs/2)])
b, a = signal.butter(5, Wn, btype='pass')
filt_left_trials = signal.filtfilt(b, a, left_trials)
filt_right_trials = signal.filtfilt(b, a, right_trials)


figure, axis = plt.subplots(2,2)

axis[0,0].plot(t*(1/Fs),filt_left_trials[0,:],c='blue',label="filtered left trial no 1")
axis[0,0].plot(t*(1/Fs),left_trials[0,:],c='black',label="left trial no 1")
axis[0,0].plot(t*(1/Fs),filt_left_trials[9,:]+2,c='red',label="filtered left trial no 10")
axis[0,0].plot(t*(1/Fs),left_trials[9,:]+2,c='black',label="left trial no 10")
axis[0,0].set_title("left-hemisphere single-trials")
axis[0,0].set_ylabel("")
axis[0,0].set_xlabel("time (s)")



axis[1,0].plot(t*(1/Fs),np.mean(left_trials,axis=0),c='black',label='original')
axis[1,0].plot(t*(1/Fs),np.mean(filt_left_trials,axis=0),c='blue',label='filtered')
axis[1,0].set_title("left-hemisphere M100 averaged response Left")
axis[1,0].set_ylabel("")
axis[1,0].set_xlabel("time (s)")

axis[0,1].plot(t*(1/Fs),filt_right_trials[0,:],c='blue',label="filtered right trial no 1")
axis[0,1].plot(t*(1/Fs),right_trials[0,:],c='black',label="right trial no 1")
axis[0,1].plot(t*(1/Fs),filt_right_trials[9,:]+2,c='red',label="filtered right trial no 10")
axis[0,1].plot(t*(1/Fs),right_trials[9,:]+2,c='black',label="right trial no 10")
axis[0,1].set_title("right-hemisphere single-trials")
axis[0,1].set_ylabel("")
axis[0,1].set_xlabel("time (s)")



axis[1,1].plot(t*(1/Fs),np.mean(right_trials,axis=0),c='black',label='original')
axis[1,1].plot(t*(1/Fs),np.mean(filt_right_trials,axis=0),c='blue',label='filtered')
axis[1,1].set_title("right-hemisphere M100 averaged response Left")
axis[1,1].set_ylabel("")
axis[1,1].set_xlabel("time (s)")

axis[0,0].legend(loc='best')
axis[1,0].legend(loc='best')
axis[0,1].legend(loc='best')
axis[1,1].legend(loc='best')
plt.show()

figure, axis = plt.subplots(1,2)

axis[0].imshow(filt_left_trials,cmap = plt.cm.get_cmap("jet"))
axis[0].set_title("filtered right-hemisphere single-trials")
axis[0].set_ylabel("")
axis[0].set_xlabel("time (sample no)")

axis[1].imshow(filt_right_trials,cmap = plt.cm.get_cmap("jet"))
axis[1].set_title("filtered right-hemisphere single-trials")
axis[1].set_ylabel("")
axis[1].set_xlabel("time (sample no)")
plt.show()