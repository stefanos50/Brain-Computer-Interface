import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import numpy as np
import scipy
from sklearn import metrics
import time
import pywt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import svm
import HelperMethods
from sklearn.utils import shuffle

def SVM_Model(X_train,y_train,X_test,y_test):
    clf = svm.SVC(C=5, coef0=0, degree=1, gamma=0.001, kernel='poly', tol=0.01)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    pred_test = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, pred_test)
    pred_train = clf.predict(X_train)
    train_accuracy = metrics.accuracy_score(y_train, pred_train)
    plot_classifier_result(X_train,y_train,pred_train,X_test,y_test,pred_test)

    return train_accuracy,test_accuracy,end-start

def KNN_Model(X_train,y_train,X_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=10,metric='cosine')
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    pred_test = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, pred_test)
    pred_train = clf.predict(X_train)
    train_accuracy = metrics.accuracy_score(y_train, pred_train)
    plot_classifier_result(X_train,y_train,pred_train,X_test,y_test,pred_test)

    return train_accuracy,test_accuracy,end-start

def LDA_Model(X_train,y_train,X_test,y_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    pred_test = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, pred_test)
    pred_train = clf.predict(X_train)
    train_accuracy = metrics.accuracy_score(y_train, pred_train)
    plot_classifier_result(X_train,y_train,pred_train,X_test,y_test,pred_test)

    return train_accuracy,test_accuracy,end-start

def plot_classifier_result(X,y,pred,X_test,y_test,pred_test):
    figure, axis = plt.subplots(1, 2)
    # plot lines
    c = ['firebrick','royalblue']
    for i in range(len(X)):
        if y[i] == pred[i]:
            axis[0].plot(X[i][0],X[i][-1],c=c[y[i]],marker='.')
        else:
            axis[0].plot(X[i][0],X[i][-1], c=c[y[i]],marker='x')
    axis[0].set_title("Train Set")
    axis[0].set_xlabel("1st Dimension")
    axis[0].set_ylabel("Last Dimension")

    # plot the legend
    for i in range(len(X_test)):
        if y_test[i] == pred_test[i]:
            axis[1].plot(X_test[i][0],X_test[i][-1],c=c[y_test[i]],marker='.')
        else:
            axis[1].plot(X_test[i][0],X_test[i][-1], c=c[y_test[i]],marker='x')
    axis[1].set_title("Test Set")
    axis[1].set_xlabel("1st Dimension")
    axis[1].set_ylabel("Last Dimension")

    handles, labels = axis[1].get_legend_handles_labels()
    patch = mpatches.Patch(color='firebrick', label='Stim Class')
    handles.append(patch)
    patch = mpatches.Patch(color='royalblue', label='Control Class')
    handles.append(patch)
    plt.legend(handles=handles, loc='upper center')
    plt.show()

def CNN(X_train,y_train,X_test,y_test):
    X_train = X_train.reshape((X_train.shape[0],7,400,1))
    X_test = X_test.reshape((X_test.shape[0],7,400,1))
    print(X_train.shape)
    import tensorflow as tf
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout,AveragePooling2D,BatchNormalization
    from keras.preprocessing.image import ImageDataGenerator
    start = time.time()
    model = Sequential([
        Conv2D(filters=40, kernel_size=(3, 3), activation=tf.nn.relu,padding='valid',),
        BatchNormalization(),
        Dropout(0.25),
        Conv2D(filters=40, kernel_size=(3, 3), activation=tf.nn.relu,padding='valid'),
        BatchNormalization(),
        Dropout(0.25),
        #AveragePooling2D(pool_size=(2, 2),strides=2),

        Flatten(),
        Dense(100, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.AdamW(lr=0.0001,weight_decay=0.0000001)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    end = time.time()
    history = model.fit(tf.convert_to_tensor(X_train),tf.convert_to_tensor(y_train),validation_split=0.10, epochs=20,batch_size=32,shuffle=True,steps_per_epoch=1)
    loss_test, accuracy_test = model.evaluate(X_test,y_test,batch_size=32)
    loss_train, accuracy_train = model.evaluate(X_train,y_train,batch_size=32)
    print(accuracy_train)
    print(accuracy_test)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    return accuracy_train,accuracy_test,end-start

Fs=625

mat = scipy.io.loadmat(
    'K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\stimulation\\stim_times.mat')
stim_times = np.array(mat['stim_times'])

mat = scipy.io.loadmat(
    'K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\stimulation\\subj3_left.mat')
subj3_left = np.array(mat['subj3_left'])

mat = scipy.io.loadmat(
    'K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\stimulation\\subj3_right.mat')
subj3_right = np.array(mat['subj3_right'])

mat = scipy.io.loadmat(
    'K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\control\\subj3_control_right.mat')
subj3_control_right = np.array(mat['subj3_control_right'])

mat = scipy.io.loadmat(
    'K:\\DataLearningSignalAnalytics\\AUDITORY_MEG_RESPONSES_worked_examples\\control\\subj3_control_left.mat')
subj3_control_left = np.array(mat['subj3_control_left'])
results = []

print("Raw data shape: "+str(subj3_left.shape))
left_trials_stim, t = HelperMethods.signal_to_trials(subj3_left, stim_times, 100, 300)
right_trials_stim, t = HelperMethods.signal_to_trials(subj3_right, stim_times, 100, 300)
left_trials_control, t = HelperMethods.signal_to_trials(subj3_control_left, stim_times, 100, 300)
right_trials_control, t = HelperMethods.signal_to_trials(subj3_control_right, stim_times, 100, 300)
print("Data shape after segmentation "+str(left_trials_stim.shape))

print('Stimulation left shape :'+str(left_trials_stim.shape))
print('Stimulation right shape :'+str(right_trials_stim.shape))
print('Control left shape :'+str(left_trials_control.shape))
print('Control right shape :'+str(right_trials_control.shape))

Wn = np.array([3 / (Fs / 2), 20 / (Fs / 2)])
b, a = signal.butter(5, Wn, btype='pass')

left_trials_stim = signal.filtfilt(b, a, left_trials_stim)
right_trials_stim = signal.filtfilt(b, a, right_trials_stim)

left_trials_control = signal.filtfilt(b, a, left_trials_control)
right_trials_control = signal.filtfilt(b, a, right_trials_control)



figure, axis = plt.subplots(2, 2)
# plot lines
for i in range(len(left_trials_stim)):
    axis[0,0].plot(t, left_trials_stim[i])
axis[0,0].plot(t, np.mean(left_trials_stim,axis=0), linewidth = '5',c="black",label = "averaged response")
axis[0,0].set_title("Stimulus Left")
axis[0,0].set_xlabel("sample no")
axis[0,0].set_ylabel("a.u. (volts)")

for i in range(len(right_trials_stim)):
    axis[0,1].plot(t, right_trials_stim[i])
axis[0,1].plot(t, np.mean(right_trials_stim,axis=0), linewidth = '5',c="black",label = "averaged response")
axis[0,1].set_title("Stimulus Right")
axis[0,1].set_xlabel("sample no")
axis[0,1].set_ylabel("a.u. (volts)")

for i in range(len(left_trials_control)):
    axis[1,0].plot(t, left_trials_control[i])
axis[1,0].plot(t, np.mean(left_trials_control,axis=0), linewidth = '5',c="black",label = "averaged response")
axis[1,0].set_title("Control Left")
axis[1,0].set_xlabel("sample no")
axis[1,0].set_ylabel("a.u. (volts)")

for i in range(len(right_trials_control)):
    axis[1,1].plot(t, right_trials_control[i])
axis[1,1].plot(t, np.mean(right_trials_control,axis=0), linewidth = '5',c="black",label = "averaged response")
axis[1,1].set_title("Control Right")
axis[1,1].set_xlabel("sample no")
axis[1,1].set_ylabel("a.u. (volts)")

plt.show()


import numpy as np
import pywt
dt = 1/Fs
fs = 1 / dt
frequencies = np.array([2,3,4,5,6,7,10]) / fs # normalize
scales = pywt.frequency2scale('cmor1.5-1.0', frequencies)
print(scales)

coeff1 , freqs1 = pywt.cwt(left_trials_stim[0],scales,'morl',sampling_period=1/Fs)
coeff2 , freqs2 = pywt.cwt(left_trials_control[0],scales,'morl',sampling_period=1/Fs)
print(coeff1.shape)
plt.Figure()
plt.subplot(221)
plt.imshow(coeff1,cmap='coolwarm',aspect='auto')
plt.title("left trial stim trial 1 scales")
plt.ylabel("scale")
plt.subplot(222)
plt.imshow(coeff2,cmap='coolwarm',aspect='auto')
plt.title("left trial control trial 1 scales")
plt.ylabel("scale")


coeff1_averaged , freqs1 = pywt.cwt(np.mean(np.concatenate((left_trials_stim,right_trials_stim),axis=0),axis=0),scales,'morl',sampling_period=1/Fs)
coeff2_averaged , freqs2 = pywt.cwt(np.mean(np.concatenate((left_trials_control,right_trials_control),axis=0),axis=0),scales,'morl',sampling_period=1/Fs)
plt.Figure()
plt.subplot(223)
plt.imshow(coeff1_averaged,cmap='coolwarm',aspect='auto')
plt.ylabel("scale")
plt.title("stim averaged scales")
plt.subplot(224)
plt.imshow(coeff2_averaged,cmap='coolwarm',aspect='auto')
plt.title("control averaged scales")
plt.ylabel("scale")
plt.show()
#-----------------------------
fig=plt.figure()
ax1 = fig.add_subplot(2,2,1,projection='3d')
Y = np.arange(1,coeff1.shape[0]+1,1)
X= np.arange(1,coeff1.shape[1]+1,1)

X,Y = np.meshgrid(X,Y)
ax1.plot_surface(X,Y,coeff1,cmap=cm.coolwarm,linewidth=0)

ax1.set_xlabel("Time",fontsize=20)
ax1.set_ylabel("Scale",fontsize=20)
ax1.set_zlabel("Amplitude",fontsize=20)
ax1.set_title("left trial stim trial 1")

ax2 = fig.add_subplot(2,2,2,projection='3d')

Y = np.arange(1,coeff2.shape[0]+1,1)
X= np.arange(1,coeff2.shape[1]+1,1)

X,Y = np.meshgrid(X,Y)

ax2.plot_surface(X,Y,coeff2,cmap=cm.coolwarm,linewidth=0)

ax2.set_xlabel("Time",fontsize=20)
ax2.set_ylabel("Scale",fontsize=20)
ax2.set_zlabel("Amplitude",fontsize=20)
ax2.set_title("right trial stim trial 1")

ax3 = fig.add_subplot(2,2,3,projection='3d')
Y = np.arange(1,coeff1_averaged.shape[0]+1,1)
X= np.arange(1,coeff1_averaged.shape[1]+1,1)

X,Y = np.meshgrid(X,Y)
ax3.plot_surface(X,Y,coeff1_averaged,cmap=cm.coolwarm,linewidth=0)

ax3.set_xlabel("Time",fontsize=20)
ax3.set_ylabel("Scale",fontsize=20)
ax3.set_zlabel("Amplitude",fontsize=20)
ax3.set_title("stim averaged scales")

ax4 = fig.add_subplot(2,2,4,projection='3d')

Y = np.arange(1,coeff2_averaged.shape[0]+1,1)
X= np.arange(1,coeff2_averaged.shape[1]+1,1)

X,Y = np.meshgrid(X,Y)

ax4.plot_surface(X,Y,coeff2_averaged,cmap=cm.coolwarm,linewidth=0)

ax4.set_xlabel("Time",fontsize=20)
ax4.set_ylabel("Scale",fontsize=20)
ax4.set_zlabel("Amplitude",fontsize=20)
ax4.set_title("control averaged scales")
plt.show()

labels_stim = [0 for x in range(len(left_trials_stim)+len(right_trials_stim))]
labels_control = [1 for x in range(len(left_trials_control)+len(right_trials_control))]
labels_stim = np.array(labels_stim)
labels_control = np.array(labels_control)
labels = np.concatenate((labels_stim,labels_control),axis=0)
print("Final labels shape: "+str(labels.shape))

data = np.concatenate((left_trials_stim,right_trials_stim,left_trials_control,right_trials_control),axis=0)
print("Final data shape: "+str(data.shape))


features = []
for i in range(len(data)):
    coeff,freqs = pywt.cwt(data[i],scales,'morl',method='conv',sampling_period=1/Fs)
    features.append(coeff.flatten())
data = np.array(features)
data = StandardScaler().fit_transform(data)

# CROSS VALIDATION
N = 5
kf = KFold(n_splits=N)
test_accuracy = []
train_accuracy = []
time_ls = []

data, labels = shuffle(data, labels, random_state=0)
data_array = np.array(data)
labels_array = np.array(labels)

if False:
    param_grid = {'n_neighbors': [4, 5, 6, 7, 8,9,10],
                  'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                  'leaf_size': [1, 2,3],
                  'p': [1, 2, 3],
                  'weights': ['uniform', 'distance']}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3, cv=5, n_jobs=-1)
    grid.fit(data, labels)
    print(grid.best_params_)
    print(grid.best_estimator_)

if False:
    param_grid = {'C': [0.1, 1, 5, 10],
                       'gamma': [ 2, 1, 0.1, 0.001],
                       'tol': [0.001, 0.01, 0.1, 0.5],
                       'coef0': [0, 0.1, 0.5, 1],
                       'degree': [1, 2, 3],
                       'kernel': ['poly','linear','sigmoid','rbf']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3,cv=5,n_jobs=-1)
    grid.fit(data, labels)
    print(grid.best_params_)
    print(grid.best_estimator_)


for train_index, test_index in kf.split(data):
    train_data_x, test_data_x = data_array[train_index], data_array[test_index]
    train_data_y, test_data_y = labels_array[train_index], labels_array[test_index]
    #trainac,testac,ttime = SVM_Model(train_data_x,train_data_y,test_data_x,test_data_y)
    trainac, testac, ttime = KNN_Model(train_data_x, train_data_y, test_data_x, test_data_y)
    #trainac, testac, ttime = LDA_Model(train_data_x, train_data_y, test_data_x, test_data_y)
    #trainac, testac, ttime = CNN(train_data_x, train_data_y, test_data_x, test_data_y)
    test_accuracy.append(testac)
    train_accuracy.append(trainac)
    time_ls.append(ttime)

print("Final average accuracy of train set: "+str(round(sum(train_accuracy)/len(train_accuracy) * 100,2))+"%")
print("Final average accuracy of test set: "+str(round(sum(test_accuracy)/len(test_accuracy) * 100,2))+"%")
print("Final average train time: "+str(sum(time_ls)/len(time_ls)))