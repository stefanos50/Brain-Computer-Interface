import networkx as nx
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import mne
import scipy
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as plt
import pandas as pd

mat = scipy.io.loadmat('G:\\Simple_CorrelationNetworkExample_2023\\EEG_data.mat')

topoplot_data = np.array(mat['topoplot_data'])
data = np.array(mat['data'])
Fs = 256

#-----------------------------MAIN PARAMS------------------------------------------
band_names = ['Delta (0.5-4)Hz','Theta (4-8)Hz', 'Alpha (8-13)Hz', 'Beta (13-30)Hz' , 'Gamma (30-100)Hz']
bands = [np.array([0.5/(Fs/2) ,4/(Fs/2)])  ,  np.array([4/(Fs/2) ,8/(Fs/2)])  ,  np.array([8/(Fs/2) ,13/(Fs/2)])  ,  np.array([13/(Fs/2) ,30/(Fs/2)]),  np.array([30/(Fs/2) ,100/(Fs/2)])]
band_id_plot = 2
threshold = 0.7
single_sensors_plot = [16,10]
#-----------------------------------------------------------------------

sensor_list= np.array([0 ,2, 10, 4, 12, 6, 14, 8,1, 3, 11, 5, 13, 7, 15, 9,16,17,18])
right_sensors =  np.array([0 ,2, 10, 4, 12, 6, 14, 8])
left_sensors = np.array([1, 3, 11, 5, 13, 7, 15, 9])
middle_sensors = np.array([16,17,18])
sensor_dict = {}

for i in range(len(sensor_list)):
    sensor_dict[i] = str(sensor_list[i]+1)


def split_chunks(data,t):
    res = []
    temp = []
    for i in range(len(data)):
        temp.append(data[i])
        if(len(temp) > t*Fs):
            res.append(temp)
            temp = []
    if len(temp)>0:
        res.append(temp)
    return res



def plot_data(res,title):
    fig1, axes1 = plt.subplots(len(res), 2, figsize=(10, 20))
    fig1.suptitle(title, fontsize=16)

    for i in range(len(res)):
        for j in range(2):
            if j == 0:
                splited = split_chunks(data[9][0:100 * Fs], 10)
                cl = 0
                for chunk in splited:
                    axes1[i, j].plot(np.array(chunk) + cl)
                    cl += 150
                axes1[i, j].title.set_text("Raw data "+str(band_names[i]))
            elif j == 1:
                splited = split_chunks(res[i][0][9][0:100 * Fs], 10)
                cl = 0
                for chunk in splited:
                    axes1[i, j].plot(np.array(chunk) + cl)
                    cl += 150
                axes1[i, j].title.set_text("Filtered data "+str(band_names[i]))
    fig1.tight_layout()
    plt.show()

def plot_corrcoef(res,title):
    fig1, axes1 = plt.subplots(len(res), 3, figsize=(20, 20))
    fig1.suptitle(title, fontsize=16)
    for i in range(len(res)):
        for j in range(3):
            if j == 0:
                axes1[i, j].imshow(res[i][1], cmap="jet")
                axes1[i, j].title.set_text("R(xi,xj) "+str(band_names[i]))
            elif j == 1:
                axes1[i, j].imshow(res[i][2], cmap="jet")
                axes1[i, j].title.set_text("abs(R(xi,xj)) "+str(band_names[i]))
            elif j == 2:
                axes1[i, j].imshow(res[i][3], cmap="jet")
                axes1[i, j].title.set_text("Ordered abs(R(xi,xj)) "+str(band_names[i]))
    fig1.tight_layout()
    plt.show()

def plot_corrcoef_head(res,band_id,title):
    fig1, axes1 = plt.subplots(5, 4, figsize=(20, 20))
    fig1.suptitle(title, fontsize=16)
    current_channel = 0
    for i in range(5):
        for j in range(4):
            if current_channel <=18:
                im, _ = mne.viz.plot_topomap(res[band_id][1][sensor_list[:, np.newaxis], sensor_list][current_channel],
                                             pos=topoplot_data[sensor_list, :] / 13,
                                             names=sensor_list + 1, cmap='jet', axes=axes1[i, j], show=False)
                axes1[i, j].title.set_text(str(band_names[band_id])+" Channel "+str(sensor_list[current_channel]+1))
                current_channel += 1

                fig1.colorbar(im, ax=axes1[i, j])
            else:
                axes1[i, j].set_axis_off()
    fig1.tight_layout()
    plt.show()

def plot_graphs(res,title):
    fig1, axes1 = plt.subplots(len(res), 5, figsize=(10, 10))
    fig1.suptitle(title, fontsize=16)
    pos =  topoplot_data[sensor_list, :]
    for i in range(len(res)):
        for j in range(5):
            if j == 0:
                G = nx.from_numpy_array(res[i][4])
                nx.draw(G, pos,ax=axes1[i, j])
                axes1[i, j].title.set_text("Strongly Connected " + str(band_names[i]))
            elif j == 1:
                G = nx.from_numpy_array(res[i][5])
                nx.draw(G, pos,ax=axes1[i, j])
                axes1[i, j].title.set_text("Correlated " + str(band_names[i]))
            elif j == 2:
                G = nx.from_numpy_array(res[i][6])
                nx.draw(G, pos,ax=axes1[i, j])
                axes1[i, j].title.set_text("Anti-Correlated " + str(band_names[i]))
            elif j == 3:
                im,_ = mne.viz.plot_topomap(res[i][1][sensor_list[:, np.newaxis], sensor_list][single_sensors_plot[0]], pos=topoplot_data[sensor_list, :] / 13,
                                     names=sensor_list + 1, cmap='jet', axes=axes1[i, j],show=False)
                axes1[i, j].title.set_text(str(band_names[i])+" Channel "+str(sensor_list[single_sensors_plot[0]]+1))
                fig1.colorbar(im, ax=axes1[i, j])
            elif j==4:
                im,_ = mne.viz.plot_topomap(res[i][1][sensor_list[:, np.newaxis], sensor_list][single_sensors_plot[1]], pos=topoplot_data[sensor_list, :] / 13,
                                     names=sensor_list + 1, cmap='jet', axes=axes1[i, j],show=False)
                axes1[i, j].title.set_text(str(band_names[i])+" Channel "+str(sensor_list[single_sensors_plot[1]]+1))
                fig1.colorbar(im, ax=axes1[i, j])
            if j < 3:
                nx.draw_networkx_labels(G, pos, sensor_dict, font_size=10,
                            font_color='black', font_family='sans-serif',ax=axes1[i, j])
    fig1.tight_layout()
    plt.show()

def plot_res(res,title):
    fig, ax = plt.subplots(1, 2)
    data = []
    for i in range(len(res)):
        data.append(res[i][7])

    column_labels = ["RC", "LC", "AI"]

    # creating a 2-dimensional dataframe out of the given data
    df = pd.DataFrame(data, columns=column_labels)

    ax[0].axis('tight')  # turns off the axis lines and labels
    ax[0].axis('off')  # changes x and y axis limits such that all data is shown
    ax[0].title.set_text(title)
    # plotting data
    table = ax[0].table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=band_names,
                     rowColours=["yellow"] * 5,
                     colColours=["red"] * 5,
                     loc="center")
    table.set_fontsize(14)
    table.scale(1, 2)

    ax[1].bar(band_names, np.array(data)[:,2], color='maroon',
        width=0.4)
    plt.xlabel("Band Type")
    plt.ylabel("Assymetry Index")
    plt.title("Results Difference")
    plt.show()

def run(band):
    results = []
    b, a = signal.butter(3, band, btype='pass')
    filt_data = signal.filtfilt(b, a, data)

    results.append(filt_data)
    corrcoef_matrix = np.corrcoef(filt_data)

    #save correlation coefficient without order
    results.append(np.round(corrcoef_matrix,4))
    results.append(abs(np.round(corrcoef_matrix,4)))

    #save correlation coefficient with order
    corrcoef_matrix_ordered = corrcoef_matrix[sensor_list[:, np.newaxis], sensor_list]
    RR = np.round(abs(corrcoef_matrix_ordered),4)

    # save correlation coefficient witH order
    results.append(RR)

    adj_matrix_strongly_connected = RR > threshold
    adj_matrix_strongly_connected = adj_matrix_strongly_connected.astype(int)
    np.fill_diagonal(adj_matrix_strongly_connected, 0)

    adj_matrix_correlated = corrcoef_matrix_ordered > threshold
    adj_matrix_correlated = adj_matrix_correlated.astype(int)
    np.fill_diagonal(adj_matrix_correlated, 0)

    adj_matrix_anti_correlated = corrcoef_matrix_ordered < -threshold
    adj_matrix_anti_correlated = adj_matrix_anti_correlated.astype(int)
    np.fill_diagonal(adj_matrix_anti_correlated, 0)

    results.append(adj_matrix_strongly_connected)
    results.append(adj_matrix_correlated)
    results.append(adj_matrix_anti_correlated)

    llist = []
    rlist = []
    for i in range(19):
        if i <=7:
            llist.append(1)
            rlist.append(0)
        else:
            llist.append(0)
            if i <= 15:
                rlist.append(1)
            else:
                rlist.append(0)
    llist = np.array(llist)
    rlist = np.array(rlist)

    RC = np.matmul(np.matmul(rlist,RR),np.transpose(rlist))
    LC = np.matmul(np.matmul(llist,RR),np.transpose(llist))
    AI = (RC-LC) / ((RC+LC)/2)
    results.append([round(RC,4),round(LC,4),round(AI,4)])

    return results





bands_results = []
for i in range(len(bands)):
    res = run(bands[i])
    bands_results.append(res)

colors = [(1, 1, 1), (1, 1, 1)]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

fig, ax = mne.viz.plot_topomap(sensor_list+1, pos=topoplot_data/13,size=10,res=64,names=sensor_list+1,contours=0,cmap= LinearSegmentedColormap.from_list('Custom', colors, N=256),show=False,axes=axs[0])
axs[0].title.set_text("Default sensors topography")
mne.viz.plot_topomap(sensor_list+1, pos=topoplot_data[sensor_list, :]/13,size=10,res=64,names=sensor_list+1,contours=0,cmap= LinearSegmentedColormap.from_list('Custom', colors, N=256),show=False,axes=axs[1])
axs[1].title.set_text("Sensors topography after reordering")
plt.show()
plot_data(bands_results,'Raw and filtered data')
plot_corrcoef(bands_results,'Correlation coefficient plots')
for band_idx in range(len(bands)):
    plot_corrcoef_head(bands_results,band_idx,'Correlation coefficient plots')
#plot_corrcoef_head(bands_results,band_id_plot,'Correlation coefficient plots')
plot_graphs(bands_results,'Connected graphs plots')
plot_res(bands_results,'Final Results Table')