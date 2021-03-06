import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import os
import copy
from soundscape_IR.soundscape_viewer import lts_maker
from soundscape_IR.soundscape_viewer.utility import gdrive_handle

class click_processing:  
  def __init__(self, folder = [], dateformat='yymmddHHMMSS', initial=[], year_initial=2000, process_all=True):
    if folder:
      self.collect_folder(folder, dateformat=dateformat, initial=initial, year_initial=year_initial)
    if process_all:
      self.assemble()

  def collect_folder(self, path, dateformat='yymmddHHMMSS', initial=[], year_initial=2000):
    file_list = os.listdir(path)
    self.link = path
    self.dateformat=dateformat
    self.initial=initial
    self.year_initial=year_initial
    self.audioname=np.array([], dtype=np.object)
    for filename in file_list:
        if filename.endswith('.txt'):
            self.audioname = np.append(self.audioname, filename)
    print('Identified ', len(self.audioname), 'files')
    
  def save_csv(self, path='.', filename='All_detections.csv'):
    self.original_detection.to_csv(path+'/'+filename, sep='\t', index=True)

  def assemble(self, start=0, num_file=None):
    self.start = start
    if num_file:
      run_list = range(self.start, self.start+num_file)
    else:
      run_list = range(self.start, len(self.audioname))

    n=0
    lts=lts_maker()
    for file in run_list:
      print('\r', end='')
      temp = self.audioname[file]
      print('Processing file no. '+str(file)+' :'+temp+', in total: '+str(len(self.audioname))+' files', flush=True, end='')

      df = pd.read_table(self.link+'/'+temp,index_col=0) 
      if n==0:
        lts.filename_check(dateformat=self.dateformat, initial=self.initial, year_initial=self.year_initial, filename=temp)
        lts.get_file_time(temp)
        data = df[['Begin Time (s)', 'End Time (s)']]+lts.time_vec
        snr = df['Maximum SNR (dB)']
      else:
        lts.get_file_time(temp)
        data = pd.concat([data, df[['Begin Time (s)', 'End Time (s)']]+lts.time_vec])
        snr = pd.concat([snr, df['Maximum SNR (dB)']])
      n+=1
    self.original_detection=data
    self.original_detection['Maximum SNR (dB)']=snr
    self.original_detection=self.original_detection.sort_values(by=['Begin Time (s)'])

class noise_filter:
  def __init__(self, click, min_snr=1, max_duration=None, min_ici=None, max_ici=0.2, min_pulses=5, max_pulses=None, max_smoothness=0.5,  remove_machine=False):
    self.original_detection=click
    detection_time=np.array(click[['Begin Time (s)','End Time (s)','Maximum SNR (dB)']])
    print('Detected '+str(detection_time.shape[0])+' signals.')

    # Filtering based on SNR
    if min_snr:
      noise_list=np.where(detection_time[:,2]<min_snr)[0]
      if len(noise_list)>0:
        detection_time=np.delete(detection_time, noise_list, axis=0)

    # Filtering based on duration
    if max_duration:
      duration=(detection_time[:,1]-detection_time[:,0])
      noise_list=np.where(duration>max_duration)[0]
      if len(noise_list)>0:
        detection_time=np.delete(detection_time, noise_list, axis=0)
      print('Removing long signals, there are '+str(detection_time.shape[0])+' signals left.')

    # Filtering based on ICI
    # min-ICI
    if min_ici:
      ICI=np.diff(detection_time[:,0])
      noise_list=np.where((ICI<min_ici))[0]+1
      detection_time=np.delete(detection_time, noise_list, axis=0)
    # max-ICI
    ICI=np.append(np.append(max_ici, np.diff(detection_time[:,0])), max_ici)
    noise_list=np.where(((ICI[0:-1]>max_ici)*(ICI[1:]>max_ici))==True)[0]
    if len(noise_list)>0:
      detection_time=np.delete(detection_time, noise_list, axis=0)
    print('Removing isolated signals, there are '+str(detection_time.shape[0])+' signals left.')

    # Filtering based on pulse number
    if min_pulses:
      ICI=np.diff(detection_time[:,0])
      train_begin=np.append(0, np.where(ICI>max_ici)[0]+1)
      train_end=np.append(np.where(ICI>max_ici)[0], detection_time.shape[0]-1)
      noise_train=np.where((train_end-train_begin+1)<min_pulses)[0].astype(int)
      if max_pulses:
        noise_train=np.sort(np.append(noise_train, np.where((train_end-train_begin+1)>max_pulses)[0])).astype(int)
      if len(noise_train)>0:
        detection_time,_=self.train_remove(detection_time, noise_train, train_begin, train_end)
      print('Removing trains with a few pulses, there are '+str(detection_time.shape[0])+' signals left.')

    # Filtering based on ICI smoothness
    if max_smoothness:
      ICI=np.diff(detection_time[:,0])
      ICI[ICI>max_ici]=np.nan
      train_begin=np.append(0, np.where(np.diff(detection_time[:,0])>max_ici)[0]+1)
      train_end=np.append(np.where(np.diff(detection_time[:,0])>max_ici)[0], detection_time.shape[0]-1)
      modulation=np.array([])
      for n in range(len(train_begin)):
        modulation=np.append(modulation, np.nanmean(np.abs(np.diff(detection_time[train_begin[n]:train_end[n]+1,0],n=2)))/np.nanmean(np.diff(detection_time[train_begin[n]:train_end[n]+1,0])))          
      noise_train=np.where(modulation>max_smoothness)[0].astype(int)
      if len(noise_train)>0:
        detection_time, self.noise_time=self.train_remove(detection_time, noise_train, train_begin, train_end)
    print('Removing unsmoothed clicks, there are '+str(detection_time.shape[0])+' clicks left.')

    # Filtering based on ICI repetition
    if remove_machine:
      interval=np.arange(-1*max_ici*1000, max_ici*1000)
      interval_list=np.where(interval>30)[0]
      train_begin=np.append(0, np.where(np.diff(detection_time[:,0])>max_ici)[0]+1)
      train_end=np.append(np.where(np.diff(detection_time[:,0])>max_ici)[0], detection_time.shape[0]-1)
      ici_estimation=np.array([0,0])
      for n in range(len(train_begin)):
        ici_t=np.diff(detection_time[train_begin[n]:train_end[n]+1,0])*1000
        kde=KernelDensity(kernel='gaussian', bandwidth=2.5).fit(ici_t.reshape(-1,1))
        dens=np.exp(kde.score_samples(np.arange(0, max_ici*1000).reshape(-1,1)))
        peak_interval=interval[interval_list[np.argmax(np.correlate(dens, dens, mode='full')[interval_list-1])]]
        peak_ici=np.arange(0, max_ici*1000)[np.argmax(dens)]
        ici_estimation=np.vstack((ici_estimation, np.array([peak_interval, peak_ici])))
      noise_train=np.where(np.divide(np.abs(ici_estimation[1:,0]-ici_estimation[1:,1]),ici_estimation[1:,1])<0.05)[0]
      #noise_train=np.append(noise_train, np.where(np.divide(np.remainder(ici_estimation[1:,1], ici_remove), np.floor(ici_estimation[1:,1]/ici_remove))<=5)[0])
      if len(noise_train)>0:
        detection_time,self.noise_time=self.train_remove(detection_time, noise_train, train_begin, train_end)
      print('Removing machine-associated clicks, there are '+str(detection_time.shape[0])+' clicks left.')

    self.click_analysis(detection_time, max_ici=max_ici)
    self.train_analysis(max_ici=max_ici)

  def click_analysis(self, detection_time, max_ici):
    # Save result
    ICI=np.append(np.diff(detection_time[:,0]), np.nan)
    ICI[ICI>max_ici]=np.nan
    self.detection=detection_time
    self.result=pd.DataFrame()
    self.result['Time']=pd.to_datetime(detection_time[:,0]/24/3600-693962, unit='D',origin=pd.Timestamp('1900-01-01'))
    self.result['Begin Time (MATLAB)']=detection_time[:,0]/24/3600
    self.result['Duration']=detection_time[:,1]-detection_time[:,0]
    self.result['ICI']=ICI
    self.result['SNR']=detection_time[:,2]

  def train_analysis(self, max_ici):
    # Analysis of click trains
    detection_time=self.detection
    ICI=self.result['ICI']
    train_begin=np.append(0, np.where(np.diff(detection_time[:,0])>max_ici)[0]+1)
    train_end=np.append(np.where(np.diff(detection_time[:,0])>max_ici)[0], detection_time.shape[0]-1)
    self.train_result=pd.DataFrame()
    self.train_result['Time']=pd.to_datetime(detection_time[train_begin,0]/24/3600-693962, unit='D',origin=pd.Timestamp('1900-01-01'))
    self.train_result['Begin Time (MATLAB)']=detection_time[train_begin,0]/24/3600
    self.train_result['Duration']=detection_time[train_end,1]-detection_time[train_begin,0]
    self.train_result['Number of clicks']=train_end-train_begin+1
    ici_result=np.array([0,0,0,0,0,0])
    for n in range(len(train_begin)):
      ici_t=np.diff(detection_time[train_begin[n]:train_end[n]+1,0])*1000
      kde=KernelDensity(kernel='gaussian', bandwidth=2.5).fit(ici_t.reshape(-1,1))
      dens=np.exp(kde.score_samples(np.arange(0, max_ici*1000).reshape(-1,1)))
      peak_ici=np.arange(0, max_ici*1000)[np.argmax(dens)]
      diversity=entropy(dens, base=10)
      temp=np.array([np.nanmean(ICI[train_begin[n]:train_end[n]+1]), np.nanmean(np.abs(np.diff(ICI[train_begin[n]:train_end[n]]))), np.nanmin(ICI[train_begin[n]:train_end[n]]), np.nanmax(ICI[train_begin[n]:train_end[n]]), diversity, peak_ici])
      ici_result=np.vstack((ici_result, temp))
    self.train_result['Mean ICI']=ici_result[1:,0]
    self.train_result['Peak ICI']=ici_result[1:,5]
    self.train_result['Minimum ICI']=ici_result[1:,2]
    self.train_result['Maximum ICI']=ici_result[1:,3]
    self.train_result['ICI Smoothness']=np.divide(ici_result[1:,1], ici_result[1:,0])    
    self.train_result['ICI Diversity']=ici_result[1:,4]

  def train_click_check(self, max_ici):
    detection_time=self.detection
    train_begin=np.append(0, np.where(np.diff(detection_time[:,0])>max_ici)[0]+1)
    train_end=np.append(np.where(np.diff(detection_time[:,0])>max_ici)[0], detection_time.shape[0]-1)
    noise_train=np.ones(train_begin.shape)
    for n in range(len(train_begin)):
      if np.round(detection_time[train_begin[n],0]*1000) in np.array(np.round(self.train_result['Begin Time (MATLAB)']*24*3600*1000)):
        noise_train[n]=0
    noise_train=np.where(noise_train==1)[0]
    detection_time, noise_time=self.train_remove(detection_time, noise_train, train_begin, train_end)
    self.click_analysis(detection_time, max_ici=max_ici)
    self.train_analysis(max_ici=max_ici)

  def train_drop(self, col1, col1_range, col2, col2_range, noise_train):
    con1_data=self.train_result[(self.train_result[col1]>=col1_range[0]) & (self.train_result[col1]<col1_range[1])]
    noise_train=np.append(noise_train, con1_data[(con1_data[col2]>=col2_range[0]) & (con1_data[col2]<col2_range[1])].index)
    return noise_train

  def train_search_drop(self, col1, col2, count, nbins=50):
    if len(nbins)==1:
      nbins=[nbins, nbins]
    H, col1_bin, col2_bin=np.histogram2d(self.train_result[col1], self.train_result[col2], bins=(nbins[0],nbins[1]))
    col2_index, col1_index=np.where(H.T>=count)
    noise_train=np.array([])
    for n in range(len(col1_index)):
      noise_train=self.train_drop(col1, [col1_bin[col1_index[n]], col1_bin[col1_index[n]+1]], col2, [col2_bin[col2_index[n]], col2_bin[col2_index[n]+1]], noise_train)
    self.noise_result=self.train_result.loc[noise_train]
    self.train_result=self.train_result.drop(index=noise_train)
    
  def train_remove(self, detection_time, noise_train, train_begin, train_end):
    noise_list=np.array([])
    for n in noise_train:
      noise_list=np.append(noise_list, np.arange(train_begin[n], train_end[n]+1)).astype(int)
    noise_time=detection_time[noise_list,:]
    detection_time=np.delete(detection_time, noise_list, axis=0)
    return detection_time, noise_time
  
  def effort_calculate(self, path, dateformat='yymmddHHMMSS', initial=[], year_initial=2000, recording_length=300):
    self.effort=np.array([], dtype=np.object)
    file_list = os.listdir(path)
    self.recording_length=recording_length
    lts=lts_maker()
    for filename in file_list:
      if filename.endswith('.txt'):
        if len(self.effort)==0:
          lts.filename_check(dateformat=dateformat, initial=initial, year_initial=year_initial, filename=file_list[0])
        lts.get_file_time(filename)
        self.effort=np.append(self.effort, lts.time_vec)       

  def temporal_changes(self, time_resolution=300, begin_date=None, end_date=None, filename='Click_analysis.csv', folder_id=[]):
    from soundscape_IR.soundscape_viewer import data_organize
    import datetime
    self.sheet=data_organize()
    if begin_date:
      yy=int(begin_date[0:4])
      mm=int(begin_date[4:6])
      dd=int(begin_date[6:8])
      date=datetime.datetime(yy,mm,dd)
      begin_time=date.toordinal()+366
      if end_date:
        yy=int(end_date[0:4])
        mm=int(end_date[4:6])
        dd=int(end_date[6:8])
        date=datetime.datetime(yy,mm,dd)
        end_time=date.toordinal()+366
    else:
      begin_time=np.floor(np.min(self.original_detection['Begin Time (s)'])/24/3600)
      end_time=np.ceil(np.max(self.original_detection['Begin Time (s)'])/24/3600)
    
    time_vec=np.arange(begin_time,end_time,time_resolution/24/3600)
    data,_=np.histogram(np.array(self.effort)/24/3600,time_vec)
    self.sheet.time_fill(time_vec[0:-1], data*self.recording_length, 'Recording Time (s)')
    data,_=np.histogram(np.array(self.result['Begin Time (MATLAB)']),time_vec)
    self.sheet.time_fill(time_vec[0:-1], data, 'Number of clicks')
    data2,_=np.histogram(np.array(self.result['Begin Time (MATLAB)'][self.result['ICI']<0.04]),time_vec)
    data2=np.divide(data2, data)
    data2[np.isnan(data2)]=0
    self.sheet.time_fill(time_vec[0:-1], data2, 'Ratio of short-range clicks')
    data,_=np.histogram(np.array(self.train_result['Begin Time (MATLAB)']),time_vec)
    self.sheet.time_fill(time_vec[0:-1], data, 'Number of trains')
    data2=0*time_vec[0:-1]
    for n in range(len(data)):
      data2[n]=np.nanmean(self.train_result['Number of clicks'].iloc[np.where((np.array(self.train_result['Begin Time (MATLAB)'])>=time_vec[n])*(np.array(self.train_result['Begin Time (MATLAB)'])<time_vec[n+1]))[0]])
    data2[np.isnan(data2)]=0
    self.sheet.time_fill(time_vec[0:-1], data2, 'Mean number of clicks/train')  

  def save(self, filename='Analysis', folder_id=[]):
    self.sheet.save_csv(filename+'_temporal_changes.csv', folder_id=folder_id)
    self.result.to_csv(filename+'_clicks.csv', sep=',')
    self.train_result.to_csv(filename+'_trains.csv', sep=',')
    if folder_id:
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(filename+'_clicks.csv')
      Gdrive.upload(filename+'_trains.csv')

  def plot_ici(self, range_y=[5,200]):
    fig = px.scatter(x=self.result['Time'], y=self.result['ICI']*1000, color=self.result['SNR'], log_y=True, range_y=range_y)
    fig.update_xaxes(title_text='Time (sec)')
    fig.update_yaxes(title_text='Inter-click interval (ms)')
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_traces(marker=dict(size=5))
    fig.show()

  def plot_click_summary(self, fig_width=15, fig_height=5):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(fig_width, fig_height))
    ax=(self.result['Duration']*1000).hist(bins=np.arange(0, 15, 1), grid=False, ax=axes[0])
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Number of clicks")
    ax=(self.result['ICI']*1000).hist(bins=np.arange(0, 150, 10), grid=False, ax=axes[1])
    ax.set_xlabel("Inter-click interval (ms)")
    ax=(self.result['SNR']).hist(bins=np.arange(1, 10, 0.5), grid=False, ax=axes[2])
    _=ax.set_xlabel("Signal-to-noise ratio (dB)")

  def plot_train_summary(self, fig_width=15, fig_height=5):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(fig_width, fig_height))
    ax=(self.train_result['Duration']).hist(bins=np.arange(0, 5, 0.2), grid=False, ax=axes[0])
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Number of trains")
    ax=(self.train_result['Mean ICI']*1000).hist(bins=np.arange(0, 150, 10), grid=False, ax=axes[1])
    ax.set_xlabel("Mean inter-click interval (ms)")
    ax=(self.train_result['Number of clicks']).hist(bins=np.arange(0, 60, 3), grid=False, ax=axes[2])
    _=ax.set_xlabel("Number of clicks")

  def plot_temporal_changes(self, min_number_trains=10, fig_width=20, fig_height=8, cmap_name='jet'):
    temp_data=copy.deepcopy(self.sheet)
    fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(fig_width, fig_height))
    temp_data.final_result[temp_data.final_result[:,4]<min_number_trains,3]=0
    temp_data.final_result[temp_data.final_result[:,4]<min_number_trains,5]=0
    for n in range(4):
      ax[n], im=self.plot_diurnal(temp_data, ax[n], col=n+2, fig_width=fig_width/4, fig_height=fig_height, nan_value=-1, cmap_name=cmap_name)
      ax[n].xaxis_date()
      ax[n].set_title(temp_data.result_header[n+2])
      plt.setp(ax[n].get_xticklabels(), rotation=45, ha='right')
      cbar = fig.colorbar(im, ax=ax[n])
      if n==0:
        ax[n].set_ylabel('Hour')

  def plot_diurnal(self, sheet, ax, col=1, vmin=None, vmax=None, fig_width=16, fig_height=8, nan_value=0, cmap_name='jet'):
    hr_boundary=[np.min(24*(sheet.final_result[:,0]-np.floor(sheet.final_result[:,0]))), np.max(24*(sheet.final_result[:,0]-np.floor(sheet.final_result[:,0])))]
    input_data=sheet.final_result[:,col]
    input_data[input_data==nan_value]=np.nan

    time_vec=sheet.final_result[:,0]
    
    hr=np.unique(24*(time_vec-np.floor(time_vec)))
    no_sample=len(time_vec)-np.remainder(len(time_vec), len(hr))
    day=np.unique(np.floor(time_vec[0:no_sample]))
    python_dt=day+693960-366

    plot_matrix=input_data.reshape((len(day), len(hr))).T
    im=ax.imshow(plot_matrix, vmin=vmin, vmax=vmax, origin='lower',  aspect='auto', cmap=plt.get_cmap(cmap_name),
                    extent=[python_dt[0], python_dt[-1], np.min(hr_boundary), np.max(hr_boundary)], interpolation='none')
    return ax, im

  
  def plot_histogram2d(self, col1='Peak ICI', col2='ICI Diversity', nbins=50):
    if len(nbins)==1:
      nbins=[nbins, nbins]
    H, col1_bin, col2_bin=np.histogram2d(self.train_result[col1], self.train_result[col2], bins=(nbins[0],nbins[1]))
    fig = go.Figure(data=go.Heatmap(z=H.T, x=col1_bin, y=col2_bin))
    fig.show()

class tonal_processing:  
  def __init__(self, folder = [], dateformat='yymmddHHMMSS', initial=[], year_initial=2000, process_all=True):
    if folder:
      self.collect_folder(folder, dateformat=dateformat, initial=initial, year_initial=year_initial)
    if process_all:
      self.assemble()

  def collect_folder(self, path, dateformat='yymmddHHMMSS', initial=[], year_initial=2000):
    file_list = os.listdir(path)
    self.link = path
    self.dateformat=dateformat
    self.initial=initial
    self.year_initial=year_initial
    self.audioname=np.array([], dtype=np.object)
    for filename in file_list:
        if filename.endswith('.txt'):
            self.audioname = np.append(self.audioname, filename)
    print('Identified ', len(self.audioname), 'files')
    
  def save_csv(self, path='.', filename='All_detections.csv'):
    self.original_detection.to_csv(path+'/'+filename, sep='\t', index=True)

  def assemble(self, start=0, num_file=None):
    self.start = start
    if num_file:
      run_list = range(self.start, self.start+num_file)
    else:
      run_list = range(self.start, len(self.audioname))

    n=0
    lts=lts_maker()
    for file in run_list:
      print('\r', end='')
      temp = self.audioname[file]
      print('Processing file no. '+str(file)+' :'+temp+', in total: '+str(len(self.audioname))+' files', flush=True, end='')
      df = pd.read_table(self.link+'/'+temp) 
      if n==0:
        lts.filename_check(dateformat=self.dateformat, initial=self.initial, year_initial=self.year_initial, filename=temp)
        lts.get_file_time(temp)
        data = df['Time']+lts.time_vec
        frequency = df['Frequency']
        snr = df['Strength']
      else:
        lts.get_file_time(temp)
        data = pd.concat([data, df['Time']+lts.time_vec])
        frequency = pd.concat([frequency, df['Frequency']])
        snr = pd.concat([snr, df['Strength']])
      n+=1
    
    self.original_detection=pd.DataFrame()
    self.original_detection['Time']=pd.to_datetime(data/24/3600-693962, unit='D',origin=pd.Timestamp('1900-01-01'))
    self.original_detection['Frequency']=frequency
    self.original_detection['Strength']=snr
    self.original_detection['Date_num']=data/24/3600
    self.original_detection=self.original_detection.sort_values(by=['Time'])

    data=np.sort(data)
    frequency=np.array(self.original_detection['Frequency'])
    data_list=self.tonal_noise_filter(data, frequency)
    self.result=self.original_detection.iloc[data_list]

  def tonal_noise_filter(self, tonal, frequency, scanning_window=0.5, scanning_frequency=3000, occupancy_th=0.2, harmonic_remove=True):
    scanning_window=scanning_window*1000
    data=np.sort(np.array(tonal))
    frequency=np.array(frequency)
    data=np.round((data-np.min(data))*1000)
    time_resolution=np.min(np.diff(np.unique(data)))
    full_count=np.ceil(scanning_window/time_resolution)-1
    presence=np.zeros(data.size)
    for n in np.arange(data.size):
      temp=np.abs(data-data[n])
      data_list=np.where(temp<=scanning_window/2)[0]
      freq_list=np.multiply(frequency[data_list]<=frequency[n]+scanning_frequency/2, frequency[data_list]>=frequency[n]-scanning_frequency/2)
      data_list=data_list[freq_list==1]
      if ((np.unique(data[data_list]).size-1)/full_count)>occupancy_th:
        presence[data_list]=1 
    data_list=np.where(presence==1)[0]

    if harmonic_remove:
      harmonic_list=np.ones(data_list.size)
      for n in np.arange(data_list.size):
        temp_list=np.where(np.abs(data[data_list]-data[data_list[n]])<15)[0]
        temp=frequency[data_list[n]]/frequency[data_list[temp_list]]
        temp=temp[temp>1.5]
        if np.sum(np.abs(temp-np.round(temp))<0.05)>1:
          harmonic_list[n]=0
      data_list=data_list[np.where(harmonic_list==1)[0]]
    return data_list

  def plot_frequency(self):
    fig = px.scatter(x=self.result['Time'], y=self.result['Frequency'], color=self.result['Strength'])
    fig.update_xaxes(title_text='Time (sec)')
    fig.update_yaxes(title_text='Frequency (Hz)')
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_traces(marker=dict(size=3))
    fig.show()
    
  def group_screening(self, group_interval=600, min_duration=60, label=None):
    data=self.result
    data_time=np.array(data['Date_num'])*24*60*60
    fragment_onset=np.hstack((np.array([0]), np.where(np.diff(data_time)>group_interval)[0]+1))
    fragment_offset=np.hstack((np.where(np.diff(data_time)>group_interval)[0], np.array([len(data_time)-1])))
    duration=data_time[fragment_offset]-data_time[fragment_onset]
    fragment_onset=fragment_onset[np.where(duration>min_duration)[0]]
    fragment_offset=fragment_offset[np.where(duration>min_duration)[0]]
    group_onset=data_time[fragment_onset]/24/60/60
    group_offset=data_time[fragment_offset]/24/60/60
        
    df=pd.DataFrame()
    df['Onset']=pd.to_datetime(group_onset-693962, unit='D',origin=pd.Timestamp('1900-01-01'))
    df['Offset']=pd.to_datetime(group_offset-693962, unit='D',origin=pd.Timestamp('1900-01-01'))
    
    if len(group_onset)>0:
        for n in np.arange(len(group_onset)):
          temp=np.arange(fragment_onset[n], fragment_offset[n]+1)
          freq_data=np.array(data['Frequency'])[temp]
          freq_time=np.array(data['Date_num'])[temp]
          freq_timebins=len(np.unique(np.round(freq_time*24*60*60)))
          if n==0:
            freq_duration=freq_timebins
            freq_percentile=np.percentile(freq_data, [5, 50, 95])[None,:]
          else:
            freq_percentile=np.vstack((np.percentile(freq_data, [5, 50, 95]),freq_percentile))
            freq_duration=np.append(freq_duration, freq_timebins)
    
    df.insert(2, 'Detected duration (s)', freq_duration)
    df2=pd.DataFrame(data=freq_percentile, columns=['Q5', 'Q50', 'Q95'])
    df=pd.concat([df,df2],axis=1)
    if label:
      label=np.matlib.repmat([label],n+1,1)
      df.insert(0, 'Label', label)
    self.df_group=df

  def save(self, filename='Analysis'):
    self.df_group.to_csv(filename+'_groups.csv', sep=',')
