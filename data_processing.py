import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        data = data.append(df[['Begin Time (s)', 'End Time (s)']]+lts.time_vec)
        snr = snr.append(df['Maximum SNR (dB)'])
      n+=1
    self.original_detection=data
    self.original_detection['Maximum SNR (dB)']=snr
    self.original_detection=self.original_detection.sort_values(by=['Begin Time (s)'])

class noise_filter:
  def __init__(self, click, min_snr=1, max_duration=None, min_ici=None, max_ici=0.15, min_pulses=4, max_pulses=1000):
    self.original_detection=click
    detection_time=np.array(click[['Begin Time (s)','End Time (s)','Maximum SNR (dB)']])
    print('Detected '+str(detection_time.shape[0])+' clicks.')
    # Filtering based on SNR
    if min_snr:
      noise_list=np.where(detection_time[:,2]<min_snr)[0]
      detection_time=np.delete(detection_time, noise_list, axis=0)

    # Filtering based on duration
    if max_duration:
      duration=(detection_time[:,1]-detection_time[:,0])
      noise_list=np.where(duration>max_duration)[0]
      detection_time=np.delete(detection_time, noise_list, axis=0)
      print('Removing long clicks, there are '+str(detection_time.shape[0])+' clicks left.')

    # Filtering based on ICI and number of pulses
    # min-ICI
    if min_ici:
      ICI=np.diff(detection_time[:,0])
      noise_list=np.where((ICI<min_ici))[0]+1
      detection_time=np.delete(detection_time, noise_list, axis=0)
    # max-ICI
    ICI=np.append(np.append(max_ici, np.diff(detection_time[:,0])), max_ici)
    noise_list=np.where(((ICI[0:-1]>max_ici)*(ICI[1:]>max_ici))==True)[0]
    detection_time=np.delete(detection_time, noise_list, axis=0)
    # pulse number
    max_ici=max_ici*2
    train_begin=np.append(0, np.where(np.diff(detection_time[:,0])>max_ici)[0]+1)
    train_end=np.append(np.where(np.diff(detection_time[:,0])>max_ici)[0], detection_time.shape[0]-1)
    num_pulses=train_end-train_begin+1
    noise_train=np.where(num_pulses<min_pulses)[0].astype(int)
    if max_pulses:
      noise_train=np.sort(np.append(noise_train, np.where(num_pulses>max_pulses)[0])).astype(int)
    noise_list=np.array([])
    for n in noise_train:
      noise_list=np.append(noise_list, np.arange(train_begin[n], train_end[n]+1)).astype(int)
    detection_time=np.delete(detection_time, noise_list, axis=0)
    print('Removing isolated clicks, there are '+str(detection_time.shape[0])+' clicks left.')

    # Save result
    duration=(detection_time[:,1]-detection_time[:,0])
    ICI=np.append(np.diff(detection_time[:,0]), np.nan)
    ICI[ICI>max_ici]=np.nan
    click_time=pd.to_datetime(detection_time[:,0]/24/3600-693962, unit='D',origin=pd.Timestamp('1900-01-01'))
    snr=detection_time[:,2]
    self.detection=detection_time
    self.result=pd.DataFrame()
    self.result['Time']=click_time
    self.result['Begin Time (MATLAB)']=detection_time[:,0]/24/3600
    self.result['Duration']=duration
    self.result['ICI']=ICI
    self.result['SNR']=snr

    # Analysis of click trains
    train_begin=np.append(0, np.where(np.diff(detection_time[:,0])>max_ici)[0]+1)
    train_end=np.append(np.where(np.diff(detection_time[:,0])>max_ici)[0], detection_time.shape[0]-1)
    self.train_result=pd.DataFrame()
    self.train_result['Time']=click_time[train_begin]
    self.train_result['Begin Time (MATLAB)']=detection_time[train_begin,0]/24/3600
    self.train_result['Duration']=detection_time[train_end,1]-detection_time[train_begin,0]
    self.train_result['Number of clicks']=train_end-train_begin+1
    ici_result=np.array([0,0,0,0])
    for n in range(len(train_begin)):
      temp=np.array([np.nanmean(ICI[train_begin[n]:train_end[n]+1]), np.nanstd(ICI[train_begin[n]:train_end[n]+1]), np.nanmin(ICI[train_begin[n]:train_end[n]+1]), np.nanmax(ICI[train_begin[n]:train_end[n]+1])])
      ici_result=np.vstack((ici_result, temp))
    self.train_result['Mean ICI']=ici_result[1:,0]
    self.train_result['SD ICI']=ici_result[1:,1]
    self.train_result['Minimum ICI']=ici_result[1:,2]
    self.train_result['Maximum ICI']=ici_result[1:,3]
    self.effort=np.array([], dtype=np.object)
  
  def effort_calculate(self, path, dateformat='yymmddHHMMSS', initial=[], year_initial=2000, recording_length=300):
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
    data,_=np.histogram(np.array(self.train_result['Begin Time (MATLAB)']),time_vec)
    self.sheet.time_fill(time_vec[0:-1], data, 'Number of trains')
    data=0*time_vec[0:-1]
    data2=0*time_vec[0:-1]
    for n in range(len(data)):
      data[n]=1/np.nanmean(self.train_result['Mean ICI'].iloc[np.where((np.array(self.train_result['Begin Time (MATLAB)'])>=time_vec[n])*(np.array(self.train_result['Begin Time (MATLAB)'])<time_vec[n+1]))[0]])
      data2[n]=np.nanmean(self.train_result['Number of clicks'].iloc[np.where((np.array(self.train_result['Begin Time (MATLAB)'])>=time_vec[n])*(np.array(self.train_result['Begin Time (MATLAB)'])<time_vec[n+1]))[0]])
    data[np.isnan(data)]=0
    data2[np.isnan(data2)]=0
    self.sheet.time_fill(time_vec[0:-1], data, 'Reciprocal of mean ICI')
    self.sheet.time_fill(time_vec[0:-1], data2, 'Mean number of clicks')  

  def save(self, filename='Analysis', folder_id=[]):
    self.sheet.save_csv(filename+'_temporal_changes.csv', folder_id=folder_id)
    self.result.to_csv(filename+'_clicks', sep=',')
    self.train_result.to_csv(filename+'_trains', sep=',')
    if folder_id:
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(filename+'_clicks')
      Gdrive.upload(filename+'_trains')

  def plot_ici(self):
    fig = px.scatter(x=self.result['Time'], y=self.result['ICI']*1000, color=self.result['SNR'], log_y=True, range_y=[5,200])
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

  def plot_temporal_changes(self, min_number_trains=10, fig_width=20, fig_height=8):
    temp_data=copy.deepcopy(self.sheet)
    fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(fig_width, fig_height))
    temp_data.final_result[temp_data.final_result[:,2]<min_number_trains,3]=0
    temp_data.final_result[temp_data.final_result[:,2]<min_number_trains,4]=0
    for n in range(4):
      ax[n], im=self.plot_diurnal(temp_data, ax[n], col=n+2, fig_width=fig_width/4, fig_height=fig_height, nan_value=-1)
      ax[n].xaxis_date()
      ax[n].set_title(temp_data.result_header[n+2])
      plt.setp(ax[n].get_xticklabels(), rotation=45, ha='right')
      cbar = fig.colorbar(im, ax=ax[n])
      if n==0:
        ax[n].set_ylabel('Hour')

  def plot_diurnal(self, sheet, ax, col=1, vmin=None, vmax=None, fig_width=16, fig_height=8, nan_value=0):
    hr_boundary=[np.min(24*(sheet.final_result[:,0]-np.floor(sheet.final_result[:,0]))), np.max(24*(sheet.final_result[:,0]-np.floor(sheet.final_result[:,0])))]
    input_data=sheet.final_result[:,col]
    input_data[input_data==nan_value]=np.nan

    time_vec=sheet.final_result[:,0]
    
    hr=np.unique(24*(time_vec-np.floor(time_vec)))
    no_sample=len(time_vec)-np.remainder(len(time_vec), len(hr))
    day=np.unique(np.floor(time_vec[0:no_sample]))
    python_dt=day+693960-366

    plot_matrix=input_data.reshape((len(day), len(hr))).T
    im=ax.imshow(plot_matrix, vmin=vmin, vmax=vmax, origin='lower',  aspect='auto', cmap=cm.jet,
                    extent=[python_dt[0], python_dt[-1], np.min(hr_boundary), np.max(hr_boundary)], interpolation='none')
    return ax, im
