import os

import numpy as np
import matplotlib.pyplot as plt
from ScanImageTiffReader import ScanImageTiffReader
from scipy.signal import find_peaks
import tifffile as tiff
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')


# Log continuous.
# ###############

path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Mauro_Pulin\Data_all\Behavior\MP105'

log1 = np.fromfile(os.path.join(path, 'MP105_20260505_122338', 'log_continuous.bin'))
log2 = np.fromfile(os.path.join(path, 'MP105_20260505_130102', 'log_continuous.bin'))

# Log1 end & Log2 start
log1_end = log1[-5000 * 20 * 6:]
log2_start = log2[:5000 * 20 * 6]
log1_ts = np.arange(0, np.round(len(log1) / 6)) / 5000
log2_ts = np.arange(0, np.round(len(log2) / 6)) / 5000

# Plot
fig, axes = plt.subplots(6, 2, figsize=(15, 8))
for i in range(6):
    axes[i, 0].plot(log1_ts[-5000 * 20:], log1_end[i::6])
    axes[i, 1].plot(log2_ts[:5000 * 20], log2_start[i::6])
fig.tight_layout()

# First bhv session end
bhv_df_1 = pd.read_csv(os.path.join(path, 'MP105_20260505_122338', 'results.csv'))
bhv_df_2 = pd.read_csv(os.path.join(path, 'MP105_20260505_130102', 'results.csv'))

# Check timestamps
print('\nSession 1: ')
print(f'last trial : {bhv_df_1.trial_time.values[-1]}, perf : {bhv_df_1.perf.values[-1]}')
print('\nSession 2: ')
print(f'last trial : {bhv_df_2.trial_time.values[0]}, perf : {bhv_df_1.perf.values[0]}')

# Stitching (Matlab crashes, nothing to remove at the end of part 1)
log_corrected = np.concatenate((log1, log2))
save_path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Mauro_Pulin\From_robin\MP105\corrected'
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, 'log_continuous.bin'), mode='wb') as fid:
    log_corrected.tofile(fid)

# Fix trial-time in second behavior table
part1_ending_time = np.round(len(log1) / 6) / 5000
n_trials_part1 = max(bhv_df_1.trial_number)
bhv_df_2['trial_time'] = bhv_df_2['trial_time'] + part1_ending_time
bhv_df_2['trial_number'] = bhv_df_2['trial_number'] + n_trials_part1

full_bhv = pd.concat([bhv_df_1, bhv_df_2])
full_bhv = full_bhv.reset_index(drop=True)
full_bhv.to_csv(os.path.join(save_path, 'results.csv'))


fig.savefig(os.path.join(save_path, 'fig.png'))


# ---------------------------------------------------------------------------------------------------------------------
# Cut last ttl up from first session due to stopping, fuse and save.
log = np.concatenate([log1[:-12000], log2])

save_path = os.path.join(path, 'corrected')
with open(os.path.join(save_path, 'log_continuous.bin'), mode='wb') as fid:
    log.tofile(fid)


path = r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\log_continuous.bin'
log_bad = np.fromfile(path)
log_bad = log_bad[::6]
log_bad = np.abs(log_bad)

path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\data\AR180\Training\AR180_20241214_194639\log_continuous.bin"
log = np.fromfile(path)
log = log[::6]
log = np.abs(log)


path = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\data\AR176\Training\AR176_20241215_160714\log_continuous.bin"
log = np.fromfile(path)


log2 = log[1::6]

plt.plot(log2[-5000*60*20:-5000*60*10])
log2_cor = np.copy(log2)
log2_cor[-5000*60*20+860000:-5000*60*20+910000] = 0.65
plt.plot(log2_cor[-5000*60*20:-5000*60*10])

log[1::6] = log2_cor
log_check = log[1::6]
plt.plot(log_check[-5000*60*20:-5000*60*10:6])

save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR176\\Training\\AR176_20241215_160714_cor\\log_continuous.bin'
with open(save_path, mode='wb') as fid:
    log.tofile(fid)


91440
157900



# Merging movies.
# ###############

# movies1 = ["D:\\AR\\AR129 24-03-01 13-28-58.avi", 'D:\\AR\\AR129 24-03-03 15-25-39.avi']
# movies2 = ["D:\\AR\\AR129 24-03-01 14-28-27.avi", 'D:\\AR\\AR129 24-03-03 16-10-25.avi']
# save_paths = ['D:\\AR\\AR129 24-03-01 13-28-58_merged.avi',
#               'D:\\AR\\AR129 24-03-03 15-25-39_merged.avi']

movie1 = "D:\\AR\\AR129 24-03-03 15-25-39.avi"
movie2 = "D:\\AR\\AR129 24-03-03 16-10-25.avi"
save_path = 'D:\\AR\\AR129 24-03-03 15-25-39_merged.avi'

videofiles = [movie1, movie2]

video_index = 0
cap = cv2.VideoCapture(videofiles[0])

fourcc = cv2.VideoWriter_fourcc(*'Y800')
out = cv2.VideoWriter(save_path, fourcc, 100.0, (640, 480))
count_frame = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # print(count_frame, end='\r')
    # count_frame += 1
    if frame is None:
        print ("end of video " + str(video_index) + " .. next one now")
        video_index += 1
        if video_index >= len(videofiles):
            break
        cap = cv2.VideoCapture(videofiles[ video_index ])
        ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print ("end.")


# Solving mess with AR141.
# ########################

""" Solve the mess of pressing start instead of resume i.e. imaging continues while video filming and logging stops.
The strategy is to find a tiume stamp in the log file which will be the end of the session -
Take the end of the last logged trial - and delete the excess of imaging frame by
rewritting the tiff; similarly for the avi file.
 """


# read the log and determine final time stamp.

log_1 = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\training\\AR141_20240520_104531\\log_continuous.bin"
log_1 = np.fromfile(log_1)
ttl_events = find_peaks((log_1[2::6]>2)[1:].astype(np.float64) - (log_1[2::6]>2)[:-1].astype(np.float64), distance=100, prominence=1)[0]

cut = ttl_events[-1] + 6 * 5000  # 6 sec after the last trial start.

# plt.plot(log_1[3::6][23959394-5000*10:23959394+5000*60])
# plt.plot(log_1[0::6][23959394-5000*10:23959394+5000*60])
# plt.plot(log_1[1::6][23959394-5000*10:23959394+5000*60])
# plt.plot(log_1[2::6][23959394-5000*10:23959394+5000*60])

log_1_corrected = np.copy(log_1)
for i in range(6):
    log_1_corrected[i::6][cut:] = 0

ttl_events_cor = find_peaks((log_1_corrected[2::6]>2)[1:].astype(np.float64) - (log_1_corrected[2::6]>2)[:-1].astype(np.float64), distance=100, prominence=1)[0]
galvo_events_cor = find_peaks(log_1_corrected[1::6], distance=100, prominence=1)[0]
cam_events_cor = find_peaks(log_1_corrected[3::6], distance=10, prominence=1)[0]

# Checking.
plt.plot(log_1_corrected[3::6])
plt.scatter(cam_events_cor, log_1_corrected[3::6][cam_events_cor])
plt.plot(log_1_corrected[0::6])
plt.plot(log_1_corrected[1::6])
plt.scatter(galvo_events_cor, log_1_corrected[1::6][galvo_events_cor])
plt.plot(log_1_corrected[2::6])
plt.scatter(ttl_events_cor, log_1_corrected[2::6][ttl_events_cor])

n_frames_imaging = galvo_events_cor.size
n_frames_filming = cam_events_cor.size

save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\training\\AR141_20240520_104531\\log_continuous_1_cor.bin'
with open(save_path, mode='wb') as fid:
    log_1_corrected.tofile(fid)


# Count frames in CI movie.

session_1_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\AR141_20240520'
tiff_list = os.listdir(session_1_path)

nframes = 0 
for tiff in tiff_list:
    tiff = os.path.join(session_1_path, tiff)
    nframes += ScanImageTiffReader(tiff).shape()[0]

session_2_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\AR141_20240520_2\\AR141_20240520_00004.tif'
nframes_2 = ScanImageTiffReader(session_2_path).shape()[0]


# Count frames in avi.

movie_1 = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32.avi"
video_capture = cv2.VideoCapture(movie_1)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


# Results:
# n imaging frames in the first session 141495 the second one has 15315.
# n imaging frames to keep: 139504 (in the first session)
141495 - 139504
# n filming frames in avi 430336
# n frames to keep 425376

# Rewrite these files

movie1 = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32.avi"
save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32_cor.avi'

video_index = 0
cap = cv2.VideoCapture(movie1)

fourcc = cv2.VideoWriter_fourcc(*'Y800')
out = cv2.VideoWriter(save_path, fourcc, 100.0, (640, 480))
count_frame = 0
while(cap.isOpened()):
    if count_frame > 430336:
        break
    else:
        print(f'frame {count_frame}/430336', end='\r')
        ret, frame = cap.read()
        out.write(frame)
        count_frame += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
out.release()
cv2.destroyAllWindows()
print ("Fin.")

# Count frames of corrected movie to check.
movie_1 = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\filming\\AR141 24-05-20 10-44-32_cor.avi'
video_capture = cv2.VideoCapture(movie_1)
video_length_cor = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# cut ci tiff

read_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\imaging\\AR141_20240520_1\\AR141_20240520_00003.tif'
tiff = ScanImageTiffReader(tiff).data
tiff.shape
tiff = tiff[:-(141495-139504)]
save_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard\\need_fix\\AR141\\AR141_20240520\\imaging\\AR141_20240520_1\\AR141_20240520_00003_cor.tif'
tiff.imsave(save_path, tiff)

# Checking n imaging frames for AR163

path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/data/AR163/Training/AR163_20241125_153447/log_continuous.bin'
log = np.fromfile(path)


galvo_events = find_peaks(log[1::6], distance=100, prominence=1)[0]
print(galvo_events.size)
print(galvo_events.size/30/3600)

tiff_path = r"//sv-nas1.rcp.epfl.ch/Petersen-Lab/data/AR163/Recording/Imaging/AR163_20241125_153447/AR163_20241125_153447.tif"
nframes = ScanImageTiffReader(tiff_path).shape()[0]
