import os
import numpy as np
import glob
from suite2p import run_s2p

def get_earliest_file_time(folder, use_mtime=True):
    """
    Return the earliest timestamp (creation or modification) among files inside the folder.
    If the folder is empty or contains no files, return a large value to sort it last.
    """
    timestamps = []
    for root, _, files in os.walk(folder):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                time = os.path.getmtime(fpath) if use_mtime else os.path.getctime(fpath)
                timestamps.append(time)
            except Exception:
                pass  # skip problematic files
    return min(timestamps) if timestamps else float('inf')


if __name__ == '__main__':

    animal = 'EC_GCaMP6sawake_13' #need to run ctrl 12
    day = None
    stim = None # 'spon' or 'grat', if None, will run both

    for roi_detection in ['functional']:

        print(f'Running suite2p for {animal} {day} {stim} {roi_detection}')

        # load ops file
        if roi_detection == 'functional':
            ops = np.load(fr'C:\Users\erica\suite2p_thy1gcamp6s.npy', allow_pickle=True).item()
        elif roi_detection == 'anatomical':
            ops = np.load(fr'C:\Users\erica\suite2p_anatomical_cellpose.npy', allow_pickle=True).item()
            ops['cellprob_threshold'] = -3

        # modify ops file
        ops['fast_disk'] = 'C:\\'
        ops['input_format'] = 'sbx'
        ops['save_folder'] = f'suite2p {roi_detection}'
        root_dir = fr'E:\vision_restored\data\{animal}'

        files = []

        if day: # if a day is specified, only process files in that day
            for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, day)):
                if os.path.basename(dirpath) == 'experiments':
                    if stim:
                        if stim in dirpath:
                            files.append(dirpath)
                    else:
                        files.append(dirpath)

        else: # if a day isnt specified, process all files
            for dirpath, dirnames, filenames in os.walk(root_dir):
                if os.path.basename(dirpath) == 'experiments':
                    if stim:
                        if stim in dirpath:
                            files.append(dirpath)
                    else:
                        files.append(dirpath)

        # sort folders according to creation time
        files.sort(key=lambda f: get_earliest_file_time(f, use_mtime=True))
        print(files)
        for file in files:
            db = {'data_path':[file]}
            opsEND = run_s2p(ops=ops, db=db)