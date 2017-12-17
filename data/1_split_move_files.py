"""
After decide the original data, we run this script to split and move all the files into
the appropriate train/test/validation folders.

Should creat train/test/validation folders ahead.
[mkdir train && mkdir test && mkdir validation && mkdir sequences && mkdir checkpoints && mkdir logs]
Should only run this file once!
"""

import os
import glob
import subprocess as sp
import random


def split_move_files():
    vid_folder = './'
    vid_files = glob.glob(vid_folder + '*.avi')

    for video_path in vid_files:
        parts = video_path.split('/')
        filename = parts[1]
        filename_no_ext = filename.split('.')[0]
        
        parts = filename_no_ext.split('_')
        classname = parts[0]
        num_data = [0, 0, 0]  # the number of data (train, val, test)

        for num in range(0,200):
            #split files
            output_name = filename_no_ext + '_%03d.avi' %num
            output_path = os.path.join('.', output_name)
            start = num * 3
            end = start + 3
            split_vid_from_path(video_path, output_path, str(start), str(end))
            
            # move files
            group, num_data = mark_train_test_validation(num, num_data)
                
            # Check if this class exists.
            if not os.path.exists(group + '/' + classname):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(group + '/' + classname)

            # Check if we have already moved this file, or at least that it
            # exists to move.
            if not os.path.exists(output_name):
                print("Can't find %s to move. Skipping." % (output_name))
                continue

            # Move it.
            dest = group + '/' + classname + '/' + output_name
            print(num, ". Moving %s to %s" % (output_name, dest))
            os.rename(output_name, dest)
        print ('Results of spliting: ', num_data)
            
    print('Done!')

def split_vid_from_path(video_file_path, output_file_path, start_time, end_time):
    pipe = sp.Popen(["ffmpeg","-v", "quiet", "-y", "-i", video_file_path, "-vcodec", "copy", "-acodec", "copy",
                 "-ss", start_time, "-to", end_time, "-sn", output_file_path ])
    pipe.wait()
    return True

'''
def mark_train_test_validation(n):
    if n<140:
        return 'train'
    elif n<170:
        return 'validation'
    else:
        return 'test'
'''

def mark_train_test_validation(n, num_data): # randomly select

    for i in range (0,50):
        ran = random.random()
        if ran * (200 - sum(num_data)) < 140 - num_data[0]:
            if num_data[0] < 140:
                num_data[0] = num_data[0]+1
                return 'train', num_data
        elif ran * (200 - sum(num_data)) < 170 - (num_data[1] + num_data[0]):
            if num_data[1] < 30:
                num_data[1] = num_data[1]+1
                return 'validation', num_data
        else:
            if num_data[2] < 30:
                num_data[2] = num_data[2]+1
                return 'test', num_data

    if num_data[0] < 140:
        num_data[0] = num_data[0] + 1
        return 'train', num_data
    elif num_data[2] < 30:
        num_data[2] = num_data[2] + 1
        return 'test', num_data
    elif num_data[1] < 30:
        num_data[1] = num_data[1] + 1
        return 'validation', num_data


    '''
    ran = random.randint(1,30)
    if ran % 3 == 0: # train
        if num_data[0] < 140:
            num_data[0] = num_data[0]+1
            return 'train', num_data
        elif num_data[1] < 30:
            num_data[1] = num_data[1] +1
            return 'validation', num_data
        else:
            num_data[2] = num_data[2] +1
            return 'test', num_data

    elif ran % 3 == 1: # val
        if num_data[1] < 30:
            num_data[1] = num_data[1]+1
            return 'validation', num_data
        elif num_data[0] < 140:
            num_data[0] = num_data[0] + 1
            return 'train', num_data
        else:
            num_data[2] = num_data[2] +1
            return 'test', num_data

    else: # test
        if num_data[2] < 30:
            num_data[2] = num_data[2]+1
            return 'test', num_data
        elif num_data[0] < 140:
            num_data[0] = num_data[0] + 1
            return 'train', num_data
        else:
            num_data[1] = num_data[1] +1
            return 'validation', num_data
    '''

def main():

    split_move_files()

if __name__== '__main__':
    main()