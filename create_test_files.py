from glob import glob
import random
from shutil import copyfile
import os


if __name__ == '__main__':
    cls = ['def_shaft_alignment', 'normal', 'def_baring', 'loose_belt', 'rotating_unbalance']

    cnt = 0
    for c in cls:
        path = f'dataset/vibration/test/{c}/*.csv'

        files = glob(path)
        for f in files[:100]:
            new_f = f.replace('test','test_500')
            new_dir = os.path.dirname(new_f)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            copyfile(f, new_f)
            cnt += 1

    print(cnt)


    cnt = 0
    for c in cls:
        path = f'dataset/current/test/{c}/*.csv'

        files = glob(path)
        for f in files[:100]:
            new_f = f.replace('test','test_500')
            new_dir = os.path.dirname(new_f)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            copyfile(f, new_f)
            cnt += 1

    print(cnt)