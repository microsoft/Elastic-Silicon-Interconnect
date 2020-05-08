
import os

def run(cmd):
    '''Run a command, fail with assert when it returns non-zero'''
    return_code = os.system(cmd)
    assert return_code == 0

def cd_filedir(file):
    os.chdir(os.path.dirname(file))
