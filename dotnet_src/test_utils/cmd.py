
import os
import subprocess

def run(cmd, timeout=15.0):
    '''Run a command, fail with assert when it returns non-zero'''
    subprocess.check_call(cmd, timeout=timeout, shell=True)

def cd_filedir(file):
    os.chdir(os.path.dirname(file))
