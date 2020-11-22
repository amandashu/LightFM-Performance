import os
import shutil

def remove_results():
    locations = ['report','result_experiments','results']
    for l in locations:
        if os.path.exists(l):
            shutil.rmtree(l)
