import os
import shutil

def default(value, default_value):
    return default_value if not value else value




class TmpDir(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __enter__(self):
        path = f'{self.root_dir}/tmp'
        os.mkdir(f'{self.root_dir}/tmp')
        return path

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(f'{self.root_dir}/tmp')
