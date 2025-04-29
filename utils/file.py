import os
import json

def check_make_dir(path, link=False):
    # Check if a diretory exists
    if not os.path.exists(path):
        os.makedirs(path)
        
    # create link
    if link:
        dst = os.path.join(os.getcwd(), path.split('/')[-1])
        create_link(path, dst, True)
        

def create_link(src, dst, target_is_dir=False):
    # Check if a link exists
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_dir)

def get_generated_files(generated_files_path):
    if os.path.exists(generated_files_path):
        with open(generated_files_path, 'r') as f:
            return json.load(f)
    return {}