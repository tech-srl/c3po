import pickle
import os
import sys
import json
from distutils.dir_util import copy_tree
if __name__ == '__main__':
    all_projects = os.path.abspath(sys.argv[1])
    train_projects = os.path.abspath(sys.argv[2])
    split_json = os.path.abspath(sys.argv[3])
    if not os.path.exists(train_projects):
        os.mkdir(train_projects)

    with open(split_json, "r") as f:
    	train_project_names = json.load(f)['train']
    for project_name in train_project_names:
    	src = os.path.join(all_projects, project_name)
    	dst = os.path.join(train_projects, project_name)
    	copy_tree(src, dst)
        