import os
import sys
import git
from git import Repo

repo_file = os.path.abspath(sys.argv[1])
repo_folder = sys.argv[2]

if not os.path.exists(repo_folder):
	os.mkdir(repo_folder)
os.chdir(repo_folder)

repo_urls = [line.strip().split('\t')[1] for line in open(repo_file)]
repo_names = [line.strip().split('\t')[0] for line in open(repo_file)]

class Progress(git.RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        print('update(%s, %s, %s, %s)'%(op_code, cur_count, max_count, message))

for i, repo_url in enumerate(repo_urls):
    Repo.clone_from(repo_url, "./{}".format(repo_names[i]), progress=Progress())
