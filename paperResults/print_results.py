import os
import sys

TEST_SIZE = 5934
LASER_TEST_SIZE = 5278

log_file = os.path.abspath(sys.argv[1])
is_laser = sys.argv[2]

with open(log_file, "r") as f:
	line = f.readlines()[-1];
raw_acc = line.split("Acc:")[-1].strip()
if is_laser == "true":
	acc = 1-((((1-float(raw_acc))*LASER_TEST_SIZE)+(TEST_SIZE-LASER_TEST_SIZE))/TEST_SIZE)
else:
	acc = float(raw_acc.split('\t')[0].strip())
print("Acc: {:0.3f}".format(acc))
