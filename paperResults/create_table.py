import re
import json
import os
import sys

dataset = os.path.abspath(sys.argv[1])
model = sys.argv[2]

partition ='test'

beam_size = 1
before_path = "{}/{}.src".format(dataset, partition)
after_path = "{}/{}.dst".format(dataset, partition)
projects_pah = "{}/{}.projects".format(dataset, partition)
results_path = "{}/{}".format(dataset, model)
pred_path = os.path.join(results_path, "pred_{}.txt".format(partition))

if not os.path.exists(results_path):
    os.mkdir(results_path)

with open(before_path, "r") as f:
    before = f.readlines()

with open(after_path, "r") as f:
    after_line = f.readlines()

with open(pred_path, "r") as f:
    pred = f.readlines()

with open(projects_pah, "r") as f:
    projects = f.readlines()

project_idx = list()
cur_project = projects[0]
i = 1
for p in projects:
    if p != cur_project:
        cur_project = p
        i = 1
    project_idx.append(i)
    i += 1

new_pred = list()
temp = list()
for i, p in enumerate(pred):
    temp.append(p)
    if (i+1) % beam_size == 0:
        new_pred.append(temp)
        temp = list()

pattern = re.compile('(.*)<@>(.*?)<\/@>(.*)')
before_line = list(map(lambda x: pattern.match(x).group(2), before))
before_context = list(map(lambda x: pattern.match(x).group(1), before))
after_context = list(map(lambda x: pattern.match(x).group(3), before))


acc_1 = 0
acc_beam = 0

json_obj = dict()
changes_pattern = re.compile(r"<%>(.*?)</%>")
template = "-------------------------------------------------------------{}_{}_{}\n\nbefore_ctx:  {}\n\nsrc: 	      {}\n\nafter_ctx: 	{}\n\nctx_changes: {}\n\ntgt_changes: {}\n\ntarget: 	 {}\npred:   	 {}\n"
with open(os.path.join(results_path, "table_{}.txt".format(partition)), "w") as f:
    for i in range(len(before)):
        project = projects[i].rstrip()
        if project not in json_obj.keys():
            json_obj[project] = list()
        acc_str = ""
        if after_line[i] in new_pred[i]:
            acc_str = "@" + str(beam_size)
            acc_beam += 1
        if after_line[i] == new_pred[i][0]:
            acc_str = "@1"
            acc_1 += 1
        ctx_changes = map(lambda s: s.strip(), changes_pattern.findall(before_context[i] + " " + after_context[i]))
        line = template.format(
                project,
                project_idx[i],
                acc_str,
                before_context[i],
                before_line[i],
                after_context[i],
                "\n             ".join(ctx_changes),
                "",
                after_line[i],
                "	    	 ".join(new_pred[i])
            )
        f.write(line)
        json_obj[project].append({
                'before_ctx': before_context[i],
                'before_line': before_line[i],
                'after_ctx': after_context[i],
                'after_line': after_line[i].rstrip(),
                'predictions': list(map(lambda x: x.strip(), new_pred[i]))
            })

    f.write("Acc@1: {}\n".format(acc_1 / len(before)))
    f.write("Acc@{}: {}\n".format(beam_size, acc_beam / len(before)))

print("Acc: {:0.3f}".format(acc_1 / len(before)))
with open(os.path.join(results_path, "res_{}.json".format(partition)), "w") as f:
    json.dump(json_obj, f)
