import os
import argparse
import copy
import numpy as np
from datetime import datetime

import subprocess

def prettyPrintDict(d):
    s = ""
    # maxL = 0
    for k in d:
        s += "%35s | %s\n" % (k, repr(d[k]))
    return s

parser = argparse.ArgumentParser()
parser.add_argument('exec', type = str, help="The program file *.py that will be executed.")
parser.add_argument('--runs', type = int, default = 1, help="How many runs for each HP setup.")
# parser.add_argument('--gpus', type = str, default = '0', help="GPUs to be used.")
parser.add_argument('-g', '--greedy', action = 'store_true', help="Use greedy search instead of grid search. Order are determined by ? order - I have no idea.")
parsed, unknown = parser.parse_known_args()

for arg in unknown:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=str)

args = parser.parse_args()
# args.gpus = args.gpus.split(',')

hpgrid = copy.deepcopy(args.__dict__)
hpgrid.pop("exec")
hpgrid.pop("runs")
hpgrid.pop("greedy")
# hpgrid.pop("gpus")
for hp in hpgrid:
    hpgrid[hp] = hpgrid[hp].split(',')

name = "%s" % (args.exec.split('.')[0])
pthprefix = "Results_new/%s/%s/%s/%s/" % ("HPTuner", name, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("[%H-%M-%S]"))
os.makedirs(pthprefix)

# Initialize multiple GPU
# nGPUs = len(args.gpus)
# nonParallel = False

# if nGPUs == 1:
#     nonParallel = True
# else:
#     nGPU_running = 0
#     GPUs_procs = [None for i in range(nGPUs)]
#     output_files = [open(os.path.join(pthprefix, "GPU_%d.log" % i), 'w') for i in range(nGPUs)]
#     print("\033[1;36mCommands for output monitoring:\033[0m")
#     for i in range(nGPUs):
#         print("less +F \"%s\"" % (os.path.join(pthprefix, "GPU_%d.log" % i)))

print("\033[1;31mAll possible HPs: " + repr(hpgrid) + "\033[0m")

# Start HPTune
cnt = 0
total_HPs = 1
if args.greedy:
    total_HPs = 0

finish = False
hpstats = {}
options = {}
keys = []

for key in hpgrid:
    
    hpstats[key] = 0
    options[key] = hpgrid[key][hpstats[key]]
    keys.append(key)
    
    if args.greedy:
        if(len(hpgrid[key]) > 1):
            total_HPs += len(hpgrid[key])
    else:
        total_HPs *= len(hpgrid[key])

results = []
current_HP = []
current_key_id = 0

while(current_key_id < len(keys) and len(hpgrid[keys[current_key_id]]) <= 1):
    current_key_id += 1

while not finish:

    # Update pane title
    cnt += 1
    print("\033]2;HP [%3d/%3d] %s\033\\" % (cnt, total_HPs, args.exec))
    
    # Run expr
    exec_cmd = ["python", args.exec]
    for key in options:
        # exec_cmd = exec_cmd + " --%s %s" % (key, options[key])
        exec_cmd.append("--%s" % key)
        exec_cmd.append("%s" % options[key])
    
    accs = []
    for r in range(args.runs):
        print(("\033[1;32m[%2d / %2d] \033[0m" % (r+1, args.runs)) + "\033[1;33m" + ' '.join(exec_cmd) + "\033[0m")
        
        # Clear existing result file if any
        if os.path.exists("out.temp"):
            os.remove("out.temp")
        
        # Run the experiment
        # os.system(exec_cmd)
        subprocess.run(exec_cmd)

        # Collect results if any
        acc = 0.0
        if os.path.exists("out.temp"):
            with open("out.temp", 'r') as fp:
                try:
                    acc = float(fp.readline().strip())
                except:
                    print("\033[4;31mNewly generated out.temp file detected, but does not contain valid content (a single floating number). \033[0m")
                    acc = 0.0
        accs.append(acc)

        print("\033[4;36mResult: %f \033[0m" % acc)
    
    accs = np.asarray(accs)
    results.append({"mean": accs.mean(), "std": accs.std(), "HP": copy.deepcopy(options)})

    # update HP
    if args.greedy:
        
        # Append current result
        current_HP.append(accs.mean())

        # Choose next
        hpstats[keys[current_key_id]] += 1

        # Check if we have enough results to determine choice of current HP
        if len(current_HP) == len(hpgrid[keys[current_key_id]]):
            
            # Pick #choice with max acc
            choice = 0
            max_acc = 0
            for i in range(len(hpgrid[keys[current_key_id]])):
                if current_HP[i] > max_acc:
                    choice = i
                    max_acc = current_HP[i]
            
            # Set hpstats to the choice
            hpstats[keys[current_key_id]] = choice

            # Notice the user
            print("\033[1;31mChoose %s = %s by greedy search.\033[0m" % (keys[current_key_id], hpgrid[keys[current_key_id]][choice]))

            # Move to next key
            current_key_id += 1
            while(current_key_id < len(keys) and len(hpgrid[keys[current_key_id]]) <= 1):
                current_key_id += 1

            # Clean results
            current_HP = []
        
        # Check if finish
        finish = False
        if current_key_id >= len(keys):
            finish = True
            print("\033[1;36mBest choice:\n%s\033[0m" % (prettyPrintDict({keys[i] : hpgrid[keys[i]][hpstats[keys[i]]] for i in range(len(keys))})))

    else:
        finish = True
        for key in hpgrid:
            hpstats[key] += 1
            hpstats[key] = hpstats[key] % len(hpgrid[key])
            if hpstats[key] != 0:
                finish = False
                break

    # Apply hpstats
    for key in hpgrid:
        options[key] = hpgrid[key][hpstats[key]]

# sort
results.sort(key = lambda x : x["mean"], reverse = True)

# write CSV
with open(os.path.join(pthprefix, "out.csv"), 'w') as fp:
    
    # Header
    fp.write("Result mean, Result std, ")
    for key in hpgrid:
        fp.write("%s, " % key)
    fp.write("\n")

    # Rows
    for result in results:
        fp.write("%f, %f, " % (result['mean'], result['std']))
        for key in hpgrid:
            fp.write("%s, " % result['HP'][key])
        fp.write("\n")