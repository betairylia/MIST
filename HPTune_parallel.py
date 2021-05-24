import os
import argparse
import copy
import numpy as np
from datetime import datetime

import subprocess
from subprocess import Popen
import select
import time
import sys

def prettyPrintDict(d):
    s = ""
    # maxL = 0
    for k in d:
        s += "%35s | %s\n" % (k, repr(d[k]))
    return s

parser = argparse.ArgumentParser()
parser.add_argument('exec', type = str, help="The program file *.py that will be executed.")
parser.add_argument('--runs', type = int, default = 1, help="How many runs for each HP setup.")
parser.add_argument('--gpus', type = str, default = '0', help="GPUs to be used.")
parser.add_argument('-g', '--greedy', action = 'store_true', help="Use greedy search instead of grid search. Order are determined by ? order - I have no idea.")
parsed, unknown = parser.parse_known_args()

for arg in unknown:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=str)

args = parser.parse_args()
args.gpus = args.gpus.split(',')

assert not args.greedy or "TODO."

hpgrid = copy.deepcopy(args.__dict__)
hpgrid.pop("exec")
hpgrid.pop("runs")
hpgrid.pop("greedy")
hpgrid.pop("gpus")
for hp in hpgrid:
    hpgrid[hp] = hpgrid[hp].split(',')

name = "%s" % (args.exec.split('.')[0])
pthprefix = "Results_new/%s/%s/%s/%s/" % ("HPTuner", name, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("[%H-%M-%S]"))
os.makedirs(pthprefix)

# Initialize multiple GPU
nGPUs = len(args.gpus)

nGPU_running = 0
GPUs_procs = [None for i in range(nGPUs)]
output_files = [open(os.path.join(pthprefix, "GPU_%s.log" % args.gpus[i]), 'w') for i in range(nGPUs)]
run_info = [None for i in range(nGPUs)]
selects = [None for i in range(nGPUs)]
print("\033[1;36mCommands for output monitoring:\033[0m")
for i in range(nGPUs):
    print("less +F -R \"%s\"" % (os.path.join(pthprefix, "GPU_%s.log" % args.gpus[i])))

print("\033[1;31mAll possible HPs: " + repr(hpgrid) + "\033[0m")

results = []
current_HP = []
current_key_id = 0

shouldSchedule = True

def GetCommand():

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
        
        # accs = []
        result_idx = len(results)
        results.append({"mean": 0, "std": 0, "raw_results": [], "HP": copy.deepcopy(options)})
        for r in range(args.runs):
            
            # # Clear existing result file if any
            # if os.path.exists("out.temp"):
            #     os.remove("out.temp")
            
            # Run the experiment
            # os.system(exec_cmd)
            # subprocess.run(exec_cmd)
            yield exec_cmd, (result_idx, r)

            # Collect results if any
            # acc = 0.0
            # if os.path.exists("out.temp"):
            #     with open("out.temp", 'r') as fp:
            #         try:
            #             acc = float(fp.readline().strip())
            #         except:
            #             print("\033[4;31mNewly generated out.temp file detected, but does not contain valid content (a single floating number). \033[0m")
            #             acc = 0.0
            # accs.append(acc)

            # print("\033[4;36mResult: %f \033[0m" % acc)
        
        # accs = np.asarray(accs)

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

# Run
cmds = iter(GetCommand())
while True:
    
    # Populate tasks
    while shouldSchedule and nGPU_running < nGPUs:

        # wait some seconds; otherwise the output folder will be the same ...
        time.sleep(2)
    
        # Get cmd
        cmd, info = next(cmds, (None, None))

        # We finished all tasks
        if cmd is None:
            shouldSchedule = False
        # Still tasks left
        else:
            # Find GPU
            gpu_id = -1
            for i in range(nGPUs):
                if GPUs_procs[i] is None:
                    gpu_id = i
                    break
            assert gpu_id >= 0
            del i

            title_str = ("\033[1;32m[%2d / %2d] \033[0m" % (info[1]+1, args.runs)) + "\033[1;33m" + ' '.join(cmd) + "\033[0m (GPU %d)" % int(args.gpus[gpu_id])
            print(title_str)
            output_files[gpu_id].write("\n\n%s\n*=*=*=*=*=*\n" % title_str)
            output_files[gpu_id].flush()

            GPUs_procs[gpu_id] = Popen(cmd, stdout = subprocess.PIPE, env = dict(os.environ, CUDA_VISIBLE_DEVICES="%s" % args.gpus[int(gpu_id)]), shell = False)
            run_info[gpu_id] = (cmd, info)

            # https://stackoverflow.com/questions/36476841/python-how-to-read-stdout-of-subprocess-in-a-nonblocking-way
            _y = select.poll()
            _y.register(GPUs_procs[gpu_id].stdout)
            selects[gpu_id] = _y

            nGPU_running += 1

    # Read stdout & Check if task finish
    for i, p in enumerate(GPUs_procs):

        if p is None:
            continue

        if selects[i].poll(0):
        # if True:

            # print("Waiting %d" % i)
            output = p.stdout.readline().decode('utf-8')

            # Finished
            if output == '' and p.poll() is not None:
                GPUs_procs[i] = None
                run_info[i] = None
                selects[i] = None
                nGPU_running -= 1
                continue

            if output:

                if output.startswith("hpt-result="):
                    acc = float(output[11:])
                    print("\033[4;36mResult obtained: %f \033[0m (%s)" % (acc, ' '.join(run_info[i][0][2:]) + ', #%d' % (run_info[i][1][1] + 1)))
                    results[run_info[i][1][0]]["raw_results"].append(acc)

                # print("%d " % i + output, end = '')
                output_files[i].write(output)
                output_files[i].flush()

        else:

            # print("%d" % i, end = '')

            # No new outputs
            if p.poll() is not None:
                GPUs_procs[i] = None
                run_info[i] = None
                selects[i] = None
                nGPU_running -= 1
                continue

    # Check should exit
    if not shouldSchedule:
        break_flag = True
        for p in GPUs_procs:
            if p is not None:
                break_flag = False
        if break_flag:
            break

    # update at a fixed rate    
    time.sleep(0.02)

for r in results:
    r["raw_results"] = np.asarray(r["raw_results"])
    r["runs_count"] = len(r["raw_results"])
    r["mean"] = r["raw_results"].mean()
    r["std"] = r["raw_results"].std()

# sort
results.sort(key = lambda x : x["mean"], reverse = True)

# write CSV
with open(os.path.join(pthprefix, "out.csv"), 'w') as fp:
    
    # Header
    fp.write("Result mean, Result std, #runs, ")
    for key in hpgrid:
        fp.write("%s, " % key)
    fp.write("\n")

    # Rows
    for result in results:
        fp.write("%f, %f, %d, " % (result['mean'], result['std'], result['runs_count']))
        for key in hpgrid:
            fp.write("%s, " % result['HP'][key])
        fp.write("\n")

    print("\033[1;36mResult sheet available at:\033[0m")
    print("\"%s\"" % os.path.join(pthprefix, "out.csv"))
