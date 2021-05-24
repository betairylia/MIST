import os
import argparse
import time

def tmux(command):
    os.system('tmux %s' % command)

def tmux_shell(command):
    command = command.replace("\"", "\\\"")
    tmux('send-keys "%s" "C-m"' % command)

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type = str, default = '0', help="GPUs to be used.")
parser.add_argument('--prefix', type = str, default = '', help="Prefix of the command.")

args = parser.parse_args()
args.gpus = args.gpus.split(',')

invoke_cmd = ""

invoke_cmd = input("Paste / Enter the agent invoking command here:\n").strip()

# invoke_cmd = "%s --count %d%s" % (invoke_cmd[:11], args.runs, invoke_cmd[11:])

print("Agent invoking command is: \n    %s" % invoke_cmd)

# Tmux session
tmux('new-window -n \'WandB Sweep Agents\'')
tmux('select-window -t $0')

# Initialize multiple GPU
nGPUs = len(args.gpus)

# print("\033[1;36mCommands for output monitoring:\033[0m")
# for i in range(nGPUs):
#     print("less +F -R \"%s\"" % (os.path.join(pthprefix, "GPU_%s.log" % args.gpus[i])))

# Run
# Populate tasks
for i in range(len(args.gpus)):

    # wait some seconds; otherwise the output folder will be the same ...
    time.sleep(0.1)
    tmux_shell("cd %s" % os.getcwd())

    # Get cmd
    cmd = invoke_cmd

    cmd = "%s ; CUDA_VISIBLE_DEVICES=%s %s" % (args.prefix, args.gpus[i], cmd)
    tmux_shell(cmd)

    if(i < (nGPUs - 1)):
        tmux('split-window -h')

tmux('select-layout tiled')
