import torch
import torch.nn
import numpy as np

import wandb

def NumEq(pred, label):
    pred_argmax = pred.detach().cpu().max(1, keepdim=True)[1]
    correct = pred_argmax.eq(label.detach().cpu().view_as(pred_argmax)).sum().item()
    return correct

def GetArgsStr(args, ignore = 'runid'):

    s = ""

    if args.keyargs == "":

        all_vars = list(vars(args).items())
        all_vars = sorted(all_vars, key = lambda x : x[0])

        for i in range(len(all_vars)):

            if all_vars[i][0] == ignore:
                continue

            s = s + "%s" % (all_vars[i][1])
    
    else:

        keys = args.keyargs.split(',')

        for k in keys:
            s += "%s: %s | " % (k, vars(args)[k])
        
        if len(s) > 0:
            s = s[:-3]

    s = s[:128]

    print("Argstr:\n%s" % s)
    return s