import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--type', type=str, default='vicuna')
parser.add_argument('--basemodel', type=str, default='basemodel')
parser.add_argument('--data', type=str, default='ShareGPT_V4.3_unfiltered_cleaned_split.json')
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s = 0
e = 68000 - 1#训练数据条数

gpus = [[4],[5],[6],[7]]


num_p = len(gpus)
outdir = '{}/sharegpt_{}_{}_mufp16'.format(args.outdir,s,e)


def split_range(start, end, n, over=False):
    length = end - start + 1  
    base_interval = length // n
    additional = length % n 
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []


gpu_start = 0

for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    gpu_index = []
    for j in range(len(gpus[i])):
        gpu_index.append(gpus[i][j])

    gpu_index_str = ','.join(map(str, gpu_index))  
    ##different type of base model
    if args.type == 'vicuna':
        command = "python ge_data_all_vicuna.py --start={} --end={} --index={} --gpu_index {} --outdir {} --basemodel {} --data {}".format(start, end, index,gpu_index_str,args.outdir,args.basemodel,args.data)
    elif args.type == 'llama2chat':
        command = "python ge_data_all_llama2chat.py --start={} --end={} --index={} --gpu_index {} --outdir {} --basemodel {} --data {}".format(start, end, index,gpu_index_str,args.outdir,args.basemodel,args.data)
    commands.append(command)
    

with ThreadPoolExecutor(max_workers=len(commands)) as executor:#多线程处理数据
    for command in commands:
        executor.submit(run_command, command)
        print(command)
