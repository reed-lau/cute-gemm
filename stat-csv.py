from pprint import pprint
import numpy as np

def load_csv_and_stat(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()

    ret = {} 
    for line in lines:
        if line.startswith('=='):
            continue

        fields = line.split(',\"')
        kernel = fields[4].replace('"', '')
        try:
          usec = float(fields[-1].replace('"', '').replace(',', '.'))
        except:
          continue

        if kernel not in ret:
            ret[kernel] = [] # usec
        ret[kernel].append(usec)

    for k, s in ret.items():
        s = np.array(s)
        num = len(s)
        mean = np.mean(s)
        std = np.std(s)
        med = np.median(s)
        ret[k] = (mean, std, med, num) 
    pprint(ret)


load_csv_and_stat('a.csv')

            

