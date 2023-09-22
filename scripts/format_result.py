import os
def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False
    
suffix = "all.csv"
dir = '/export/home/cond-text-gen'
dir  = os.path.join(dir)
files = os.listdir(dir)
os.chdir(dir)
files = [ f for f in files if f.endswith(suffix)]
import csv
for f in files:
    new_data = [] 
    prefix = f[:-4]
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            head = row
            break
        for row in reader:
            data.append(row)
    # go over the first row
    manips = []
    for name, value in zip(head, data[0]):
        if is_float(value) and float(value) < 1:
            manips.append(name)
    for d in data:
        tmp = []
        for name, value in zip(head, d):
            if name in manips:
                tmp.append(100*float(value))
            else:
                tmp.append(value)    
        new_data.append(tmp)
    print(new_data)
    
    with open(f"{prefix}_convert.csv", 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(head)
        for d in new_data:
            spamwriter.writerow(d)
