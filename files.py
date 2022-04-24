import csv
def readFile(path=""):
    file=csv.reader(open(path,'r'))
    next(file)
    return [list(map(float,row)) for row in file]