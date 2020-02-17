import pandas as pd
import csv



#reading datasets
with open('cho.txt','r') as f:
	cho = f.read()


# for st in cho.split('\n'):


#print(cho)


with open('cho.txt',newline='') as f:
	cho = csv.reader(f,delimiter='\t')
	for row in cho:
		print(row[1:])
