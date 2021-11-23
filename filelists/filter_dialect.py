import random

filter_ls = list()
for i in range(100677, 101438, 1):
	filter_ls.append(i)
for i in range(102158, 102608, 1):
	filter_ls.append(i)
for i in range(103515, 103837, 1):
	filter_ls.append(i)
for i in range(108151, 108628, 1):
	filter_ls.append(i)
for i in range(111026, 111058, 1):
	filter_ls.append(i)
for i in range(111225, 115000, 1):
	filter_ls.append(i)
print(len(filter_ls))
r = open('metadata_IITP_train_g2p.csv', 'r', encoding='utf-8-sig')
w = open('metadata_IITP_train_g2p_filtered.csv', 'w', encoding='utf-8')

lines = r.readlines()
for line in lines:
	floc, text, emo, tag = line.split('|')
	fname = floc.split('/')[8]
	fnum = fname.split('_')[1][:-4]
	fnum = int(fnum)
	if fnum not in filter_ls:
		w.write(line)
	else:
		print(fnum)
