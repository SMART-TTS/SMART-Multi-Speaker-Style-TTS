import random
r = open('metadata_IITP_non_train_g2p.csv', 'r', encoding='utf-8-sig')
spkr_dict = dict()
lines = r.readlines()
for line in lines:
	floc, text, emo, tag = line.split('|')
	db_type = floc.split('/')[5]
	if db_type == 'SMALL':
		cur_spkr = floc.split('/')[6]
		if cur_spkr not in spkr_dict:
			spkr_dict[cur_spkr] = dict()
			spkr_dict[cur_spkr]['neu'] = list()
			spkr_dict[cur_spkr]['emo'] = list()
			spkr_dict[cur_spkr]['all'] = list()
		if tag.strip() == '지문':
			spkr_dict[cur_spkr]['neu'].append(line)
		else:
			spkr_dict[cur_spkr]['emo'].append(line)
		spkr_dict[cur_spkr]['all'].append(line)
#print(len(spkr_dict['MPHG0']['neu']), len(spkr_dict['MPHG0']['emo']))
w = open('metadata_IITP_test_g2p_small.csv', 'w', encoding='utf-8')
for spkr in spkr_dict:
	print(spkr)
	emo_ref_lines = random.sample(spkr_dict[spkr]['emo'], 5)
	neu_ref_lines = random.sample(spkr_dict[spkr]['neu'], 5)
	ref_lines = emo_ref_lines+neu_ref_lines
	text_lines = random.sample(spkr_dict[spkr]['all'], 10)
	for i, text_line in enumerate(text_lines):
		text = text_line.split('|')[1]
		ref_floc, _, emo, tag = ref_lines[i].split('|')
		new_line = ref_floc+'|'+text+'|'+emo+'|'+tag
		print(new_line)
		w.write(new_line)
