import random
r = open('metadata_IITP_non_train_g2p.csv', 'r', encoding='utf-8-sig')
spkr_dict = dict()
text_dict = dict()
text_dict['emo'] = list()
text_dict['neu'] = list()

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
			text_dict['neu'].append(line)
		else:
			spkr_dict[cur_spkr]['emo'].append(line)
			text_dict['emo'].append(line)
		spkr_dict[cur_spkr]['all'].append(line)

#print(len(spkr_dict['FSNR0']['neu']), len(spkr_dict['FSNR0']['emo']))
print(len(text_dict['emo']), len(text_dict['neu']))
w = open('metadata_IITP_test_g2p_small_same_text_ref_20.csv', 'w', encoding='utf-8')
emo_text_lines = random.sample(text_dict['emo'], 10)
neu_text_lines = random.sample(text_dict['neu'], 10)
text_lines = emo_text_lines+neu_text_lines

for spkr in spkr_dict:
	emo_ref_lines = random.sample(spkr_dict[spkr]['emo'], 10)
	neu_ref_lines = random.sample(spkr_dict[spkr]['neu'], 10)
	ref_lines = emo_ref_lines+neu_ref_lines

	for i, ref_line in enumerate(ref_lines):
		w.write(ref_line)
	'''
	for i, text_line in enumerate(text_lines):
		_, text, emo, tag = text_line.split('|')
		ref_floc, _, _, _ = ref_lines[i].split('|')
		new_line = ref_floc+'|'+text+'|'+emo+'|'+tag
		w.write(new_line)
	'''
