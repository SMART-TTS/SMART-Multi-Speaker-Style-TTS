from g2pk import G2p as g2p
import tqdm

G2P = g2p()
with open('metadata_IITP_LARGE_FSNR0_cer_val_g2p.csv', 'w') as w:
	with open('metadata_IITP_LARGE_FSNR0_cer_val.csv', 'r', encoding='utf-8-sig') as r:
		lines = r.readlines()
		print(len(lines))
		for i, line in enumerate(lines):
			floc, text, emo, tag = line.split('|')
			text_g2p = G2P(text)
			w.write(floc+'|'+text_g2p+'|'+emo+'|'+tag)
			print(floc)
			if i % 100 == 0:
				print(i)
