import sys
import time
from functional import pad_batch
import random
#from g2pk import G2p as g2p

hangul_symbols = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''
class TextProcessor:
    def __init__(self):
        self.dict_cho   = {0:u"ᄀ",  1:u"ᄁ",  2:u"ᄂ",  3:u"ᄃ",  4:u"ᄄ",  5:u"ᄅ",  6:u"ᄆ",  7:u"ᄇ",  8:u"ᄈ",  9:u"ᄉ",
            10:u"ᄊ", 11:u"ᄋ", 12:u"ᄌ", 13:u"ᄍ", 14:u"ᄎ", 15:u"ᄏ", 16:u"ᄐ", 17:u"ᄑ", 18:u"ᄒ"}
        self.dict_jung  = {0:u"ᅡ",  1:u"ᅢ",  2:u"ᅣ",  3:u"ᅤ",  4:u"ᅥ",  5:u"ᅦ",  6:u"ᅧ",  7:u"ᅨ",  8:u"ᅩ",  9:u"ᅪ",
            10:u"ᅫ", 11:u"ᅬ", 12:u"ᅭ", 13:u"ᅮ", 14:u"ᅯ", 15:u"ᅰ", 16:u"ᅱ", 17:u"ᅲ", 18:u"ᅳ", 19:u"ᅴ", 20:u"ᅵ"}
        self.dict_jong  = { 0:u" ",   1:u"ᆨ",  2:u"ᆩ",  3:u"ᆪ",  4:u"ᆫ",  5:u"ᆬ",  6:u"ᆭ",  7:u"ᆮ",  8:u"ᆯ",  9:u"ᆰ",  
            10:u"ᆱ", 11:u"ᆲ", 12:u"ᆳ", 13:u"ᆴ", 14:u"ᆵ", 15:u"ᆶ", 16:u"ᆷ", 17:u"ᆸ", 18:u"ᆹ", 19:u"ᆺ", 
            20:u"ᆻ", 21:u"ᆼ", 22:u"ᆽ", 23:u"ᆾ", 24:u"ᆿ", 25:u"ᇀ", 26:u"ᇁ", 27:u"ᇂ"}
        self.hangul_symbols = hangul_symbols
        self.grapheme_to_idx = {char: idx for idx, char in enumerate(hangul_symbols)}
        self.idx_to_grapheme = {idx: char for idx, char in enumerate(hangul_symbols)}

    def text_process(self, text):
        return pad_batch(self.syllables_to_cjj(text))

    def syllables_to_cjj(self, syllable_seq): # ##i cjj : Cho/Jung/Jong in Hangul
        cjj_id = []
        for j in range(len(syllable_seq)):
            cjj = []
            sen = syllable_seq[j]

            sen = u" "+ sen + u" " #+ u"␃" # ␃: EOS
            for i in range(len(sen)):
                if len(sen[i].encode()) == 3:
                    h___ = sen[i].encode()[0]-224
                    _h__ = (sen[i].encode()[1]-128) // 4
                    next_ = (sen[i].encode()[1]-128) % 4
                    __h_ = (next_*64 + sen[i].encode()[2]-128) // 16
                    ___h = (next_*64 + sen[i].encode()[2]-128) % 16
                    hex = h___ * 4096 + _h__ * 256 + __h_ * 16 + ___h

                    if hex == 9219: 
                        cjj = cjj + [u"␃"] # ##i EOS
                        continue
                    if hex < 44032:
                        continue
                    cho  = self.dict_cho[(hex - 44032) // 588]
                    jung = self.dict_jung[((hex - 44032) % 588) // 28]
                    jong  = self.dict_jong[((hex - 44032) % 588) % 28]
                    if jong == u" ": cjj = cjj + [cho, jung]
                    else : cjj = cjj + [cho, jung, jong]
                else:
                    if sen[i] not in self.hangul_symbols: continue
                    cjj = cjj + [sen[i]]
            cjj_id.append([self.grapheme_to_idx.get(char,1) for char in cjj])

        return cjj_id        

sys.path.append('./KoBERT/')
from gluonnlp.data import SentencepieceTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

class TagProcessor:
    def __init__(self):
        self.tokenizer = SentencepieceTokenizer(get_tokenizer(cachedir='./kobert/saved'))
        _, self.vocab = get_pytorch_kobert_model(cachedir='./kobert/saved')

    def tag_process(self, tag):
        tag_list = []
        for i in range(len(tag)):
            tag_list.append(self.vocab[self.tokenizer(tag[i])])
        return pad_batch(tag_list)

    def tag_augment(self, tag):
        if tag == '지문':
            tag = random.choice(['평범하게', '또박또박하게'])
    
        augment_list = [tag]
        tag_end = tag[-1]
    
        if tag_end == '서':
            pass
        elif tag_end == '한':
            augment_list.append(tag+' 목소리로')
            augment_list.append(tag+' 듯한')
            augment_list.append(tag[:-1]+'하게')
        elif tag_end == '운':
            augment_list.append(tag+' 목소리로')
            augment_list.append(tag+' 듯한')
            augment_list.append(tag[:-2]+'꼽게')
        elif tag_end == '된':
            augment_list.append(tag+' 목소리로')
            augment_list.append(tag+' 듯한')
            augment_list.append(tag[:-1]+'되어')
        elif tag_end == '이':
            augment_list.append(tag[:-1])
            augment_list.append(tag[:-2]+'는')
        elif tag_end == '는':
            augment_list.append(tag+' 목소리로')
        elif tag_end == '척':
            augment_list.append(tag+'하며')
        elif tag_end == '은':
            augment_list.append(tag[:-1]+'게')
            augment_list.append(tag+' 목소리로')
        elif tag_end == '인':
            augment_list.append(tag[:-1]+'이게')
            augment_list.append(tag[:-1]+'으로')
            augment_list.append(tag+' 목소리로')
        elif tag_end == '며':
            augment_list.append(tag[:-1]+'듯')
            augment_list.append(tag[:-1]+'는')
        elif tag_end == '로':
            pass
        elif tag_end == '게':
            if tag[-2:] == '하게':
                augment_list.append(tag[:-2]+'한 목소리로')
                augment_list.append(tag[:-2]+'한')
            elif tag[-2:] == '럽게':
                augment_list.append(tag[:-2]+'러운 목소리로')
                augment_list.append(tag[:-2]+'러운')
            elif tag[-2:] == '롭게':
                augment_list.append(tag[:-2]+'로운 목소리로')
                augment_list.append(tag[:-2]+'로운')
            elif tag[-2:] == '있게':
                augment_list.append(tag[:-2]+'있는 목소리로')
                augment_list.append(tag[:-2]+'있는')
            elif tag[-2:] == '맞게':
                augment_list.append(tag[:-2]+'맞은 목소리로')
                augment_list.append(tag[:-2]+'맞은')
            elif tag[-2:] == '차게':
                augment_list.append(tag[:-2]+'찬 목소리로')
                augment_list.append(tag[:-2]+'찬')
        elif tag_end == '듯':
            augment_list.append(tag+'이')
            if tag[-2:] == ' 듯':
                augment_list.append(tag[:-2])
            elif tag[-2:] == '하듯':
                augment_list.append(tag[:-1]+'는')
                augment_list.append(tag[:-1]+'며')
                augment_list.append(tag[:-1]+'는 목소리로')
            elif tag == '엄살떨듯':
                pass
            else:
                augment_list.append(tag[:-1]+'는')
                augment_list.append(tag[:-1]+'는 목소리로')
 
        tag = random.choice(augment_list)
        return tag
