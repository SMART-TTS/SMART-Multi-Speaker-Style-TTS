# SMART-Multi-Speaker-Style-TTS
VITS(Conditional Variational Autoeencoder with Adversarial Learning for End-to-End Text-To-Speech) 기반의 Multi-Speaker Style TTS 모델입니다.
공개된 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한
"소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발"
과제의 일환으로 공개된 코드입니다.

SMART-Multi-Speaker-Style-TTS 모델은 [VITS 모델](https://github.com/jaywalnut310/vits)을 기반으로
Reference speaker embedding와 Style Tag Text로 부터 speaker embedding과 style embedding을 각각 추출하여 반영하는 Multi-Speaker Style TTS 모델입니다.

학습에 사용한 다화자 한국어 데이터셋은 본 과제의 일환으로 SKT에서 수집한 데이터셋으로, 2022년 초에 공개될 예정입니다.

VITS 모델을 기반으로 하여 아래 부분들을 개선하였습니다.

Implemented
- [x] Text encoder를 Transformer block에서 Convolutional block으로 교체, 불안정한 한국어 학습 개선
- [x] pre-trained 된 Ko-Sentence-BERT를 활용하여 Style Tag로 부터 style embedding을 추출, conditional input으로 주어 학습
- [x] 화자, 스타일,  간의 disentanglement를 위한 random reference sampling 기법
- [x] 대용량 화자 + 소용량 화자에 대한 학습
- [ ] Speaker / Style / Text disentanglement via mutual information minimization 기법 적용
- [ ] Multi-language TTS (한국어/영어)

## Requirements
* python 3.6.13
* Ko-Sentence-BERT-SKTBERT dependency (sub directory의 requirements 확인)
* VITS dependency (requirements.txt)

## Training
To train the model, run this command:
<pre>
<code>
python train.py -c configs/[config].json -m [name]
</code>
</pre>

## Evaluation
To evaluate, run:
<pre>
<code>
python inference.py
</code>
</pre>

## Results
현재 [samples](https://github.com/SMART-TTS/SMART-Multi-Speaker-Style-TTS/tree/main/samples)에 저장된 샘플들은 한국어 DB를 사용해 학습한 샘플이며,
2022년 초 새로운 한국어 DB 공개 예정에 있습니다.

## Reference code
* VITS : https://github.com/jaywalnut310/vits
* Ko-Sentence-BERT-SKTBERT : https://github.com/BM-K/KoSentenceBERT-SKT

## Technical document
본 프로젝트 관련 개선사항들에 대한 기술문서는 [여기](https://drive.google.com/file/d/1qHCai1v6KvlRyPcVIYCwYYHwTiPixYSR/view?usp=sharing)를 참고해 주세요.

## Licence
**SMART-Multi-Speaker-Style-TTS** is licensed under the GNU General Public License version 3. The license can be found under LICENSE.
**SMART-Multi-Speaker-Style-TTS** uses a modified version of code from **vits** in which the original code can be found [**here**](https://github.com/jaywalnut310/vits), and is licensed under the MIT license.
