# SMART-Multi-Speaker-Style-TTS
VITS(Conditional Variational Autoeencoder with Adversarial Learning for End-to-End Text-To-Speech) 기반의 Multi-Speaker Style TTS 모델입니다.
공개된 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한
"소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발"
과제의 일환으로 공개된 코드입니다.

SMART-Multi-Speaker-Style-TTS 모델은 [VITS 모델](https://github.com/jaywalnut310/vits)을 기반으로
Speaker code와 Style Tag Text로 부터 speaker embedding과 style embedding을 각각 추출하여 반영하는 Multi-Speaker Style TTS 모델입니다.

학습에 사용한 다화자 한국어 데이터셋은 본 과제의 일환으로 SKT에서 수집한 데이터셋으로, 2022년 초에 공개될 예정입니다.

VITS 모델을 기반으로 하여 아래 부분들을 개선하였습니다.

Done
* 한국어 Text 반영을 위해 Convolutional layer 기반의 text encoder
* Style Tag 를 활용한 발화 스타일 인코딩 기법
* 화자와 text 간의 disentanglement를 위한 random reference sampling 기법
* 대용량 화자 + 소용량 화자에 대한 학습

To do
* Speaker / Style / Text disentanglement via mutual information minimization 기법 적용
* Multi-language TTS (한국어/영어)

## Requirements (incomplete)
* python 3.7.9

## Preprocessing (incomplete)
To preprocess mel spectrogram and linear spectrogram magnitude, run this command:
<pre>
<code>
python prepare_data.py
</code>
</pre>

## Training (incomplete)
To train the model, run this command:
<pre>
<code>
python train_transformer.py
</code>
</pre>


## Evaluation (incomplete)
To evaluate, run:
<pre>
<code>
python synthesize.py --rhythm_scale=1.0 --restore_step1=860000
</code>
</pre>

## Results (incomplete)
Synthesized audio samples can be found in ./samples

현재 ./samples에 저장된 샘플들은 연구실 보유중인 DB를 사용해 학습한 샘플이며,
내년초 새로운 한국어 DB 공개 예정에 있습니다.

## Reference code (incomplete)
* Transformer-TTS : https://github.com/soobinseo/Transformer-TTS
* SMART-Vocoder : https://github.com/SMART-TTS/SMART-G2P
* SMART-G2P : https://github.com/SMART-TTS/SMART-Vocoder
