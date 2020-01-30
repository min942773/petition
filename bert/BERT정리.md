# BERT
![bert](https://miro.medium.com/max/1000/1*oUpWrMdvDWcWE_QSne-jOw.jpeg)
<b>B</b>idirectional <b>E</b>ncoder <b>R</b>epresentations from <b>T</b>ransformers

## BERT란?
대형 코퍼스에서 Unsupervised Learning으로 General-Purpose Language Understanding 모델을 구축하고 (pretraining) Supervised Learning으로 Fine-tuning 해서 QA, STS등의 하위 NLP 태스크에 적용하는 Semi-supervised Learning 모델

> - General-Purpose Language Understanding 모델 : 여러 가지 경우를 대비하여 많은 코퍼스를 학습시킨 모델(?) <br>
> - Fine-tuning : 기존에 학습되어져 있는 모델을 기반으로 아키텍쳐를 새로운 목적에 맞게 변형하고 이미 학습된 모델 Weights로 부터 학습을 업데이트 하는 것 (정교한 파라미터 튜닝)<br>
> - QA : Question & Answer
> - STS : Semantic Textual Similarity : 문장 쌍이 얼마나 유사한지 확인하는 문제

## BERT model
![The Transformer - model architecture](https://user-images.githubusercontent.com/1250095/49935094-73f99c80-ff13-11e8-8ba5-50a008ed4d20.png)
BERT의 모델은 Transformer 기반, BERT는 위 그림에서 왼쪽, 인코더만 사용하는 모델이다.

> - Transformer : 구글이 2017년 발표한 논문인 "Attention is all you need"에서 나온 모델, 기본의 seq2seq의 구조인 인코더-디코더를 따르면서도, 어텐션만으로 구현한 모델, RNN을 쓰지 않고 어텐션만을 사용하여 학습 속도가 무척빠르고 성능도 RNN보다 우수함
> - Attention : Attention(Q, K, V) = Attention Value, 주어진 쿼리에 대해서 모든 키와의 유사도를 각각 구하고 이를 키와 맵핑되어있는 각각의 value에 반영, value를 모두 더해서 리턴한 것이 어텐션 값

## Input Encoding
![Input Embeddings](https://user-images.githubusercontent.com/1250095/50039788-8e4e8a00-007b-11e9-9747-8e29fbbea0b3.png)
BERT는 Transformer와 달리 Positional Encoding을 사용하지 않고 대신 Positional Embeddings를 사용. 여기에 Segment Embeddings를 추가해 각각의 임베딩, 즉 3개의 임베딩을 합산한 결과를 취한다.
- Token Embeddings : 토큰 하나하나
- Segment Embeddings : 입력 문장의 종류에 따라 다른 값 부여
- Position Embeddings : 위치에 따라 차례대로 값이 부여됨<br>
→ 이 값에 대한 각각의 임베딩을 얻어와 합산하고 여기에 LayerNorm & Dropout한 최종 결과를 인코더 블록의 입력값으로 함

> - Layer Normalization(LayerNorm) : Batch Normalization을 변형하여 입력 데이터의 평균과 분산을 이용해 적용
> - Batch Normalization : 활성화함수의 활성화값 또는 출력값을 정규화하는 작업
> - Dropout : 신경망의 뉴런을 부분적으로 생략시키는 것

## Encoder Block
![Encoder Block](https://cdn-images-1.medium.com/max/1600/1*EblTBhM-9mOqYWMARk6ajQ.png)<br>
BERT는 N개의 인코더 블럭을 지니고 있음. Base 모델은 12개, Large 모델은 24개로 구성되는데, 이는 입력 시퀀스 전체의 의미를 N번 만큼 반복적으로 구축하는 것을 의미.<br>
인코더 블럭은 이전 출력값을 현재의 입력값으로 하는 RNN과 유사함. 이 부분은 병렬처리가 아닌 Base 모델은 12번, Large 모델은 24번, 전체가 Recursive하게 반복 처리됨. 블럭 내에서 각각의 입력과 처리 결과는 그림에서와 같이 매 번 Residual connection로 처리하여 이미지 분야의 ResNet이 보여준 것 처럼 그래디언트가 non-linear actications를 거치지 않고 네트워크를 직접 흐르게하여 Explode 또는 Vanishing Gradients 문제를 최소화 하고자 함.

> - Residual Connection : Skip connection과 같은 말로, gradients가 non-linear activation function을 거치지 않고 네트워크로 바로 흐르도록 하는 것

## Multi-Head Attention
![Multi-Head Attention](https://cdn-images-1.medium.com/max/1600/1*9W5_CpuM3Iq09kOYyK9CeA.png)<br>
인코더 블럭의 가장 핵심적인 부분으로 헤드가 여러개인 어텐션을 뜻함. 서로 다른 가중치 행렬을 이용해 어텐션을 h번 계산한 다음 이를 서로 연결한 결과를 가짐.<br>
BERT-base 모델의 경우 각각의 토큰 벡터 768차원을 헤드 수 만큼인 12등분 하여 64개씩 12조각으로 차례대로 분리함. 여기에 Scaled Dot-Product Attention을 적용하고 다시 768차원으로 합침.
이렇게하면 768차원 벡터는 각각 부위별로 12번 Attention 받은 결과가 된다. Softmax는 e의 n승으로 계산하므로 변동폭이 매우 크며, 작은 차이에도 쏠림이 두드러짐. 즉, 값이 큰 스칼라는 살아남고, 작은 쪽은 거의 0에 가까운 값을 multiply하게되어 배제되는 결과를 가져옴.

> - 
