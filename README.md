
# 🏆 Sketch 이미지 분류를 위한 Image Classification

## 🥇 팀 구성원
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/Yeon-ksy"><img src="https://avatars.githubusercontent.com/u/124290227?v=4" width="100px;" alt=""/><br /><sub><b>김세연</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/jihyun-0611"><img src="https://avatars.githubusercontent.com/u/78160653?v=4" width="100px;" alt=""/><br /><sub><b>안지현</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/dhfpswlqkd"><img src="https://avatars.githubusercontent.com/u/123869205?v=4" width="100px;" alt=""/><br /><sub><b>김상유</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/K-ple"><img src="https://avatars.githubusercontent.com/u/140207345?v=4" width="100px;" alt=""/><br /><sub><b>김태욱</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/myooooon"><img src="https://avatars.githubusercontent.com/u/168439685?v=4" width="100px;" alt=""/><br /><sub><b>김윤서</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br />

## 😷 프로젝트 소개
Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 그 중 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이번 프로젝트는 이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다.  

<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.09.10 ~ 2024.09.26

프로젝트 세부 일정

- 2024.09.10 ~ 2024.09.12 : 데이터셋 분석 및 EDA
- 2024.09.12 ~ 2024.09.15 : Augmentation 실험
- 2024.09.10 ~ 2024.09.26 : Model 실험
- 2024.09.13 ~ 2024.09.14 : 코드 모듈화
- 2024.09.24 ~ 2024.09.26 : 모델 앙상블 실험
- 2024.09.25 ~ 2024.09.26 : 모델 평가
- 2024.09.25 ~ 2024.09.26 : Wandb 연동

<br />

## 🥈 프로젝트 결과
- Private 리더보드에서 최종적으로 아래와 같은 결과를 얻었습니다.
<img align="center" src="imgs/result.png" width="600" height="80">

<br />

## 🥉 데이터셋 구조
```
 data/
 ├── eval
 │   ├── images
 │   │   └── eval_dataset
 │   └── info.csv
 └── train
     ├── imgaes_gen
     │   └── ID_gender_race_age
     │       ├── mask1
     │       ├── mask2
     │       ├── mask3
     │       ├── mask4
     │       ├── mask5
     │       ├── incorrect_mask
     │       └── normal
     └── train.csv
```
이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
- 전체 사람 명 수 : 4,500
- 한 사람당 사진의 개수 : 7장 (마스크 착용 5, 이상하게 착용(코스크, 턱스크) 1, 미착용 1)
- 이미지 크기 : (384, 512)
- 분류 클래스 : 마스크 착용 여부(3), 성별(2), 나이(3)를 기준으로 총 18개의 클래스 존재
- 전체 데이터 중 학습데이터 60%, 평가데이터 40%로 사용

<br />

## 🥉 프로젝트 구조
```
project/
├── .gitignore
├── data_gen.ipynb
├── data_rembg.ipynb
├── dataset.py
├── rembg_dataset.py
├── loss.py
├── model.py
├── train.py
├── skf_train.py
├── train_multiclass.py
├── train_optuna.py
├── train_cutmix_60s.py
├── train_cutmix_all_ages.py
├── skf_train_multiclass.py
├── train_cutmix_multiclass.py
├── rembg_train.py
├── rembg_train_multiclass.py
├── inference.py (기본 코드와 동일하여 업로드X)
├── inference_multiclass.py
└── rembg_inference_multiclass.py
```

#### 1) `dataset.py`
- 마스크 데이터셋을 읽고 전처리를 진행한 후 데이터를 하나씩 꺼내주는 Dataset 클래스를 구현한 파일 
- CustomAugmentation, MaskBaseDataset, MaskMultiLabelDataset 구현 
- `rembg_dataset.py` : 배경제거한 데이터셋을 활용하기 위해 이미지를 로드할 때 .convert(‘RGB’)를 추가한 파일
#### 2) `loss.py`
- 이미지 분류에 사용될 수 있는 다양한 Loss 들을 정의한 파일
- Cross Entropy, Focal Loss, Label Smoothing Loss, F1 Loss 구현
#### 3) `model.py`
- 데이터를 받아 연산을 처리한 후 결과 값을 내는 Model 클래스를 구현하는 파일 
- vit_base_patch16_224, vit_small_patch16_384, vgg16_bn, resnet50, resnet101, densenet121, densenet201, efficientnet_b1, inception_resnet_v2, swin_tiny_patch4_window7_224, swin_large_patch4_window12_384 구현
#### 4) `train.py`
- 실제로, 마스크 데이터셋을 통해 CNN 모델 학습을 진행하고 완성된 모델을 저장하는 파일 
- `skf_train.py` : Stratified k-fold 적용
- `train_multiclass.py` : Multi-Labeling 적용
- `train_optuna.py` : optuna 적용
- `train_cutmix_60s.py` : 60세 이미지 패치로 CutMix 
- `train_cutmix_all_ages.py` : 모든 나이대에 이미지 패치로 CutMix 
- `skf_train_multiclass.py` : Multi-Labeling에 Stratified k-fold 적용 
- `train_cutmix_multiclass.py` : Multi-Labeling에 모든 나이대에 이미지 패치 CutMix 적용 
- `rembg_train.py, rembe_train_multiclass.py` : 배경제거한 데이터셋을 활용하기 위해 라이브러리 선언만 변경 
#### 5) `inference.py`
- 학습 완료된 모델을 통해 test set 에 대한 예측 값을 구하고 이를 .csv 형식으로 저장하는 파일 
- `inference_multiclass.py` : Multi-Labeling 적용
- `rembg_inference_multiclass.py` : 배경제거한 데이터셋을 활용하기 위해 라이브러리 선언만 변경 
#### 6) `etc (.ipynb)`
- `data_gen.ipynb`  : albumentations 라이브러리를 활용하여 train/images datasets에 1개씩 존재하던 incorrect_mask, normal image를 4개씩 Augmentation 적용하여 Up-Sampling 하는 코드
- `data_rembg.ipynb` : rembg 라이브러리를 활용하여 train/images datasets에 remove 함수를 통해 사람을 제외한 배경을 제거하는 코드

<br />

## ⚙️ 설치

#### Dependencies
이 모델은 Ubuntu 20.04.6 LTS, CUDA Version: 12.2, Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

- albumentations==1.4.4
- matplotlib==3.8.4
- numpy==2.1.1
- opencv_python==4.9.0.80
- opencv_python_headless==4.10.0.84
- pandas==2.2.3
- Pillow==10.4.0
- scikit_learn==1.4.2
- timm==1.0.9
- torch==2.1.0
- torchvision==0.16.0
- tqdm==4.66.1

Install dependencies: `pip install -r requirements.txt`

<br />

## 🚀 빠른 시작
#### Train
`python train.py --name [실험명]`

이 외 다양한 학습 방법은 `🥉프로젝트 구조/4) train.py`를 참고해주세요!

#### Evaluation
`python inference.py --model_dir [모델저장경로]`

<br />

## 🏅 Wrap-Up Report   
- [Wrap-Up Report👑](https://onedrive.live.com/edit?id=1D3C82CAEE19B27B!sbb0352dc60244cd4bd69c3597c7c9088&resid=1D3C82CAEE19B27B!sbb0352dc60244cd4bd69c3597c7c9088&cid=1d3c82caee19b27b&ithint=file%2Cdocx&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3cvYy8xZDNjODJjYWVlMTliMjdiL0VkeFNBN3NrWU5STXZXbkRXWHg4a0lnQnFfbU1nbTNfUVlnUFhyQ193d0otQWc_ZT16ZDRlbWI&migratedtospo=true&wdo=2)