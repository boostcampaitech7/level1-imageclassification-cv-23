# 🏆 마스크 착용 상태 분류를 위한 Image Classification

## 🥇 팀 구성원
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/bogeoung"><img src="https://avatars.githubusercontent.com/u/50127209?v=4?s=100" width="100px;" alt=""/><br /><sub><b>김보경</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/SangphilPark"><img src="https://avatars.githubusercontent.com/u/81211140?v=4?s=100" width="100px;" alt=""/><br /><sub><b>박상필</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/LTSGOD"><img src="https://avatars.githubusercontent.com/u/78635028?v=4?s=100" width="100px;" alt=""/><br /><sub><b>이태순</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/d-a-d-a"><img src="https://avatars.githubusercontent.com/u/109848297?v=4?s=100" width="100px;" alt=""/><br /><sub><b>임현명</b></sub><br />
    </td>
        <td align="center"><a href="https://github.com/CheonJiEun"><img src="https://avatars.githubusercontent.com/u/53997172?v=4?s=100" width="100px;" alt=""/><br /><sub><b>천지은</b></sub><br />
    </td>
  </tr>
</table>
</div>

<br />

## 😷 프로젝트 소개
COVID-19의 확산 방지를 위해 사람들은 마스크 착용 및 사회적 거리 두기 등의 많은 노력을 하고 있습니다. 이 중 마스크 착용은 감염자로부터의 전파 경로를 차단하기 위한 중요한 방법으로, 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 많은 사람들의 마스크 착용 상태를 수동으로 검사하는 것은 많은 비용과 시간이 요구됩니다. 이를 해결하기 위해 적은 비용으로 마스크 착용 여부를 판별 가능한 이미지 분류 모델을 개발하였습니다.

이번 프로젝트는 `부스트캠프 AI Tech` CV 트랙 내에서 진행된 대회이며 F1-Score를 통해 최종평가를 진행하였습니다.

<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2023.04.10 ~ 2023.04.20

프로젝트 세부 일정

- 2023.04.10 ~ 2023.04.13 : 데이터셋 분석 및 EDA, 프로젝트 환경 설정
- 2023.04.13 ~ 2023.04.14 : Baseline 코드 정립, Augmentation 실험
- 2023.04.14 ~ 2023.04.16 : Up-Sampling, 이미지 배경 제거
- 2023.04.15 ~ 2023.04.19 : Model 실험
- 2023.04.16 ~ 2023.04.19 : Stratified k-fold, Label Smoothing, Multi-Labeling 구현 및 실험
- 2023.04.18 ~ 2023.04.20 : Cutmix 적용, Loss 및 나이 기준 변경 실험
- 2023.04.20 ~ 2023.04.20 : Optuna 연동

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
이 모델은 Ubuntu 18.04.5 LTS, Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

- torch == 1.7.1
- torchvision==0.8.2
- pandas~=1.2.0
- scikit-learn~=0.24.1
- matplotlib==3.5.1
- numpy~=1.21.5
- python-dotenv~=0.16.0
- Pillow~=7.2.0
- sklearn~=0.0
- timm==0.6.13

Install dependencies: `pip3 install -r requirements.txt`

<br />

## 🚀 빠른 시작
#### Train
`python train.py --name [실험명]`

이 외 다양한 학습 방법은 `🥉프로젝트 구조/4) train.py`를 참고해주세요!

#### Evaluation
`python inference.py --model_dir [모델저장경로]`

<br />

## 🏅 Wrap-Up Report   
- [Wrap-Up Report👑](https://github.com/boostcampaitech5/level1_imageclassification-cv-08/blob/main/docs/%5BCV-08%5D%20LV1%20%EB%9E%A9%EC%97%85%20%EB%A6%AC%ED%8F%AC%ED%8A%B8.pdf)
