
# 🏆 Sketch 이미지 분류를 위한 Image Classification

<br />

## ✏️ Introduction
Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 그 중 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상이나 질감과 같은 세부적인 정보가 상대적으로 부족하지만, 객체의 기본적인 형태와 구조에 중점을 두고 표현되는 특징이 있습니다. 이번 프로젝트는 이러한 스케치 데이터의 특성을 분석하여 모델이 객체의 기본적인 형태와 구조를 학습 및 인식하도록 함으로써 일반적인 이미지 데이터와의 차이점을 이해하고 모델 개발 역량을 높이는데에 초점을 두었습니다.  
<br />

## 📅 Schedule
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

## 🥈 Result
- Private 리더보드에서 최종적으로 아래와 같은 결과를 얻었습니다.
<img align="center" src="imgs/result.png" width="600" height="50">

<br />

## 🗃️ Dataset
```
data/
│
├── sample_submission.csv
├── test.csv
├── train.csv
│
├── test/
│   ├── 0.JPEG
│   ├── 1.JPEG
│   ├── 2.JPEG
│   ├── ...
│
	├── train/
│   ├── n01443537/
│   ├── n01484850/
│   ├── ... 
```
데이터셋은 검수 및 정제된 ImageNet Skech 데이터셋으로 이미지 수량이 많은 상위 500개의 객체로 이뤄져 있으며, 데이터는 다음과 같이 요약됩니다.
- 각 클래스에 따라 파충류, 개 등 유사한 클래스가 다수 포함되어 있습니다.

<div style="display: flex; justify-content: center; gap: 10px;">
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/1a30b986-8b15-439b-97b5-c1f79f9f3579" width="300"/>
    <p>n01729322 (target 32)</p>
  </div>
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/bcba02eb-384d-47bf-b992-f21f2e95746c" alt="포메라니안 이미지" width= "300"/>
    <p>n01735189 (target 33)</p>
  </div>
</div>


- 각 클래스 당 29~31개의 이미지를 가지고 있습니다.

  <img src="https://github.com/user-attachments/assets/57b6af62-329c-4401-89ad-22c8e534f42d" width="500"/>

- 이미지의 크기는 다양하며, 밑의 그래프를 따릅니다.

  <img src="https://github.com/user-attachments/assets/d4a88a8e-b85c-46fb-8f65-ce46f994fa1c" width="500"/>

- 학습데이터는 15,021개이며, 평가데이터는 10,014개입니다.
<br />
<br />
## ⚙️ Requirements

### env.
이 프로젝트는 Ubuntu 20.04.6 LTS, CUDA Version: 12.2, Tesla v100 32GB의 환경에서 훈련 및 테스트되었습니다.

### Installment
또한, 이 프로젝트에는 다앙한 라이브러리가 필요합니다. 다음 단계를 따라 필요한 모든 의존성을 설치할 수 있습니다.
``` bash
  git clone https://github.com/boostcampaitech7/level1-imageclassification-cv-23.git
  cd sketch-classification
  pip install -r requirements.txt
```

<br />

### 

<br />

## 🧑‍🤝‍🧑 Contributors
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

## ⚡️ Wrap-Up Report   
- 프로젝트에 대한 자세한 내용은 [Wrap-Up Report](https://onedrive.live.com/edit?id=1D3C82CAEE19B27B!sbb0352dc60244cd4bd69c3597c7c9088&resid=1D3C82CAEE19B27B!sbb0352dc60244cd4bd69c3597c7c9088&cid=1d3c82caee19b27b&ithint=file%2Cdocx&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3cvYy8xZDNjODJjYWVlMTliMjdiL0VkeFNBN3NrWU5STXZXbkRXWHg4a0lnQnFfbU1nbTNfUVlnUFhyQ193d0otQWc_ZT16ZDRlbWI&migratedtospo=true&wdo=2) 에서 확인할 수 있습니다.