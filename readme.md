1. wandb.ai 에서 회원 가입 → API 키 저장 
2. pip install wandb
3. wandb_train.py의 main() 에서 sweep_config 설정 (https://docs.wandb.ai/ko/guides/sweeps/define-sweep-configuration)
4. sweep_run.sh에서 pretrained_model_path에 하이퍼파리마터를 최적화하고 싶은 학습된 모델 파일 위치 경로 저장(.pt)
5. bash sweep_run.sh
6. 실행하면 로그인하라고 함 -> API key 입력
