# test_noise
영상 노이즈 생성 도구의 정확성 분석

```bash
pip install -r requirements.txt # 다음 명령어를 통해 의존성을 설치합니다.
conda install -c conda-forge py6s -y # conda 환경이라면 해당 명령어를 통해 설치를 진행합니다.
```



### 하위 모듈 설명

- metadata_sample 

    예시 메타데이터 샘플 파일이 들어있습니다.
- noise_generator 

    각 noise들이 구현된 class들이 존재합니다
- noise_eval

    noise generator에서 구현된 noise를 평가하기 위한 모듈입니다.
- denoise

    오픈 소스로 제공되어 있는 왜곡(노이즈) 전처리 프로그램을 사용해 생성된 보정 이미지와 L1, L2 간의 유사도 비교하기 위한 모듈입니다.