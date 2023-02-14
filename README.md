# [KDT Programmers Dev Course] Final Project
  
## &#128221; 프로젝트 소개
![READ UR MIND](http://ec2-54-93-233-65.eu-central-1.compute.amazonaws.com/static/assets/img/logo.png)

일기를 공유해주세요, 지금 상황에 알맞은 책을 추천드립니다.

[READ UR MIND](http://ec2-54-93-233-65.eu-central-1.compute.amazonaws.com/main/)

> "일기를 단순히 기록하는 것에서 그치는 것이 아닌, 일기에서 추출된 자신의 감정을 되돌아 보고 비슷한 감정, 키워드로 추천되는 도서를 읽으며 해소하는 것까지 서비스화 하여 현대인의 감정 해소에 도움을 주고자 한다."


## &#128101; 팀 소개
### A-4
양정우 (Leader) [@widdn955](https://github.com/wjddn955)  
김유라 [@yr2351](https://github.com/yr2351)  
심지현 [@jihhyeon](https://github.com/jihhyeon)  
여언주 [@eejj357](https://github.com/eejj357)  
이상민 [@skybluelee](https://github.com/skybluelee)


## &#128187; 서비스 구조
### Service Architecture
![archi1](http://ec2-54-93-233-65.eu-central-1.compute.amazonaws.com/static/assets/img/portfolio/archi_1.png)
1. 사용자가 일기를 입력하면 일기 데이터를 감정 분류 모델과 키워드 추출 모델에 입력으로 전달한다.
2. 감정 분류 모델에서 일기의 상위 감정 3개가 추출된다.
3. 키워드 추출 모델에서 일기의 상위 키워드 3개가 추출된다.
4. 2과 3의 결과(감정, 키워드)과 DB의 도서 데이터와 비교하여 비슷한 키워드와 감정을 가진 도서를 추출한다.
5. 추출한 감정, 키워드와 추천하는 도서를 페이지에 출력한다.

### BERT Model
![archi2](http://ec2-54-93-233-65.eu-central-1.compute.amazonaws.com/static/assets/img/portfolio/bert_img2.png)
- [KcELECTRA_base](https://github.com/Beomi/KcELECTRA)
- [KcBERT_base](https://github.com/Beomi/KcBERT)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
- [DistilKoBERT](https://github.com/monologg/DistilKoBERT)

위 다섯가지 BERT계열 모델들에 대해 fine-tuning과 early-stopping을 진행하여 accuracy가 가장 높게 나온 **KcELECTRA_base** 모델을 선정

### KeyBERT Model
![archi3](http://ec2-54-93-233-65.eu-central-1.compute.amazonaws.com/static/assets/img/portfolio/keybert.png)

일기와 도서에서 키워드를 추출하기 위한 키워드 추출 모델

[KeyBERT](https://github.com/MaartenGr/KeyBERT)는 BERT기반의 모델로, N-gram을 위해 단어를 임베딩 후 cosine similarity를 계산하여 어떤 N-gram 또는 단어가 문서와 가장 유사한지 찾아냄

## &#128736; Skills
- OS
    - <img src="https://img.shields.io/badge/Ubuntu 20.04 -E95420?style=flat&logo=Ubuntu&logoColor=white"/>
- Frontend
    - <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=HTML5&logoColor=white"/>
    - <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=CSS3&logoColor=white"/>
    - <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=JavaScript&logoColor=black"/>
- Backend
    - <img src="https://img.shields.io/badge/Amazon EC2-FF9900?style=flat&logo=Amazon EC2&logoColor=white"/>
    - <img src="https://img.shields.io/badge/Django-092E20?style=flat&logo=Django&logoColor=white"/>
    - <img src="https://img.shields.io/badge/SQLite-003B57?style=flat&logo=SQLite&logoColor=white"/>
- Deep learning
    - <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
    -  <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat&logo=Google Colab&logoColor=white"/>
    - <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>
    - <img src="https://img.shields.io/badge/Pytorch Lightning-792EE5?style=flat&logo=PyTorch Lightning&logoColor=white"/>
- Groupware
    - <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=GitHub&logoColor=white"/>
    - <img src="https://img.shields.io/badge/Asana-273347?style=flat&logo=Asana&logoColor=white"/>
    - <img src="https://img.shields.io/badge/Slack-4A154B?style=flat&logo=Slack&logoColor=white"/>


## &#128269; References
- [KcElectra](https://github.com/Beomi/KcELECTRA)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [AI HUB](https://aihub.or.kr/)
