# fashion-how

## Project
  * 대회 : 2022 FASHION-HOW 자율성장 인공지능 대회 참가
  * 참가 이유
      1. 대화를 통해서 사용자에게 좋아할 만한 패션 코디를 추천해준다는 내용이 어렵고 재미있겠다라는 생각이 들었다.
  * 목표
      1. 대화와 패션 추천 어떻게 조합할 지에 대한 고민해보고 실력을 향상시킨다.
      2. 대회를 통해서 전형적인 NLP 모델이 아닌 좀 더 고도화된 모델을 만들어본다.
      
## Dataset
  * 2022 FASHION-HOW 대회
      1. Sub-Task 1,2
          1. FASCODE(FAShion COordination DatasEt / Fashion CODE)는 14,000 장의 패션 이미지 데이터
      2. Sub-Task 3,4
          1. FASCODE(FAShion COordination DatasEt / FAShion CODE)는 옷을 추천해주는 AI 패션 코디네이터와 사용자가 대화를 나눈 데이터셋
          2. AI 패션 코디네이터와 사용자가 나눈 대화에 대하여 발화자 정보, 대화 순서, 대화의 기능에 대해 태깅 정보와 추천받은 옷에 대한 텍스트 정보
  
## Baseline
  1. Sub-Task 1
      1. CutMix를 통해서 데이터 증강
      2. ResNet, DenseNet을 기반으로 이미지를 인코딩하여 분류를 진행
  2. Sub-Task 3,4
      1. 주어진 Baseline을 정확히 이해하고 이를 기반으로 고도화를 진행
          1. 패션 코디네이터가 추천한 옷의 조합과 유사한 옷의 조합으로 negative 데이터를 만든다.
          2. 모델이 구분하기 어려운 조합들을 분류하게끔 하여서(모델 학습을 어렵게 하여서) 모델의 성능을 향상시킨다.
      2. 사용자와 패션코디네이터 간의 대화를 Memory Network을 통해서 인코딩한다.
      3. 인코딩 된 대화 Feature와 Embedding Layer를 통해 구한 옷들의 Feature를 활용해서 옷 조합들 간의 Ranking을 매긴다.
      
      
 ## Model Structure
   1. Sub Task 3
       * 모델 설계 방향
           * ![스크린샷 2022-12-15 오후 2 42 11](https://user-images.githubusercontent.com/48673702/207781767-bbd14897-fb46-49d6-8eb7-4a5c0bbab533.png)
       * 대화 인코더 모델 (Memory Network)
           * ![스크린샷 2022-12-15 오후 2 44 49](https://user-images.githubusercontent.com/48673702/207782122-a32712d2-44f1-4460-b38d-661b7e34bbdf.png)   
       * 모델 구조
           * ![스크린샷 2022-12-15 오후 2 41 28](https://user-images.githubusercontent.com/48673702/207781664-b4af031f-4e25-419c-bb9d-ac4a49c850df.png)
   
## Point
   1. Sub-Task 3
       1. 대회 요구사항
           * 6개의 데이터를 순차적으로 학습 및 저장을 진행하면서 망각효과를 최대한 줄이는 과정이 필요했다. (Continual Learning)
       2. 방향
           * 베이스라인에서 사용된 ewc loss(https://github.com/kuc2477/pytorch-ewc)를 이해하고 활용
   2. Sub-Task 4
       1. 대회 요구사항
           * 학습할 때는 악세사리가 입력되지 않지만 추론을 할 때는 악세사리 데이터가 입력되어서 이에 대한 대처가 필요했다. (Zero-Shot Learning)
       2. 방향
           * 각 옷에 대한 특징을 평균을 매기면서 옷 각각에 대한 Feature에서 옷 조합에 대한 Feature로 처리를 하였다.
           * Sub-Task 3에서는 외투, 상의, 하의, 신발을 각각 Embedding Layer를 거쳐서 Feature를 추출
           * Sub-Task 4에서는 외투, 상의, 하의, 신발에 대한 성, 장식성, 일상성의 Feature를 각각 추출하고 해당 Feature들의 평균을 구한다.
               1. 결과적으로 옷 조합의 성 Feature, 장식성 Feature, 일상성 Feature를 얻게 되고 이를 기반으로 학습 때 보지 않았던 종류의 옷들도 대처할 수 있다.

## Leaderboard
|Task|Score|Rank|
|-----|----|----|
|Sub-Task1|0.7927|3rd|
|Sub-Task3|0.7825|1st|
|Sub-Task4|0.5809|1st|

## Result
  * Sub Task 3,4 우승 & 전체 종합 우승
  



       
        
        
        
        
    
    
    
    
    
  
    
