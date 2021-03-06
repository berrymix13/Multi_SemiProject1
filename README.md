# 장애인 콜택시 대기시간 예측

#### MultiCampus Semi Project (2022.05.02~2022.05.19)



## 프로젝트 설명 ๑◕‿‿◕๑

일반 택시와 달리 콜 예약 - 배차까지의 시간이 긴 장애인 콜택시에 대해

대기시간을 길게 만드는 변수들에 대해 분석하고, 

회귀 모델을 만들어 대기시간을 예측해보고자 한다.

## 프로젝트 목표
- 콜택시 증차량과 평균 대기시간 사이의 상관관계 파악
- 실시간 데이터를 이용한실시간 대기시간 예측

## 데이터 설명

- 사용 데이터

  - 장애인 콜택시 일일 이용 Open API
  - 서울시 교통량(속도) Open API
  - 년도별 시간대별 콜택시 운행차량 수 데이터
  - 2019-2022 일별 강수량, 적설량 데이터

  

- 모델에 사용되는 변수 종류

  - 출발지 구군   : 콜택시를 이용하는 지역구
  - 출발지 상세
  - 시간대
  - 요일
  - 언급량    : 비슷한 위치에서 동시간대의 수요
  - 강수량
  - 적설량
  - 운행차량수 

## 모델링 과정

1) statsmodels의 ols를 사용해 다중 회귀분석을 수행
2) 후진제거법을 통해 변수를 선택
3) 모형에 대한 잔차분석을 진행
4) SVM, 의사결정 회귀나무, XGBoost 등의 모델 적용해보기
5) 성능 비교 후 모델 선택

