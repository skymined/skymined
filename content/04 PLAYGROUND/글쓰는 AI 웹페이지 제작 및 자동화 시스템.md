---
tags:
  - imwriteri
created: 2026-03-01 02:17
---
> **구현 기록**  
> 실험 및 웹페이지 제작에 필요한 대부분의 기능 구현  
> google 애널리스트와 연동 및 좋아요 기능 구현은 추후에 진행  
> Github Actions 스케줄러를 이용한 자동화 시스템 미구현 <- 구현은 했으나 외부 이슈로 후순위  
> 글쓰기 고도화 작업 로그는 다른 페이지에서 기술


자랑하기에 앞서, 해당 프로젝트를 위해 나는 github 두 개를 팠다. imwriteri는 웹페이지를 보여주기 위한 용도이기 때문에 public이지만 imeverything은 실험을 위한 용도라 private. 후자의 이름을 이렇게 정한 이유는 언젠가 글쓰기 말고 다른 것도 해보고 싶기 때문이다. 어쨌든, 앞으로 웹페이지를 변경하는 경우에는 imwriteri를 변경한 것이고, 글쓰기 및 자동화와 관련된 부분은 전부 imeverything에서 일어났다는 것! 


## IMWRTIERI 웹페이지
완성된 웹페이지를 만드는 것은 얼마나 어려운가. 그러나 나는 github와 사랑스러운 codex를 이용하여 아주 빠르게 웹페이지의 개괄을 만들었다. 그리고 나서 여러 가지 수정사항을 거쳐 만들어진 홈페이지의 대문은 다음과 같다.
### Home

![](https://blog.kakaocdn.net/dna/ceKeHq/dJMcabQFnCl/AAAAAAAAAAAAAAAAAAAAADVxhXD1bs8lMIYx0bJlrcUMs_qQrKxVDfEzcfYCFLqa/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1774969199&allow_ip=&allow_referer=&signature=GFk7L3W4s2P7DJRuoOa5fiNxELk%3D)

당연히 차차 변경되겠지만 일단 각 페이지를 설명해보겠다.
Home은 대문. scroll을 이용해 아래로 내리면 최신 순으로 발행한 글이 나타난다.

### Content
![](https://blog.kakaocdn.net/dna/cky6I2/dJMcabQFnD5/AAAAAAAAAAAAAAAAAAAAAJK3gHPWDNGMtqlP8usTL98ktBaP40033JISHDI_Zf7c/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1774969199&allow_ip=&allow_referer=&signature=ZzBi5vZj8YjFiKA1PKWR%2BaO1mho%3D)
모든 글을 볼 수 있는 곳이다. 마찬가지로 최신 순으로 정렬되며, 제목 아래에는 어떤 단어를 기준으로 작성했는지 볼 수 있다.
search by word는 모든 단어들이 아닌, AI가 글을 쓰기 위해 던져진 '주제 단어'만을 기준으로 찾을 수 있다. 

### What words?
![](https://blog.kakaocdn.net/dna/c3DazG/dJMcagxFmSe/AAAAAAAAAAAAAAAAAAAAAFpKG5zsBLDT64zFqoRtFwYaLCDpLmJQunMW3I4QbXoa/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1774969199&allow_ip=&allow_referer=&signature=%2BI9zevlUd7wOXYPNtSQ7dV77v%2Fw%3D)
내가 무조건 구현해야겠다고 생각한 페이지! 그건 바로 단어들을 눌러서 이에 맞는 글을 찾는 것이다. 이렇게 사용된 단어가 정렬되는 것을 너무 보고 싶었다. 궁금하지 않은가? 대체 정밀추척이라는 단어를 이용해서 무슨 글을 썼을까. 어쩌면 내가 글을 쓸 때 이것들 중에서 찾아볼 수도 있을 거다.
사실 내가 글 쓸 때 도움이 될 만한 기능을 많이 구현해보고 싶었는데... 일단 프로젝트의 메인에 집중하기로 했다. 아직 갈 길이 머니까요~ 느리다처럼 두 번 사용된 건 숫자로 표시되게 해놨다.

###  About
![](https://blog.kakaocdn.net/dna/IseDR/dJMcaada48w/AAAAAAAAAAAAAAAAAAAAAJAO5OE0qmbZodet3zwcFDtgfzyNOVViuCX7VLfSFNp9/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1774969199&allow_ip=&allow_referer=&signature=%2Bt%2FLshk2F0VkibhHsFk30LyGdCc%3D)

About은 말 그대로 About.
아직은 기능을 구현하느라 바빠서 어떤 식으로 작성할 지 깊게 생각하지는 못했다. X를 쓰게 된 이유는 자잘한 것들을 기록하기 위해서. 그리고 많은 이들의 도움을 받아보고 싶다는 약간의 욕심도 들어가 있다. 예를 들어 이 시점(2026.02.15)에서 AI는 랜덤 조건으로 글을 생성한다. 그럼 이 조건을 어떻게 세분화하면 AI가 더 잘 글을 쓸 수 있을까? 나 혼자 글을 읽고 생각해도 좋지만 혹시 의견을 얻을 수 있을까 하여...ㅎㅎ 아직 제대로 활성화는 하지 않았다. 

## **구현기록**
구현기록이라 하고 미구현기록이라고 부른다.
본래 하고자 했던 것은 **아침 6시 30분에 자동으로 글을 써서 발행하는 자동화 시스템**이었다.
그래서 파이프라인을 단어/장르/태그 샘플링 -> 생성 -> 품질 검수 -> md 생성 -> git push로 만들어놨고 Github Action 스케줄러를 사용해서 run_daily.py를 구현했다. 
그런데 하나 문제가 있었으니... Local LLM으로 글 쓰려면 내 컴퓨터가 켜져 있어야 한다. 당연함... 
방법은

_1) 24시간 켜져 있는 서버를 이용하거나_

_2) 돈을 내고 클라우드 서비스를 이용하거나_

_3) 돈을 내고 API를 쓰거나_

_4) 내 소중한 노트북을 24시간 가동하거나_

절망적인 상황이 아닐 수 없다.
그래서 실행은 내가 손으로 직접 실행을 시켜줘야 한다. 이 부분은 구현의 문제가 아니라 나의 재정 및 상황의 문제니 일단은 패스하기로 했다. 언젠가 시스템이 안정화된다면 해보는 것으로. 어차피 실험하려면 하루 하나의 글로는 안되기 때문에 큰 상관은 없다. 그냥 나의 꿈 하나가 살짝 뒤로 밀린 것 뿐이지...


웹페이지 : [https://skymined.github.io/imwriteri/](https://skymined.github.io/imwriteri/)