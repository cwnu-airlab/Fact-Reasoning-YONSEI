# IITP Fact Reasoning API 정의서

본 문서는 '비정형 텍스트를 학습하여 쟁점별 사실과 논리적 근거 추론이 가능한 인공지능 원천기술' 과제 관련 각 모듈 API 연동에 관한 내용을 정의한다.

아래 모듈들은 모두 POST 방식으로 동작한다.

## 06-KnowledgeMerging-YONSEI

* 웹 API 정보: 

* 입력 파라미터
  * 04-RelationExtractoin의 출력과 동일

| Key         | Value                    | Explanation                                         |
| ----------- | ------------------------ | --------------------------------------------------- |
| question    | dict                     | (required) 질문                                     |
| ㄴ text     | str                      | 질문 문장                                           |
| ㄴ language | str                      | 질문 문장의 언어                                    |
| ㄴ domain   | str                      | 질문의 분야                                         |
| triples     | list[tuple[str,str,str]] | (required) 문서에서 추출된 Arg0, Arg1 트리플 리스트 |

* 예시

```json
{
  "question":{
    "text":"데드풀 감독이랑 킬러의 보디가드 감독이 같은 사람이야?",
    "language":"kr",
    "domain":"common-sense"
  },
  "triples":[
    ["팀 밀러","데드풀","감독"],
    ["패트릭 휴스","킬러의 보디가드","감독"],
    ["패트릭 휴스","영화 감독","직업"]
    ]
}
```
