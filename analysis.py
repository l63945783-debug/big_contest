import os
import re
import json
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------
# 1. Gemini 초기화
# --------------------------
def init_model(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.5
    )


# --------------------------
# 2. Google Sheet CSV 로드
# --------------------------
def load_sheet(sheet_url: str) -> pd.DataFrame:
    try:
        if "/edit?gid=" in sheet_url:
            sheet_url = sheet_url.replace("/edit?gid=", "/edit#gid=")

        if "/edit#gid=" in sheet_url:
            sheet_id = sheet_url.split("/d/")[1].split("/")[0]
            gid = sheet_url.split("gid=")[-1]
            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        else:
            raise ValueError("잘못된 Google Sheet URL 형식입니다. (예: .../edit#gid=0)")

        df = pd.read_csv(export_url)
        print("Google Sheet 데이터 불러오기 성공")
        return df
    except Exception as e:
        print("시트를 불러올 수 없습니다:", e)
        return None


# --------------------------
# 3. 질문 해석
# --------------------------
def interpret_question(chat_model, df: pd.DataFrame, question: str):
    column_list = ", ".join(df.columns)
    prompt = f"""
    다음은 데이터프레임의 칼럼 목록입니다:
    [{column_list}]

    사용자의 질문은 다음과 같습니다:
    "{question}"

    질문에서 다음 항목을 JSON 형태로 추출하세요:
    {{
        "target_column": "분석 기준 칼럼명 (예: 매출금액)",
        "condition": "상위 / 하위 / 없음",
        "percentage": "숫자 (예: 30), 없으면 null",
        "region": "상권명 또는 지역명 (있으면)",
        "industry": "업종명 (있으면)",
        "analysis_goal": "사용자가 궁금해하는 주제 (예: 매출 증대 전략)"
    }}
    """
    response = chat_model.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    try:
        parsed = json.loads(re.search(r"\{.*\}", text, re.S).group())
        return parsed
    except Exception:
        print("Gemini 응답 해석 실패:", text)
        return {}


# --------------------------
# 4. 그룹별 상관분석
# --------------------------
def analyze_groups(df: pd.DataFrame, groups: dict, exclude_col=None):
    results = {}
    for group_name, cols in groups.items():
        available_cols = [c for c in cols if c in df.columns and c != exclude_col]
        if len(available_cols) < 2:
            continue

        sub_df = df[available_cols].select_dtypes(include=[np.number])
        if sub_df.empty:
            continue

        corr_matrix = sub_df.corr()
        corr_mean = corr_matrix.abs().mean().sort_values(ascending=False)
        top_col = corr_mean.index[0]
        top_corr_value = corr_mean.iloc[0]

        results[group_name] = {
            "대표칼럼": top_col,
            "상관계수평균": round(top_corr_value, 3),
            "칼럼목록": available_cols
        }
    return results


# --------------------------
# 5. 그룹별 업종 평균 비교
# --------------------------
def compare_to_industry_grouped(df, store_df, industry_df, column_groups):
    results = {}
    for group_name, cols in column_groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue

        store_mean = store_df[valid_cols].mean(numeric_only=True)
        industry_mean = industry_df[valid_cols].mean(numeric_only=True)

        group_diff = (store_mean - industry_mean).dropna()
        group_avg_diff = group_diff.mean()
        results[group_name] = {
            "대표칼럼": group_diff.abs().idxmax() if not group_diff.empty else None,
            "편차평균": round(group_avg_diff, 2),
            "세부항목": group_diff.to_dict()
        }
    return results


# --------------------------
# 6. 비교 결과 요약
# --------------------------
def summarize_comparison_grouped(results):
    summary_lines = []
    for group, data in results.items():
        if not data or not data["세부항목"]:
            continue
        positives = [k for k, v in data["세부항목"].items() if v > 0]
        negatives = [k for k, v in data["세부항목"].items() if v < 0]
        summary_lines.append(
            f"{group}: 강점({', '.join(positives) if positives else '없음'}) / 약점({', '.join(negatives) if negatives else '없음'})"
        )
    return "\n".join(summary_lines)


# --------------------------
# 7. 보고서 생성
# --------------------------
def generate_report(chat_model, df: pd.DataFrame, question: str):
    parsed = interpret_question(chat_model, df, question)
    target_col = parsed.get("target_column")
    industry = parsed.get("industry")

    column_groups = {
        "매출 관련": ["매출금액", "매출건수", "객단가", "배달매출비율"],
        "업종/상권 관련": ["동일업종평균대비매출비율", "동일업종내매출순위", "동일상권내매출순위"],
        "타겟층 관련": ["20대이하남자", "30대남자", "40대남자", "50대남자", "60대이상남자",
                    "20대이하여자", "30대여자", "40대여자", "50대여자", "60대이상여자"],
        "고객 관련": ["신규고객", "재방문고객"],
        "이용 관련": ["거주이용", "직장이용", "유동인구이용"]
    }

    # 점포 비교 모드 or 상관분석 모드 구분
    store_name = None
    store_code = None
    comparison_results = None

    if "상호명" in df.columns and "가맹점구분번호" in df.columns:
        matched_stores = [name for name in df["상호명"].astype(str) if name in question]
        matched_codes = [str(code) for code in df["가맹점구분번호"].astype(str) if str(code) in question]
        if matched_stores and matched_codes:
            store_name = matched_stores[0]
            store_code = matched_codes[0]
            store_df = df[(df["상호명"] == store_name) & (df["가맹점구분번호"].astype(str) == store_code)]
            if not store_df.empty:
                industry = store_df["업종"].iloc[0] if "업종" in store_df.columns else None
                industry_df = df[df["업종"] == industry]
                comparison_results = compare_to_industry_grouped(df, store_df, industry_df, column_groups)
                summary_text = summarize_comparison_grouped(comparison_results)
            else:
                comparison_results = None
                summary_text = "선택한 상호명과 가맹점번호에 해당하는 데이터가 없습니다."
        else:
            comparison_results = None
            summary_text = "질문에서 상호명 또는 가맹점번호를 찾지 못했습니다."
    else:
        # 일반 상관분석 모드
        results = analyze_groups(df, column_groups, exclude_col=target_col)
        summary_text = "\n".join(
            [f"{g}: 대표칼럼={i['대표칼럼']} / 평균상관={i['상관계수평균']}" for g, i in results.items()]
        )
        comparison_results = None

    # 프롬프트 (변경 금지)
    prompt = f"""
    당신은 상권 데이터와 고객 행동 데이터를 기반으로,
    소상공인에게 실질적인 마케팅 전략을 제시하는 전문가 **SOPL 마케팅 컨설턴트**입니다.

    {'업체명: ' + store_name if store_name else ''}
    분석 대상 업종: {industry or '미확인'}
    ---
    **상관분석 요약**
    {summary_text}
    ---
    **분석 가이드라인**
    1. **강점(상관관계가 높은 요인)** 은 “핵심 성장 포인트”로 정의하고 강화 전략을 제시하세요.
    2. **약점(상관관계가 낮은 요인)** 은 “보완 포인트”로 간주하고 개선 방향을 제시하세요.
    3. 전략 제안 시에는 **STP**, **RFM**, **번들·교차판매 전략(Apriori)** 관점을 반드시 반영하세요.
    4. 제안은 데이터 근거(지표명/비율/비교군 등)를 인용해 근거 기반으로 서술하세요.
    5. 실무자가 바로 활용할 수 있는 구체적 행동(Action Plan) 형태로 작성하세요.
    ---
    **출력 형식**
    - [1] 요약: 분석의 전체 방향과 핵심 발견 요약 (2~3문장) 
    - [2] 주요 인사이트: 상관관계 기반으로 파악된 강점/약점 및 시사점 (데이터 근거 포함) 
    - [3] 마케팅 전략 제안: STP, RFM, 탄력성, 번들/교차판매 관점에서 세분화된 전략 제시 - 각 전략은 “무엇을 / 누구에게 / 언제 / 어떻게” 구조로 작성 
    - [4] SNS 홍보문구: 분석 인사이트를 반영한 짧은 홍보 카피 1~2문장 
    - 톤은 실무형, 긍정적으로 작성 
    - 여기서 요청한 고객의 가게를 홍보하는 문구이므로 지표나 SOPL을 보이지 말고, 위에 강점을 살려서 작성 
    --- 
    **주의사항** 
    - 글머리표 대신 짧은 문단 중심으로 작성
    - 수치, 비율, 지표명 등은 그대로 인용 
    - 과도한 마케팅 수사는 피하고, 분석 중심으로 구체적 작성
    - 가상데이터를 만들어 내지말고 상관관계 계수만 활용하여 작성
    """
    res = chat_model.invoke([HumanMessage(content=prompt)])
    return {
        "report": res.content.strip(),
        "comparison_df": comparison_results,
        "summary_text": summary_text
    }


# --------------------------
# 8. 추가 요청 처리
# --------------------------
def handle_follow_up(chat_model, previous_report: str, follow_up: str):
    relevance_prompt = f"""
    다음 추가 요청이 이전 보고서 내용과 관련이 있는지 판별해줘.
    이전 보고서: {previous_report}
    추가 요청: "{follow_up}"
    관련이 있으면 "related", 아니면 "new"라고만 답해.
    """
    result = chat_model.invoke([HumanMessage(content=relevance_prompt)]).content.strip().lower()

    if "related" in result:
        prompt = f"""
        이전 보고서:
        {previous_report}

        추가 요청:
        "{follow_up}"

        관련 내용을 보완 또는 확장한 새로운 섹션을 작성해줘.
        """
    else:
        prompt = f"""
        새로운 요청:
        "{follow_up}"

        독립적 보고서를 작성:
        1. 요약
        2. 주요 인사이트
        3. 전략 제안
        """

    res = chat_model.invoke([HumanMessage(content=prompt)])
    return res.content.strip()
