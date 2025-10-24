import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import init_model, load_sheet, generate_report, handle_follow_up

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="SOPL", layout="wide")
st.title("소상공인과 함께하는 AI 마케터 SOPL입니다")
st.caption("AI와 데이터를 활용하여 마케팅 방법을 제시합니다")

with st.sidebar:
    st.header("보안 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    sheet_url = st.text_input("Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/...")
    st.markdown("---")

    st.subheader("분석 질문 입력")
    question = st.text_area(
        "기본 분석 질문",
        placeholder="예: 김밥천국 101번점의 매출 증대 전략은?",
        height=100,
    )

    if st.button("기본 분석 실행"):
        if not api_key or not sheet_url:
            st.error("API Key와 Google Sheet URL을 모두 입력해주세요.")
        else:
            with st.spinner("분석 중입니다..."):
                chat_model = init_model(api_key)
                df = load_sheet(sheet_url)
                if df is not None:
                    result = generate_report(chat_model, df, question)
                    st.session_state["report"] = result["report"]
                    st.session_state["comparison_df"] = result["comparison_df"]
                    st.session_state["summary_text"] = result["summary_text"]
                    st.session_state["df"] = df
                    st.session_state["chat_model"] = chat_model
                    st.success("분석 완료")
                else:
                    st.error("Google Sheet를 불러오지 못했습니다.")

tab1, tab2, tab3 = st.tabs(["메인 보고서", "상관관계 히트맵", "업체 비교 분석"])

# 메인 보고서
with tab1:
    st.header("SOPL 마케팅 방법 추천")
    if "report" in st.session_state:
        st.markdown(st.session_state["report"])
        st.markdown("---")
        st.subheader("추가 질문 또는 이용 목적 입력")
        follow_up_text = st.text_area("추가로 알고 싶은 점을 입력하세요.", height=80)
        if st.button("추가 분석 요청"):
            if follow_up_text.strip() == "":
                st.warning("추가 질문을 입력해주세요.")
            else:
                chat_model = st.session_state.get("chat_model")
                report = st.session_state.get("report")
                with st.spinner("추가 분석 중..."):
                    follow_result = handle_follow_up(chat_model, report, follow_up_text)
                    st.session_state["follow_up"] = follow_result
                    st.success("추가 분석 완료.")
        if "follow_up" in st.session_state:
            st.markdown("### 추가 분석 결과")
            st.markdown(st.session_state["follow_up"])
    else:
        st.info("왼쪽에서 API Key와 Sheet URL을 입력하고 분석을 시작하세요.")

# 상관관계 히트맵
with tab2:
    st.header("상관분석 히트맵")
    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_df = df.select_dtypes(include=[float, int])
        if numeric_df.empty:
            st.warning("숫자형 데이터가 없습니다.")
        else:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="RdBu_r", linewidths=0.5, ax=ax)
            ax.set_title("주요 변수 간 상관관계")
            st.pyplot(fig)
    else:
        st.info("먼저 데이터를 불러오고 분석을 실행해주세요.")

# 업체 비교 분석
with tab3:
    st.header("업체 비교 분석 결과")
    if "comparison_df" in st.session_state and st.session_state["comparison_df"] is not None:
        df_compare = st.session_state["comparison_df"]
        group_df = pd.DataFrame([
            {"그룹": g, "대표지표": v["대표칼럼"], "평균편차": v["편차평균"]}
            for g, v in df_compare.items()
        ])

        st.subheader("그룹별 업소-업종 평균 편차")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=group_df, x="그룹", y="평균편차", palette="coolwarm", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("편차 (업체 - 업종)")
        ax.set_title("그룹별 업소 vs 업종 평균 비교")
        st.pyplot(fig)
        st.dataframe(group_df)
        st.markdown(st.session_state["summary_text"])
    else:
        st.info("점포 비교 분석 결과가 없습니다.")
