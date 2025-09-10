# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from lib import find_latest_csv, smart_read_csv, normalize_and_parse, to_numeric_strict

st.set_page_config(page_title="🌡️ 기온 대시보드 — 메인", layout="wide")

st.title("🌡️ 기온 대시보드 — 메인(설명 + 시각화)")
st.markdown("""
이 앱은 **교육용**으로 설계된 기온 분석/모델링 대시보드입니다.
- 왼쪽 사이드바에서 **데이터 로딩 옵션**을 설정하세요.
- 상단 7행 스킵, **cp949 우선 인코딩**을 자동 적용합니다.
- 하단 메뉴에서 ▶ **회귀**, ▶ **클러스터링** 페이지로 이동해 심화 기능을 확인하세요.
""")

# ---- 사이드바: 데이터 로딩 ----
with st.sidebar:
    st.header("⚙️ 데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7)
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"], horizontal=False)
    st.caption("인코딩: cp949 → utf-8-sig → utf-8 → euc-kr 순 자동 시도")

# ---- 데이터 불러오기 ----
if src == "CSV 업로드":
    up = st.file_uploader("CSV 업로드 (권장: cp949)", type=["csv"])
    if up is None:
        st.info("CSV 업로드 또는 좌측에서 '기본'을 선택하세요.")
        st.stop()
    df_raw = smart_read_csv(up, skip_top_rows=skip_n)
    loaded_from = "(uploaded)"
else:
    latest = find_latest_csv(("data", "."))
    if latest is None:
        st.warning("기본 CSV를 찾지 못했습니다. 업로드를 이용하세요.")
        st.stop()
    df_raw = smart_read_csv(latest, skip_top_rows=skip_n)
    loaded_from = latest

df = normalize_and_parse(df_raw)
st.success(f"데이터 소스: **{loaded_from}**")

if "date" not in df.columns:
    st.error("데이터에 '날짜' 또는 'date' 컬럼이 필요합니다.")
    st.dataframe(df.head())
    st.stop()

# ---- 요약 ----
st.subheader("데이터 요약")
c1, c2, c3 = st.columns(3)
with c1: st.metric("행(일 수)", f"{len(df):,}")
with c2: st.metric("열(특성 수)", f"{df.shape[1]:,}")
with c3: st.metric("결측 총합", f"{int(df.isna().sum().sum()):,}")

with st.expander("데이터 타입 / 결측치 요약", expanded=False):
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))
    miss = df.isna().sum()
    st.dataframe(miss[miss > 0].to_frame("missing_count") if (miss > 0).any()
                 else pd.DataFrame({"message": ["결측치 없음 ✅"]}))

# ---- EDA: 라인/히스토/월별 박스/연-월 히트맵 ----
num_cols = df.select_dtypes(include=np.number).columns.tolist()
pref_cols = [c for c in ["평균기온(℃)", "최고기온(℃)", "최저기온(℃)"] if c in df.columns]
default_show = pref_cols[:2] or num_cols[:2]

st.subheader("① 라인 차트(일 단위)")
sel_cols = st.multiselect("표시할 기온(숫자형) 컬럼", options=num_cols, default=default_show)
if sel_cols:
    melt = df[["date"] + sel_cols].melt("date", var_name="metric", value_name="value")
    st.altair_chart(
        alt.Chart(melt).mark_line().encode(
            x="date:T", y="value:Q", color="metric:N",
            tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=".2f")]
        ).properties(height=320),
        use_container_width=True
    )

st.subheader("② 전체 히스토그램")
hist_metric = st.selectbox("히스토그램 대상", options=(pref_cols or num_cols))
if hist_metric:
    bins = st.slider("bins", 10, 80, 40)
    st.altair_chart(
        alt.Chart(df.dropna(subset=[hist_metric])).mark_bar().encode(
            x=alt.X(f"{hist_metric}:Q", bin=alt.Bin(maxbins=bins), title=hist_metric),
            y="count()",
            tooltip=[alt.Tooltip(f"{hist_metric}:Q", format=".2f"), "count()"]
        ).properties(height=300),
        use_container_width=True
    )

st.subheader("③ 월별 박스플랏 — 모든 연도 합산")
d2 = df.copy(); d2["month"] = d2["date"].dt.month
box_metric = st.selectbox("지표 선택", options=(pref_cols or num_cols), index=0, key="box_metric_all")
select_mode = st.checkbox("월 선택 모드", value=False)
months = list(range(1, 13))
months_sel = months if not select_mode else st.multiselect("표시할 월(1~12)", options=months, default=[1,7,12])
if box_metric:
    sub = d2[(d2["month"].isin(months_sel)) & (~d2[box_metric].isna())]
    if not sub.empty:
        st.altair_chart(
            alt.Chart(sub).mark_boxplot(size=25).encode(
                x=alt.X("month:O", title="월"), y=alt.Y(f"{box_metric}:Q", title=box_metric)
            ).properties(height=320),
            use_container_width=True
        )

st.subheader("④ 연-월 히트맵")
hm_metric = st.selectbox("히트맵 지표", options=(pref_cols or num_cols), index=0)
hm = df[["date", hm_metric]].dropna().copy()
if not hm.empty:
    hm["year"] = hm["date"].dt.year.astype(int)
    hm["month"] = hm["date"].dt.month.astype(int)
    g = (hm.groupby(["year","month"])[hm_metric].mean().reset_index(name="val"))
    st.altair_chart(
        alt.Chart(g).mark_rect().encode(
            x=alt.X("month:O", title="월"),
            y=alt.Y("year:O", title="연도"),
            color=alt.Color("val:Q", title=hm_metric, scale=alt.Scale(scheme="turbo")),
            tooltip=["year:O", "month:O", alt.Tooltip("val:Q", format=".2f")]
        ).properties(height=20*len(g['year'].unique()), width=600),
        use_container_width=True
    )

st.info("➡️ 상단 메뉴의 **회귀**, **클러스터링** 페이지에서 심화 기능을 사용하세요.")
