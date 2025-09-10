# pages/2_클러스터링.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from lib import find_latest_csv, smart_read_csv, normalize_and_parse, to_numeric_strict

st.set_page_config(page_title="🌈 K-means 클러스터링", layout="wide")
st.title("🌈 K-means 비지도 계절 구분 — 최저/최고기온 기반")

with st.sidebar:
    st.header("데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7)
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"])
    use_complete = st.checkbox("완전한 연도만 사용", value=True)
    scaler_opt = st.selectbox("스케일링", ["표준화(Standard)", "MinMax", "없음"], index=0)
    k_clusters = st.slider("클러스터 수 K", 3, 6, 4, step=1)

# 데이터 로딩
if src == "CSV 업로드":
    up = st.file_uploader("CSV 업로드", type=["csv"])
    if up is None: st.stop()
    df_raw = smart_read_csv(up, skip_top_rows=skip_n); loaded = "(uploaded)"
else:
    latest = find_latest_csv(("data","."))
    if latest is None: st.error("기본 CSV 없음"); st.stop()
    df_raw = smart_read_csv(latest, skip_top_rows=skip_n); loaded = latest

df = normalize_and_parse(df_raw)
st.caption(f"소스: {loaded}")
if "date" not in df.columns: st.error("'날짜' 또는 'date' 필요"); st.stop()

# tmin/tmax 선택
cols = df.columns.tolist()
auto_tmin = "최저기온(℃)" if "최저기온(℃)" in df.columns else None
auto_tmax = "최고기온(℃)" if "최고기온(℃)" in df.columns else None
c1, c2 = st.columns(2)
with c1:
    tmin_col = st.selectbox("최저기온 컬럼 (tmin)", options=[None] + cols,
                            index=(cols.index(auto_tmin)+1 if auto_tmin in cols else 0))
with c2:
    tmax_col = st.selectbox("최고기온 컬럼 (tmax)", options=[None] + cols,
                            index=(cols.index(auto_tmax)+1 if auto_tmax in cols else 0))

if not tmin_col or not tmax_col:
    st.info("tmin/tmax 컬럼을 선택하세요.")
    st.stop()

dfK = df.copy()
dfK["year"] = dfK["date"].dt.year
dfK[tmin_col] = to_numeric_strict(dfK[tmin_col])
dfK[tmax_col] = to_numeric_strict(dfK[tmax_col])
dfK = dfK.dropna(subset=[tmin_col, tmax_col, "date"]).copy()

# 완전 연도 필터
if use_complete:
    max_dt = dfK["date"].dropna().max()
    if pd.notna(max_dt):
        last_complete = max_dt.year if max_dt >= pd.Timestamp(max_dt.year, 12, 31) else (max_dt.year - 1)
        dfK = dfK[dfK["year"] <= last_complete]

if dfK.empty:
    st.info("클러스터링에 사용할 데이터가 없습니다.")
    st.stop()

# 스케일링 + KMeans
X = dfK[[tmin_col, tmax_col]].to_numpy()
if scaler_opt == "표준화(Standard)":
    X = StandardScaler().fit_transform(X)
elif scaler_opt == "MinMax":
    X = MinMaxScaler().fit_transform(X)

km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
dfK["cluster"] = km.fit_predict(X)

# 계절 라벨 매핑(평균기온 낮→높)
dfK["temp_mean"] = (dfK[tmin_col] + dfK[tmax_col]) / 2
cluster_means = dfK.groupby("cluster")["temp_mean"].mean().sort_values().reset_index()
if k_clusters == 4:
    season_names = ["겨울", "봄", "가을", "여름"]
    season_map = {row.cluster: season_names[i] for i, row in enumerate(cluster_means.itertuples(index=False, name="Row"))}
else:
    mids = [f"중간{i+1}" for i in range(k_clusters-2)]
    names = ["추움"] + mids + ["더움"]
    season_map = {row.cluster: names[i] for i, row in enumerate(cluster_means.itertuples(index=False, name="Row"))}
dfK["season_unsup"] = dfK["cluster"].map(season_map)

# ① 산점도
st.subheader("① tmin–tmax 산점도(클러스터 색)")
st.altair_chart(
    alt.Chart(dfK).mark_point(filled=True, opacity=0.6).encode(
        x=alt.X(f"{tmin_col}:Q", title=tmin_col),
        y=alt.Y(f"{tmax_col}:Q", title=tmax_col),
        color=alt.Color("season_unsup:N"),
        tooltip=["date:T", f"{tmin_col}:Q", f"{tmax_col}:Q", "season_unsup:N"]
    ).properties(height=360),
    use_container_width=True
)

# ② 타임라인
st.subheader("② 날짜 타임라인(계절 색)")
st.altair_chart(
    alt.Chart(dfK).mark_bar(height=8).encode(
        x=alt.X("date:T", title="날짜"),
        y=alt.value(10),
        color=alt.Color("season_unsup:N"),
        tooltip=["date:T", "season_unsup:N"]
    ).properties(height=80),
    use_container_width=True
)

# ③ 연도별 계절 일수 & 계절 길이 추세(선택)
st.subheader("③ 연도별 계절 일수 / 계절 길이 추세")
dfKc = dfK.copy()
season_order = (dfKc.groupby("season_unsup")["temp_mean"].mean().sort_values().reset_index()["season_unsup"].tolist())
hottest_label = season_order[-1]  # 가장 더운 라벨(여름/더움 등)

counts = dfKc.groupby(["year", "season_unsup"]).size().reset_index(name="days")
line_all = alt.Chart(counts).mark_line(point=True).encode(
    x=alt.X("year:O", title="연도"),
    y=alt.Y("days:Q", title="일수"),
    color=alt.Color("season_unsup:N", sort=season_order),
    tooltip=["year:O", "season_unsup:N", "days:Q"]
).properties(height=360)
st.altair_chart(line_all, use_container_width=True)

default_idx = season_order.index(hottest_label) if hottest_label in season_order else len(season_order)-1
season_to_view = st.selectbox("추세를 볼 계절", options=season_order, index=default_idx)
sel = counts[counts["season_unsup"] == season_to_view].copy()
if not sel.empty:
    base = alt.Chart(sel).mark_line(point=True).encode(
        x="year:O", y=alt.Y("days:Q", title=f"{season_to_view} 일수")
    ).properties(height=300)
    if len(sel["year"].unique()) >= 3:
        lr = LinearRegression().fit(sel[["year"]].astype(int), sel["days"])
        sel["pred"] = lr.predict(sel[["year"]].astype(int))
        slope = float(lr.coef_[0]); slope_dec = slope * 10
        trend = alt.Chart(sel).mark_line(color="orange").encode(x="year:O", y="pred:Q")
        st.altair_chart(base + trend, use_container_width=True)
        st.metric(f"{season_to_view} 변화(추세 기울기)", f"{slope:+.2f} 일/년  ≈  {slope_dec:+.1f} 일/10년")
    else:
        st.altair_chart(base, use_container_width=True)
else:
    st.info(f"{season_to_view} 데이터 없음")

# ④ 전이 시점 — 가장 더운 계절(여름/더움 등)
st.subheader(f"④ {hottest_label} 전이 시점 (첫/마지막 {hottest_label} 날짜)")
hot_df = dfKc[dfKc["season_unsup"] == hottest_label].copy()
if hot_df.empty:
    st.info(f"{hottest_label} 데이터가 부족합니다.")
else:
    trans = hot_df.groupby("year").agg(
        first_hot=("date", "min"),
        last_hot=("date", "max")
    ).reset_index()
    first_chart = alt.Chart(trans).mark_line(point=True).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("first_hot:T", title=f"첫 {hottest_label} 도달일"),
        tooltip=["year:O", alt.Tooltip("first_hot:T", title=f"첫 {hottest_label}")]
    ).properties(height=200)
    last_chart = alt.Chart(trans).mark_line(point=True, color="red").encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("last_hot:T", title=f"마지막 {hottest_label} 종료일"),
        tooltip=["year:O", alt.Tooltip("last_hot:T", title=f"마지막 {hottest_label}")]
    ).properties(height=200)
    st.altair_chart(first_chart & last_chart, use_container_width=True)

st.markdown("---")
st.caption("교육 메모: K가 달라도 ‘가장 더운 계절’을 자동 인식해 전이 시점을 산출합니다.")
