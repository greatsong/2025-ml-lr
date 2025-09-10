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
st.title("🌈 K-means 비지도 계절 구분 — 사후 규칙(봄/가을)")

# ===== 계절 색상 팔레트 =====
SEASON_COLORS = {
    "봄": "#FFD700",   # 노랑(개나리)
    "여름": "#FF0000", # 빨강
    "가을": "#D2691E", # 단풍색
    "겨울": "#1E90FF"  # 하늘색
}
def season_color_encoding(present_labels):
    domain = [s for s in ["봄","여름","가을","겨울"] if s in present_labels]
    if not domain:
        return alt.Color("season_unsup:N")
    rng = [SEASON_COLORS[d] for d in domain]
    return alt.Color("season_unsup:N", scale=alt.Scale(domain=domain, range=rng))

# ===== 사이드바 / 데이터 로딩 =====
with st.sidebar:
    st.header("데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7)
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"])
    use_complete = st.checkbox("완전한 연도만 사용", value=True)
    scaler_opt = st.selectbox("스케일링", ["표준화(Standard)", "MinMax", "없음"], index=0)
    k_clusters = st.slider("클러스터 수 K", 3, 6, 4, step=1)
    st.markdown("---")
    # 사후 규칙 경계일: 이 날 이전은 상반기(봄), 이후는 하반기(가을)
    bound_doy = st.slider("봄/가을 경계(연중일수, 기본 183≈7/1)", min_value=150, max_value=220, value=183)

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
if "date" not in df.columns:
    st.error("'날짜' 또는 'date' 컬럼이 필요합니다."); st.stop()

# ===== 피처 선택/전처리 =====
cols = df.columns.tolist()
auto_tmin = "최저기온(℃)" if "최저기온(℃)" in df.columns else None
auto_tmax = "최고기온(℃)" if "최고기온(℃)" in df.columns else None
c1, c2 = st.columns(2)
with c1:
    tmin_col = st.selectbox("최저기온 컬럼 (tmin)", options=[None]+cols,
                            index=(cols.index(auto_tmin)+1 if auto_tmin in cols else 0))
with c2:
    tmax_col = st.selectbox("최고기온 컬럼 (tmax)", options=[None]+cols,
                            index=(cols.index(auto_tmax)+1 if auto_tmax in cols else 0))
if not tmin_col or not tmax_col:
    st.info("tmin/tmax 컬럼을 선택하세요."); st.stop()

dfK = df.copy()
dfK["year"]  = dfK["date"].dt.year
dfK["month"] = dfK["date"].dt.month
dfK["doy"]   = dfK["date"].dt.dayofyear
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
    st.info("클러스터링에 사용할 데이터가 없습니다."); st.stop()

# ===== K-means (tmin, tmax만 사용) =====
feature_cols = [tmin_col, tmax_col]
X = dfK[feature_cols].to_numpy()
if scaler_opt == "표준화(Standard)":
    X = StandardScaler().fit_transform(X)
elif scaler_opt == "MinMax":
    X = MinMaxScaler().fit_transform(X)

km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
dfK["cluster"] = km.fit_predict(X)

# 클러스터 요약
dfK["temp_mean"] = (dfK[tmin_col] + dfK[tmax_col]) / 2
centers = dfK.groupby("cluster")[[tmin_col, tmax_col, "temp_mean"]].mean().reset_index()
st.subheader("클러스터 요약(평균)")
st.dataframe(centers.round(2))

# ===== 계절 라벨 매핑(기본: 평균기온 낮→높) + 사후 규칙로 봄/가을 분리 =====
cluster_order = centers.sort_values("temp_mean")["cluster"].tolist()
if k_clusters == 4:
    base_names = ["겨울", "봄", "가을", "여름"]  # 초기 이름
    season_map = {cl: base_names[i] for i, cl in enumerate(cluster_order)}
    dfK["season_unsup"] = dfK["cluster"].map(season_map)

    # 두 중간온도 클러스터(봄/가을 후보)를 경계일로 분할
    mid_clusters = cluster_order[1:3]
    mask_mid = dfK["cluster"].isin(mid_clusters)
    dfK.loc[mask_mid & (dfK["doy"] <  bound_doy), "season_unsup"] = "봄"
    dfK.loc[mask_mid & (dfK["doy"] >= bound_doy), "season_unsup"] = "가을"
else:
    # K!=4: 설명적 라벨로 유지
    mids = [f"중간{i+1}" for i in range(max(k_clusters-2, 0))]
    dyn = ["추움"] + mids + ["더움"]
    season_map = {cl: dyn[i] for i, cl in enumerate(cluster_order)}
    dfK["season_unsup"] = dfK["cluster"].map(season_map)

# 가장 더운/추운 라벨 (시각화/전이 시점용)
season_order = (dfK.groupby("season_unsup")["temp_mean"].mean()
                .sort_values().reset_index()["season_unsup"].tolist())
hottest_label = season_order[-1]
coldest_label = season_order[0]

# ===== ① tmin–tmax 산점도 (계절 색) =====
st.subheader("① tmin–tmax 산점도 (계절 색)")
present_labels = dfK["season_unsup"].dropna().unique().tolist()
color_enc = season_color_encoding(present_labels)
scatter = alt.Chart(dfK).mark_point(filled=True, opacity=0.6).encode(
    x=alt.X(f"{tmin_col}:Q", title=tmin_col),
    y=alt.Y(f"{tmax_col}:Q", title=tmax_col),
    color=color_enc,
    tooltip=["date:T", f"{tmin_col}:Q", f"{tmax_col}:Q", "season_unsup:N"]
).properties(height=360)
st.altair_chart(scatter, use_container_width=True)

# ===== ② 연도별 계절 일수 추세 =====
st.subheader("② 연도별 계절 일수 추세")
counts = dfK.groupby(["year","season_unsup"]).size().reset_index(name="days")
present_labels2 = counts["season_unsup"].dropna().unique().tolist()
color_enc2 = season_color_encoding(present_labels2)
line_all = alt.Chart(counts).mark_line(point=True).encode(
    x=alt.X("year:O", title="연도"),
    y=alt.Y("days:Q", title="일수"),
    color=color_enc2,
    tooltip=["year:O", "season_unsup:N", "days:Q"]
).properties(height=360)
st.altair_chart(line_all, use_container_width=True)

# ===== ③ (선택 계절) 길이 추세 =====
st.subheader("③ (선택 계절) 길이 추세")
default_idx = season_order.index(hottest_label) if hottest_label in season_order else len(season_order)-1
season_to_view = st.selectbox("추세를 볼 계절", options=season_order, index=default_idx)
sel = counts[counts["season_unsup"] == season_to_view].copy()
if not sel.empty:
    base = alt.Chart(sel).mark_line(point=True).encode(
        x="year:O",
        y=alt.Y("days:Q", title=f"{season_to_view} 일수"),
        color=alt.value(SEASON_COLORS.get(season_to_view, None)),
        tooltip=["year:O", "days:Q"]
    ).properties(height=300)
    if sel["year"].nunique() >= 3:
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

# ===== ④ {가장 더운 계절} 전이 시점 — 첫/마지막 (연중일수) =====
st.subheader(f"④ {hottest_label} 전이 시점 — 첫/마지막 (연중일수)")
hot_df = dfK[dfK["season_unsup"] == hottest_label].copy()
if hot_df.empty:
    st.info(f"{hottest_label} 데이터가 부족합니다.")
else:
    trans = (hot_df.groupby("year")
                    .agg(first_hot=("date","min"), last_hot=("date","max"))
                    .reset_index())
    trans["first_hot_doy"] = trans["first_hot"].dt.dayofyear
    trans["last_hot_doy"]  = trans["last_hot"].dt.dayofyear
    trans["first_hot_label"] = trans["first_hot"].dt.strftime("%m-%d")
    trans["last_hot_label"]  = trans["last_hot"].dt.strftime("%m-%d")
    trans["hot_span_days"] = (trans["last_hot_doy"] - trans["first_hot_doy"] + 1).clip(lower=0)

    # 연속 축용 빈 연도 채우기
    all_years = pd.DataFrame({"year": sorted(dfK["year"].unique())})
    trans = all_years.merge(trans, on="year", how="left")

    hot_color = SEASON_COLORS.get(hottest_label, None)
    ch_first = alt.Chart(trans).mark_line(point=True, color=hot_color).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("first_hot_doy:Q", title=f"첫 {hottest_label} (연중 일수)"),
        tooltip=["year:O",
                 alt.Tooltip("first_hot_label:N", title=f"첫 {hottest_label}"),
                 alt.Tooltip("first_hot_doy:Q", title="연중일수")]
    ).properties(height=220, title=f"첫 {hottest_label} 도달 — 낮을수록 빨라짐")

    ch_last = alt.Chart(trans).mark_line(point=True, color=hot_color).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("last_hot_doy:Q", title=f"마지막 {hottest_label} (연중 일수)"),
        tooltip=["year:O",
                 alt.Tooltip("last_hot_label:N", title=f"마지막 {hottest_label}"),
                 alt.Tooltip("last_hot_doy:Q", title="연중일수")]
    ).properties(height=220, title=f"마지막 {hottest_label} 종료 — 높을수록 늦어짐")

    # 간단 추세선(오렌지)
    layers_first = [ch_first]
    df_fit_first = trans.dropna(subset=["first_hot_doy"])
    if df_fit_first["year"].nunique() >= 3:
        lr_f = LinearRegression().fit(df_fit_first[["year"]].astype(int), df_fit_first["first_hot_doy"])
        df_fit_first["pred"] = lr_f.predict(df_fit_first[["year"]].astype(int))
        layers_first.append(alt.Chart(df_fit_first).mark_line(color="orange").encode(x="year:O", y="pred:Q"))

    layers_last = [ch_last]
    df_fit_last = trans.dropna(subset=["last_hot_doy"])
    if df_fit_last["year"].nunique() >= 3:
        lr_l = LinearRegression().fit(df_fit_last[["year"]].astype(int), df_fit_last["last_hot_doy"])
        df_fit_last["pred"] = lr_l.predict(df_fit_last[["year"]].astype(int))
        layers_last.append(alt.Chart(df_fit_last).mark_line(color="orange").encode(x="year:O", y="pred:Q"))

    st.altair_chart(alt.layer(*layers_first), use_container_width=True)
    st.altair_chart(alt.layer(*layers_last), use_container_width=True)

# ===== ⑤ 연-월 히트맵 =====
st.subheader("⑤ 연-월 히트맵 (평균기온)")
hm_metric = [c for c in ["평균기온(℃)", "최고기온(℃)", "최저기온(℃)"] if c in df.columns]
hm_metric = hm_metric[0] if hm_metric else df.select_dtypes(include=np.number).columns.tolist()[0]
hm = df[["date", hm_metric]].dropna().copy()
if not hm.empty:
    hm["year"] = hm["date"].dt.year.astype(int)
    hm["month"] = hm["date"].dt.month.astype(int)
    g = hm.groupby(["year","month"])[hm_metric].mean().reset_index(name="val")
    heat = alt.Chart(g).mark_rect().encode(
        x=alt.X("month:O", title="월"),
        y=alt.Y("year:O", title="연도"),
        color=alt.Color("val:Q", title=hm_metric, scale=alt.Scale(scheme="turbo")),
        tooltip=["year:O", "month:O", alt.Tooltip("val:Q", format=".2f")]
    ).properties(height=20*len(g['year'].unique()), width=600)
    st.altair_chart(heat, use_container_width=True)

st.markdown("---")
with st.expander("교육 메모", expanded=False):
    st.markdown(f"""
- **사후 규칙**: K-means는 (tmin, tmax)만 사용하고, 중간 온도대(봄·가을 후보)를 **연중일수 경계({bound_doy}일)** 기준으로 앞=봄, 뒤=가을로 나눕니다.
- **색상 팔레트**: 봄=노랑, 여름=빨강, 가을=단풍색, 겨울=하늘색.
- **전이 시점(DOY)**: 연중일수로 표현해 연도 효과를 제거, “얼마나 빨라졌나/늦어졌나”를 직관적으로 비교합니다.
""")
