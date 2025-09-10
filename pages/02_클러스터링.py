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

# =====================================
# 계절 색상 팔레트 (요청 반영)
# 봄: 노랑(개나리), 여름: 빨강, 가을: 단풍색, 겨울: 하늘색
# =====================================
SEASON_COLORS = {
    "봄":   "#FFD700",  # 노랑(개나리)
    "여름": "#FF0000",  # 빨강
    "가을": "#D2691E",  # 단풍색(초콜릿톤)
    "겨울": "#1E90FF",  # 하늘색(겨울왕국)
}
def season_color_encoding(present_labels):
    """present_labels(list[str]) 중 사전에 있는 계절만 색 지정. (그 외 라벨은 Altair 기본색)"""
    domain = [s for s in ["겨울","봄","가을","여름"] if s in present_labels]
    if not domain:
        return alt.Color("season_unsup:N")  # 지정 색 없음 → 기본
    rng = [SEASON_COLORS[d] for d in domain]
    return alt.Color("season_unsup:N", scale=alt.Scale(domain=domain, range=rng))

# =========================
# 사이드바 & 데이터 로딩
# =========================
with st.sidebar:
    st.header("데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7)
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"])
    use_complete = st.checkbox("완전한 연도만 사용", value=True)
    scaler_opt = st.selectbox("스케일링", ["표준화(Standard)", "MinMax", "없음"], index=0)
    k_clusters = st.slider("클러스터 수 K", 3, 6, 4, step=1)

    st.markdown("---")
    season_split_method = st.radio(
        "봄/가을 구분 기준",
        options=["주기 인코딩(DOY)", "사후 규칙(연도 내 순서)"],
        index=0,
        help=(
            "• DOY: 연중일수(1~365)를 sin/cos로 변환해 계절성을 피처에 반영\n"
            "• 사후 규칙: 클러스터링 후 중간온도대를 달력 순서로 상반기=봄, 하반기=가을로 라벨링"
        )
    )

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

# =========================
# tmin/tmax 선택
# =========================
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
    st.info("tmin/tmax 컬럼을 선택하세요."); st.stop()

# =========================
# 전처리 & 선택 피처 만들기
# =========================
dfK = df.copy()
dfK["year"] = dfK["date"].dt.year
dfK["month"] = dfK["date"].dt.month
dfK["doy"] = dfK["date"].dt.dayofyear
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

# 주기 인코딩(DOY) 특성
dfK["sin_doy"] = np.sin(2*np.pi*dfK["doy"]/365.0)
dfK["cos_doy"] = np.cos(2*np.pi*dfK["doy"]/365.0)

# =========================
# K-means 학습
# =========================
if season_split_method == "주기 인코딩(DOY)":
    feature_cols = [tmin_col, tmax_col, "sin_doy", "cos_doy"]
else:
    feature_cols = [tmin_col, tmax_col]

X = dfK[feature_cols].to_numpy()
if scaler_opt == "표준화(Standard)":
    X = StandardScaler().fit_transform(X)
elif scaler_opt == "MinMax":
    X = MinMaxScaler().fit_transform(X)

km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
dfK["cluster"] = km.fit_predict(X)

# 클러스터 요약(원 스케일)
dfK["temp_mean"] = (dfK[tmin_col] + dfK[tmax_col]) / 2
centers = dfK.groupby("cluster")[[tmin_col, tmax_col, "temp_mean"]].mean().reset_index()
st.subheader("클러스터 요약(평균)")
st.dataframe(centers.round(2))

# 계절 라벨 매핑: 평균기온 낮→높
cluster_order = (centers.sort_values("temp_mean")["cluster"].tolist())
if k_clusters == 4:
    # 초기 매핑(낮→높)
    base_names = ["겨울", "봄", "가을", "여름"]
    season_map = {cl: base_names[i] for i, cl in enumerate(cluster_order)}

    if season_split_method == "사후 규칙(연도 내 순서)":
        # 중간 두 개(봄/가을 후보)를 합쳐 '중간'으로 보고 달력 순서로 분리
        mid_clusters = cluster_order[1:3]
        dfK["season_unsup"] = dfK["cluster"].map(season_map)
        # 경계는 7/1(DOY≈183)로 설정
        bound = 183
        mask_mid = dfK["cluster"].isin(mid_clusters)
        dfK.loc[mask_mid & (dfK["doy"] < bound),  "season_unsup"] = "봄"
        dfK.loc[mask_mid & (dfK["doy"] >= bound), "season_unsup"] = "가을"
    else:
        # DOY 인코딩: 온도+시기를 함께 학습했으므로 기본 매핑 유지
        dfK["season_unsup"] = dfK["cluster"].map(season_map)
else:
    mids = [f"중간{i+1}" for i in range(max(k_clusters-2, 0))]
    dyn = ["추움"] + mids + ["더움"]
    season_map = {cl: dyn[i] for i, cl in enumerate(cluster_order)}
    dfK["season_unsup"] = dfK["cluster"].map(season_map)

# 가장 더운/추운 라벨
season_order = (dfK.groupby("season_unsup")["temp_mean"].mean()
                .sort_values().reset_index()["season_unsup"].tolist())
hottest_label = season_order[-1]
coldest_label = season_order[0]

# =========================
# ① tmin–tmax 산점도 (계절 색)
# =========================
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

# =========================
# ② 연도별 계절 일수 추세
# =========================
st.subheader("② 연도별 계절 일수 추세")
counts = dfK.groupby(["year", "season_unsup"]).size().reset_index(name="days")
present_labels2 = counts["season_unsup"].dropna().unique().tolist()
color_enc2 = season_color_encoding(present_labels2)
line_all = alt.Chart(counts).mark_line(point=True).encode(
    x=alt.X("year:O", title="연도"),
    y=alt.Y("days:Q", title="일수"),
    color=color_enc2,
    tooltip=["year:O", "season_unsup:N", "days:Q"]
).properties(height=360)
st.altair_chart(line_all, use_container_width=True)

# =========================
# ③ (선택 계절) 길이 추세
# =========================
st.subheader("③ (선택 계절) 길이 추세")
default_idx = season_order.index(hottest_label) if hottest_label in season_order else len(season_order) - 1
season_to_view = st.selectbox("추세를 볼 계절", options=season_order, index=default_idx)
sel = counts[counts["season_unsup"] == season_to_view].copy()
if not sel.empty:
    base = alt.Chart(sel).mark_line(point=True).encode(
        x="year:O",
        y=alt.Y("days:Q", title=f"{season_to_view} 일수"),
        color=alt.value(SEASON_COLORS.get(season_to_view, None)),  # 해당 계절 고정색
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

# =========================
# ④ {가장 더운 계절} 전이 시점 — 첫/마지막 (연중일수)
# =========================
st.subheader(f"④ {hottest_label} 전이 시점 — 첫/마지막 (연중일수)")
hot_df = dfK[dfK["season_unsup"] == hottest_label].copy()
if hot_df.empty:
    st.info(f"{hottest_label} 데이터가 부족합니다.")
else:
    trans = (hot_df.groupby("year")
                  .agg(first_hot=("date", "min"), last_hot=("date", "max"))
                  .reset_index())
    trans["first_hot_doy"] = trans["first_hot"].dt.dayofyear
    trans["last_hot_doy"]  = trans["last_hot"].dt.dayofyear
    trans["first_hot_label"] = trans["first_hot"].dt.strftime("%m-%d")
    trans["last_hot_label"]  = trans["last_hot"].dt.strftime("%m-%d")
    trans["hot_span_days"] = (trans["last_hot_doy"] - trans["first_hot_doy"] + 1).clip(lower=0)

    # 연속 축을 위해 빈 연도 채우기
    all_years = pd.DataFrame({"year": sorted(dfK["year"].unique())})
    trans = all_years.merge(trans, on="year", how="left")

    # 첫 도달일(해당 계절 색)
    first_color = SEASON_COLORS.get(hottest_label, None)
    ch_first = alt.Chart(trans).mark_line(point=True, color=first_color).encode(
        x=alt.X("year:O", title="연도"),
        y=alt.Y("first_hot_doy:Q", title=f"첫 {hottest_label} (연중 일수)"),
        tooltip=["year:O",
                 alt.Tooltip("first_hot_label:N", title=f"첫 {hottest_label}"),
                 alt.Tooltip("first_hot_doy:Q", title="연중일수")]
    ).properties(height=220, title=f"첫 {hottest_label} 도달 — 낮을수록 빨라짐")
    # 마지막 종료일(같은 계절색의 약간 어두운 톤이 필요하면 별도 지정 가능)
    ch_last = alt.Chart(trans).mark_line(point=True, color=first_color).encode(
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
        trend_f = alt.Chart(df_fit_first).mark_line(color="orange").encode(x="year:O", y="pred:Q")
        layers_first.append(trend_f)

    layers_last = [ch_last]
    df_fit_last = trans.dropna(subset=["last_hot_doy"])
    if df_fit_last["year"].nunique() >= 3:
        lr_l = LinearRegression().fit(df_fit_last[["year"]].astype(int), df_fit_last["last_hot_doy"])
        df_fit_last["pred"] = lr_l.predict(df_fit_last[["year"]].astype(int))
        trend_l = alt.Chart(df_fit_last).mark_line(color="orange").encode(x="year:O", y="pred:Q")
        layers_last.append(trend_l)

    st.altair_chart(alt.layer(*layers_first), use_container_width=True)
    st.altair_chart(alt.layer(*layers_last), use_container_width=True)

# =========================
# ⑤ 연-월 히트맵 (선택)
# =========================
st.subheader("⑤ 연-월 히트맵 (평균기온)")
hm_metric = [c for c in ["평균기온(℃)", "최고기온(℃)", "최저기온(℃)"] if c in df.columns]
hm_metric = hm_metric[0] if hm_metric else df.select_dtypes(include=np.number).columns.tolist()[0]
hm = df[["date", hm_metric]].dropna().copy()
if not hm.empty:
    hm["year"] = hm["date"].dt.year.astype(int)
    hm["month"] = hm["date"].dt.month.astype(int)
    g = (hm.groupby(["year","month"])[hm_metric].mean().reset_index(name="val"))
    heat = alt.Chart(g).mark_rect().encode(
        x=alt.X("month:O", title="월"),
        y=alt.Y("year:O", title="연도"),
        color=alt.Color("val:Q", title=hm_metric, scale=alt.Scale(scheme="turbo")),
        tooltip=["year:O", "month:O", alt.Tooltip("val:Q", format=".2f")]
    ).properties(height=20*len(g['year'].unique()), width=600)
    st.altair_chart(heat, use_container_width=True)

st.markdown("---")
with st.expander("교육 메모", expanded=False):
    st.markdown("""
- **주기 인코딩(DOY)**: 날짜를 sin/cos로 변환해 **연중 시기** 정보를 피처에 녹입니다 → 같은 온도라도 4월/10월이 자연스럽게 분리.
- **사후 규칙**: 중간 온도대(봄·가을 후보)를 **달력 순서(7/1 기준)**로 앞=봄, 뒤=가을로 라벨링합니다.
- **색상 팔레트**: 봄=노랑, 여름=빨강, 가을=단풍색, 겨울=하늘색으로 통일해 해석을 돕습니다.
- **전이 시점(DOY)**: 연도를 제거한 연중일수로 비교해, **얼마나 빨라졌는지/늦어졌는지**를 직접적으로 해석합니다.
""")
