import os
import glob
import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_page_config(page_title="🌡️ 기온 EDA·회귀·예측 + K-means 계절", layout="wide")

# =========================
# 파일/CSV 로딩 유틸
# =========================
def find_latest_csv(search_dirs=("data", ".")):
    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        candidates.extend(glob.glob(os.path.join(d, "*.csv")))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def smart_read_csv(file_or_path, skip_top_rows=7):
    encodings = ["cp949", "utf-8-sig", "utf-8", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(
                file_or_path,
                encoding=enc,
                skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
            )
        except Exception:
            continue
    return pd.read_csv(
        file_or_path,
        skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
    )

def load_default_or_simulated(skip_top_rows=7):
    latest = find_latest_csv(("data", "."))
    if latest is not None:
        df = smart_read_csv(latest, skip_top_rows=skip_top_rows)
        return df, latest
    # 시뮬 데이터
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    doy = np.arange(365)
    base = 15 + 10*np.sin(2*np.pi*(doy/365))
    noise = rng.normal(0, 1.5, 365)
    tavg = base + noise
    tmax = tavg + rng.normal(3, 0.7, 365)
    tmin = tavg - rng.normal(3, 0.7, 365)
    df = pd.DataFrame({
        "날짜": days,
        "평균기온(℃)": np.round(tavg, 2),
        "최고기온(℃)": np.round(tmax, 2),
        "최저기온(℃)": np.round(tmin, 2),
    })
    return df, "(simulated)"

def to_numeric_strict(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^-\d\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# =========================
# 연평균 계산
# =========================
def compute_yearly_mean(df_daily, target_col, miss_threshold=0.02):
    df = df_daily.copy()
    if "date" not in df.columns:
        raise ValueError("date 컬럼 필요")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    max_dt = df["date"].dropna().max()
    if pd.isna(max_dt):
        return pd.DataFrame(columns=["year", "avg"]), None
    last_complete_year = max_dt.year if max_dt >= pd.Timestamp(max_dt.year, 12, 31) else (max_dt.year - 1)

    df["year"] = df["date"].dt.year
    out = []
    for y, g in df.groupby("year", dropna=True):
        y = int(y)
        if y > last_complete_year:
            out.append({"year": y, "avg": np.nan})
            continue

        start, end = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
        full_idx = pd.date_range(start, end, freq="D")
        full = pd.DataFrame(index=full_idx)

        if target_col not in g.columns:
            out.append({"year": y, "avg": np.nan})
            continue

        g2 = g.set_index("date")[[target_col]].rename(columns={target_col: "val"})
        merged = full.join(g2, how="left")
        miss_ratio = merged["val"].isna().sum() / len(merged)
        avg_val = np.nan if miss_ratio > miss_threshold else merged["val"].mean(skipna=True)
        out.append({"year": y, "avg": avg_val})

    df_year = pd.DataFrame(out).sort_values("year").reset_index(drop=True)
    df_year = df_year[df_year["year"] <= last_complete_year].reset_index(drop=True)
    return df_year, last_complete_year

# =========================
# 데이터 로드
# =========================
with st.sidebar:
    st.header("⚙️ 데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7)
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"])
    miss_threshold = st.number_input("연간 결측 임계값(비율)", 0.0, 1.0, 0.02, 0.01)

if src == "CSV 업로드":
    up = st.file_uploader("CSV 업로드", type=["csv"])
    if up is None:
        st.stop()
    df_raw = smart_read_csv(up, skip_top_rows=skip_n)
    loaded_from = "(uploaded)"
else:
    df_raw, loaded_from = load_default_or_simulated(skip_top_rows=skip_n)

df = df_raw.copy()
if "date" not in df.columns and "날짜" in df.columns:
    df = df.rename(columns={"날짜": "date"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
    if col in df.columns:
        df[col] = to_numeric_strict(df[col])

# =========================
# 🌈 K-means 계절 구분
# =========================
st.header("🌈 K-means 비지도 계절 구분 — 최저/최고기온 기반")

auto_tmin = "최저기온(℃)" if "최저기온(℃)" in df.columns else None
auto_tmax = "최고기온(℃)" if "최고기온(℃)" in df.columns else None

tmin_col = st.selectbox("최저기온 컬럼 (tmin)", options=[None] + df.columns.tolist(),
                        index=(df.columns.tolist().index(auto_tmin)+1 if auto_tmin in df.columns else 0))
tmax_col = st.selectbox("최고기온 컬럼 (tmax)", options=[None] + df.columns.tolist(),
                        index=(df.columns.tolist().index(auto_tmax)+1 if auto_tmax in df.columns else 0))

if tmin_col and tmax_col:
    dfK = df.copy()
    dfK["year"] = dfK["date"].dt.year
    dfK[tmin_col] = to_numeric_strict(dfK[tmin_col])
    dfK[tmax_col] = to_numeric_strict(dfK[tmax_col])
    dfK = dfK.dropna(subset=[tmin_col, tmax_col, "date"])

    k_clusters = st.slider("클러스터 수 K(권장 4)", 3, 6, 4)
    scaler_opt = st.selectbox("스케일링", ["표준화(Standard)", "MinMax", "없음"], index=0)

    X = dfK[[tmin_col, tmax_col]].to_numpy()
    if scaler_opt == "표준화(Standard)":
        X = StandardScaler().fit_transform(X)
    elif scaler_opt == "MinMax":
        X = MinMaxScaler().fit_transform(X)

    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    dfK["cluster"] = km.fit_predict(X)
    dfK["temp_mean"] = (dfK[tmin_col] + dfK[tmax_col]) / 2
    cluster_means = dfK.groupby("cluster")["temp_mean"].mean().sort_values().reset_index()

    # ✅ 여기서 row["cluster"] 대신 row.cluster
    if k_clusters == 4:
        season_names = ["겨울", "봄", "가을", "여름"]
        season_map = {row.cluster: season_names[i] for i, row in enumerate(cluster_means.itertuples(index=False))}
    else:
        mid = [f"중간{i+1}" for i in range(k_clusters-2)]
        dynamic_names = ["추움"] + mid + ["더움"]
        season_map = {row.cluster: dynamic_names[i] for i, row in enumerate(cluster_means.itertuples(index=False))}

    dfK["season_unsup"] = dfK["cluster"].map(season_map)

    st.write(dfK[["date", tmin_col, tmax_col, "season_unsup"]].head())
