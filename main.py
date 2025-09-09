import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🌡️ 일일 EDA + 연평균 선형회귀(연도)", layout="wide")

# =========================
# 유틸: 파일/CSV 로딩
# =========================
def find_latest_csv(search_dirs=("data", ".")):
    """
    search_dirs 순서대로 탐색해 가장 최신 수정(csv) 파일 경로를 반환.
    없으면 None.
    """
    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        candidates.extend(glob.glob(os.path.join(d, "*.csv")))
    if not candidates:
        return None
    # 최신 수정시간 기준 정렬
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def smart_read_csv(file_or_path, skip_top_rows=7):
    """
    cp949 → utf-8-sig → utf-8 → euc-kr 순으로 시도.
    상단 skip_top_rows(기본 7행) 스킵.
    """
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
    # 마지막 시도(인코딩 지정 없이)
    return pd.read_csv(
        file_or_path,
        skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
    )

def load_default_or_simulated(skip_top_rows=7):
    """
    data/ 또는 현재 폴더에서 최신 CSV를 찾아 로드.
    없으면 시뮬레이션 데이터 생성.
    """
    latest = find_latest_csv(("data", "."))
    if latest is not None:
        df = smart_read_csv(latest, skip_top_rows=skip_top_rows)
        return df, latest
    # 시뮬레이션(하루 단위 1년)
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    doy = np.arange(365)
    base = 15 + 10*np.sin(2*np.pi*(doy/365))
    noise = rng.normal(0, 1.5, 365)
    tavg = base + noise
    tmax = tavg + rng.normal(3, 0.7, 365)
    tmin = tavg - rng.normal(3, 0.7, 365)
    df = pd.DataFrame({"date": days, "tavg": tavg.round(2), "tmax": tmax.round(2), "tmin": tmin.round(2)})
    return df, "(simulated)"

# =========================
# 유틸: 연평균 계산(연도별)
# =========================
def compute_yearly_mean(df_daily, target_col, miss_threshold=0.02):
    """
    연도별로 target_col의 연평균을 계산.
    - 한 해에서 target_col 결측 비율이 miss_threshold(기본 2%) 초과면 해당 연도 avg=NaN으로 처리.
    """
    df = df_daily.copy()
    if "date" not in df.columns:
        raise ValueError("데이터에 'date' 컬럼이 필요합니다.")
    # 날짜 파싱
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    out = []
    for y, g in df.groupby("year", dropna=True):
        # target 존재 확인
        if target_col not in g.columns:
            out.append({"year": y, "avg": np.nan})
            continue
        n_total = len(g)
        if n_total == 0:
            out.append({"year": y, "avg": np.nan})
            continue
        n_missing = g[target_col].isna().sum()
        miss_ratio = n_missing / n_total
        if miss_ratio > miss_threshold:
            avg_val = np.nan  # 2% 초과 → 해당 연도 제외
        else:
            avg_val = g[target_col].mean(skipna=True)
        out.append({"year": int(y), "avg": avg_val})
    df_year = pd.DataFrame(out).sort_values("year").reset_index(drop=True)
    return df_year

# =========================
# 사이드바: 데이터 입력/옵션
# =========================
with st.sidebar:
    st.header("⚙️ 데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7,
                       help="요청: 1~7행은 메타/설명일 수 있어 기본 7행 스킵")
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"], horizontal=False)
    miss_threshold = st.number_input("연간 결측 임계값(비율)", min_value=0.0, max_value=1.0, value=0.02, step=0.01,
                                     help="한 해의 결측 비율이 이 값을 초과하면 해당 연도는 제외")
    st.caption("인코딩은 cp949 → utf-8-sig → utf-8 → euc-kr 순으로 자동 시도합니다.")

# =========================
# 데이터 로드
# =========================
if src == "CSV 업로드":
    up = st.file_uploader("CSV 업로드 (권장: cp949)", type=["csv"])
    if up is None:
        st.info("CSV를 업로드해 주세요. (또는 사이드바에서 '기본'을 선택)")
        st.stop()
    df_daily = smart_read_csv(up, skip_top_rows=skip_n)
    loaded_from = "(uploaded)"
else:
    df_daily, loaded_from = load_default_or_simulated(skip_top_rows=skip_n)

st.success(f"데이터 소스: **{loaded_from}**")

# 날짜 파싱
if "date" not in df_daily.columns:
    st.error("데이터에 'date' 컬럼이 필요합니다. (YYYY-MM-DD 등)")
    st.dataframe(df_daily.head())
    st.stop()

try:
    df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")
except Exception:
    pass

# =========================
# EDA (일 단위)
# =========================
st.header("📊 EDA — 일 단위 데이터")
c1, c2, c3 = st.
