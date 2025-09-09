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
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("행(일 수)", f"{len(df_daily):,}")
with c2:
    st.metric("열(특성 수)", f"{df_daily.shape[1]:,}")
with c3:
    st.metric("결측 총합", f"{int(df_daily.isna().sum().sum()):,}")

with st.expander("데이터 타입 / 결측치 요약", expanded=False):
    st.write("**데이터 타입**")
    st.dataframe(pd.DataFrame(df_daily.dtypes, columns=["dtype"]))
    st.write("**결측치 합계(열별)**")
    miss = df_daily.isna().sum()
    miss_df = miss[miss > 0].to_frame("missing_count")
    if miss_df.empty:
        st.success("결측치 없음 ✅")
    else:
        st.dataframe(miss_df)

# 숫자형 후보 중 '기온 3종' 자동 후보 지정
num_cols = df_daily.select_dtypes(include=np.number).columns.tolist()
heuristic_order = ["tavg", "temp", "tmean", "avg_temp", "tmax", "tmin"]
default_targets = [c for c in heuristic_order if c in num_cols]
default_show = default_targets[:2] if default_targets else (num_cols[:2] if len(num_cols) >= 2 else num_cols)

st.subheader("일 단위 라인 차트")
eda_cols = st.multiselect("표시할 기온(숫자형) 컬럼", options=num_cols, default=default_show)
if eda_cols:
    # 여러 컬럼을 겹쳐보기
    df_melt = df_daily[["date"] + eda_cols].melt("date", var_name="metric", value_name="value")
    line = alt.Chart(df_melt).mark_line().encode(
        x="date:T",
        y=alt.Y("value:Q", title="값"),
        color="metric:N",
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=".2f")]
    ).properties(height=320)
    st.altair_chart(line, use_container_width=True)
else:
    st.info("표시할 숫자형(기온) 컬럼을 선택해 주세요.")

# =========================
# 연평균 회귀 (연도 단위)
# =========================
st.header("📈 연평균 선형 회귀 — X=연도, Y=선택지표(연평균)")
# 목표(기온 지표) 선택
target_choices = default_targets if default_targets else num_cols
if not target_choices:
    st.error("연평균 대상이 될 숫자형 컬럼이 필요합니다.")
    st.stop()
target_col = st.selectbox("연평균으로 사용할 기온 지표", options=target_choices, index=0)

# 연평균 계산(2% 규칙 적용)
df_year = compute_yearly_mean(df_daily, target_col=target_col, miss_threshold=miss_threshold)

# 품질 로그
with st.expander("연도별 품질 로그(결측비율 2% 초과 연도는 제외됨)", expanded=False):
    df = df_daily.copy()
    df["year"] = df["date"].dt.year
    logs = []
    for y, g in df.groupby("year", dropna=True):
        n_total = len(g)
        n_miss = g[target_col].isna().sum()
        r = (n_miss / n_total) if n_total else np.nan
        logs.append({"year": int(y), "days": n_total, "missing": int(n_miss), "missing_ratio": r})
    st.dataframe(pd.DataFrame(logs).sort_values("year"))

st.subheader("연평균 테이블(결측 연도 제외 전)")
st.dataframe(df_year)

# 회귀에 사용할 데이터(결측 연도 제거)
df_fit = df_year.dropna(subset=["avg"]).copy()

if len(df_fit) < 3:
    st.warning("연평균 유효 연도가 충분하지 않습니다(최소 3년 권장). 데이터/결측 임계값을 확인하세요.")
else:
    # Train/Test 분할
    X = df_fit[["year"]]  # X=연도
    y = df_fit["avg"]
    # 시간 데이터라 랜덤 분할 대신 과적합을 피하려면 '최근 데이터 테스트'도 가능하지만, 교육 목적 상 간단 분할 사용
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=max(1, int(len(df_fit)*0.2)) / len(df_fit), random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 호환성: RMSE는 sqrt(MSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("훈련 샘플(연도)", f"{len(X_train)}")
    with c2:
        st.metric("테스트 샘플(연도)", f"{len(X_test)}")
    with c3:
        st.metric("RMSE (테스트)", f"{rmse:.3f}")

    # 회귀식 표시: y = a*year + b
    a = float(model.coef_[0])
    b = float(model.intercept_)
    st.caption(f"회귀식: **avg ≈ {a:.4f} × year + {b:.4f}**")

    # 전체 연도에 대한 예측선
    df_fit["pred"] = model.predict(df_fit[["year"]])

    chart = (
        alt.Chart(df_fit).mark_circle(size=70, opacity=0.85).encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y("avg:Q", title=f"연평균 {target_col}"),
            tooltip=["year:O", alt.Tooltip("avg:Q", format=".2f")]
        )
        + alt.Chart(df_fit).mark_line().encode(
            x="year:O",
            y="pred:Q"
        )
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)

# 푸터: 교육 메모
st.markdown("---")
st.markdown("""
**교육 메모**  
- EDA는 **일 단위**로 패턴/이상치를 먼저 확인합니다.  
- 모델링은 **연평균(연도 단위)**로 축약해 **장기 추세**를 간단한 선형회귀로 살펴봅니다.  
- 한 해의 결측이 일정 비율(기본 2%)을 넘으면 **해당 연도를 제외**해 데이터 품질을 확보합니다.  
- 필요시 임계값을 바꿔 결과 변화를 관찰해 보세요.
""")
