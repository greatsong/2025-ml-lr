import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="🌡️ EDA(일 단위)+연평균 회귀+미래예측", layout="wide")

# =========================
# 파일/CSV 로딩 유틸
# =========================
def find_latest_csv(search_dirs=("data", ".")):
    """search_dirs 순서대로 *.csv를 모아 가장 최근 수정 파일 경로를 반환. 없으면 None."""
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
    """cp949 → utf-8-sig → utf-8 → euc-kr 순으로 시도. 상단 N행 스킵."""
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
    # 마지막 시도(인코딩 미지정)
    return pd.read_csv(
        file_or_path,
        skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
    )

def load_default_or_simulated(skip_top_rows=7):
    """data/ 또는 현재 폴더에서 최신 CSV 자동 로드. 없으면 시뮬레이션 생성."""
    latest = find_latest_csv(("data", "."))
    if latest is not None:
        df = smart_read_csv(latest, skip_top_rows=skip_top_rows)
        return df, latest
    # 시뮬레이션(1년)
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
# 연평균 계산(연도별)
# =========================
def compute_yearly_mean(df_daily, target_col, miss_threshold=0.02):
    """
    연도별로 target_col의 연평균을 계산.
    - 한 해에서 target_col 결측 비율이 miss_threshold(기본 2%) '초과'면 해당 연도 avg=NaN(제외).
    """
    df = df_daily.copy()
    if "date" not in df.columns:
        raise ValueError("date 컬럼이 필요합니다.")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    out = []
    for y, g in df.groupby("year", dropna=True):
        if target_col not in g.columns or len(g) == 0:
            out.append({"year": int(y), "avg": np.nan})
            continue
        n_total = len(g)
        n_missing = g[target_col].isna().sum()
        miss_ratio = (n_missing / n_total) if n_total else 1.0
        if miss_ratio > miss_threshold:
            avg_val = np.nan
        else:
            avg_val = g[target_col].mean(skipna=True)
        out.append({"year": int(y), "avg": avg_val})

    df_year = pd.DataFrame(out).sort_values("year").reset_index(drop=True)
    return df_year

# =========================
# 사이드바: 옵션
# =========================
with st.sidebar:
    st.header("⚙️ 데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7,
                       help="요청사항: 1~7행은 메타/설명일 수 있어 기본 7행 스킵")
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

# === 날짜 컬럼명 정규화: '날짜' → 'date' ===
if "date" not in df_daily.columns and "날짜" in df_daily.columns:
    df_daily = df_daily.rename(columns={"날짜": "date"})

# 날짜 컬럼 확인
if "date" not in df_daily.columns:
    st.error("데이터에 'date' (또는 '날짜') 컬럼이 필요합니다.")
    st.dataframe(df_daily.head())
    st.stop()

# 날짜 파싱
try:
    df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")
except Exception:
    pass

# =========================
# EDA (일 단위)
# =========================
st.header("📊 EDA — 일 단위")
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
    st.dataframe(miss_df if not miss_df.empty else pd.DataFrame({"message": ["결측치 없음 ✅"]}))

# 숫자형 후보 및 기본 기온 컬럼 추정
num_cols = df_daily.select_dtypes(include=np.number).columns.tolist()
heuristic_order = ["tavg", "temp", "tmean", "avg_temp", "tmax", "tmin", "평균기온", "최고기온", "최저기온"]
default_targets = [c for c in heuristic_order if c in num_cols]
default_show = default_targets[:2] if default_targets else (num_cols[:2] if len(num_cols) >= 2 else num_cols)

# 색상 맵: tmax=red, tavg계열=green, tmin=blue (+한글명 포함)
base_color_map = {
    "tmax": "red", "최고기온": "red",
    "tavg": "green", "temp": "green", "tmean": "green", "avg_temp": "green", "평균기온": "green",
    "tmin": "blue", "최저기온": "blue"
}

# ① 라인차트(일 단위)
st.subheader("① 라인 차트(일 단위)")
eda_cols = st.multiselect("표시할 기온(숫자형) 컬럼", options=num_cols, default=default_show)
if eda_cols:
    df_melt = df_daily[["date"] + eda_cols].melt("date", var_name="metric", value_name="value")
    present = df_melt["metric"].unique().tolist()
    domain, colors = [], []
    for m in present:
        if m in base_color_map:
            domain.append(m); colors.append(base_color_map[m])
    color_enc = alt.Color("metric:N") if not domain else alt.Color("metric:N", scale=alt.Scale(domain=domain, range=colors))
    line = alt.Chart(df_melt).mark_line().encode(
        x="date:T", y=alt.Y("value:Q", title="값"), color=color_enc,
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=".2f")]
    ).properties(height=320)
    st.altair_chart(line, use_container_width=True)

# ② 전체 히스토그램
st.subheader("② 전체 히스토그램")
hist_metric = st.selectbox("히스토그램 대상 컬럼", options=(default_targets or num_cols))
if hist_metric:
    bins = st.slider("Bins", 10, 80, 40)
    chart_hist = alt.Chart(df_daily.dropna(subset=[hist_metric])).mark_bar().encode(
        x=alt.X(f"{hist_metric}:Q", bin=alt.Bin(maxbins=bins), title=hist_metric),
        y="count()",
        tooltip=[alt.Tooltip(f"{hist_metric}:Q", format=".2f"), "count()"]
    ).properties(height=300)
    st.altair_chart(chart_hist, use_container_width=True)

# ③ 박스플랏 — 월 선택 → 모든 연도(합산 1개 박스)
st.subheader("③ 박스플랏 — 월 선택 → 모든 연도(합산 1개 박스)")
month_for_all = st.selectbox("월 선택(1~12)", options=list(range(1, 13)), index=0, key="box_month_all")
box_metric_all = st.selectbox("지표 선택", options=(default_targets or num_cols), index=0, key="box_metric_all")
if box_metric_all:
    df_month_all = df_daily.copy()
    df_month_all["month"] = df_month_all["date"].dt.month
    sub_all = df_month_all[(df_month_all["month"] == month_for_all) & (~df_month_all[box_metric_all].isna())]
    if sub_all.empty:
        st.info("해당 월 데이터가 없습니다.")
    else:
        sub_all["label"] = f"{month_for_all:02d}월 (모든 연도)"
        box_all = alt.Chart(sub_all).mark_boxplot(size=80).encode(
            x=alt.X("label:N", title="기간"),
            y=alt.Y(f"{box_metric_all}:Q", title=f"{box_metric_all}"),
            tooltip=[alt.Tooltip(f"{box_metric_all}:Q", format=".2f")]
        ).properties(height=320)
        st.altair_chart(box_all, use_container_width=True)

# ④ 박스플랏 — 월 선택 → 여러 연도 선택(비교)
st.subheader("④ 박스플랏 — 월 선택 → 여러 연도 선택")
df_years = df_daily.copy()
df_years["year"] = df_years["date"].dt.year
avail_years = sorted(df_years["year"].dropna().unique().tolist())

month_for_multi = st.selectbox("월 선택 (1~12)", options=list(range(1, 13)), index=0, key="box_month_multi")
years_for_multi = st.multiselect(
    "연도 선택 (2개 이상 선택 권장)", options=avail_years,
    default=avail_years[-2:] if len(avail_years) >= 2 else avail_years
)
box_metric_multi = st.selectbox("지표 선택", options=(default_targets or num_cols), index=0, key="box_metric_multi")

if years_for_multi and box_metric_multi:
    sub_multi = df_daily.copy()
    sub_multi["year"] = sub_multi["date"].dt.year
    sub_multi["month"] = sub_multi["date"].dt.month
    sub_multi = sub_multi[
        (sub_multi["year"].isin(years_for_multi)) &
        (sub_multi["month"] == month_for_multi) &
        (~sub_multi[box_metric_multi].isna())
    ]
    if sub_multi.empty:
        st.info("해당 월/연도 데이터가 없습니다.")
    else:
        box_multi = alt.Chart(sub_multi).mark_boxplot(size=40).encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y(f"{box_metric_multi}:Q", title=f"{box_metric_multi}"),
            color="year:O",
            tooltip=[alt.Tooltip(f"{box_metric_multi}:Q", format=".2f"), "year:O"]
        ).properties(height=320)
        st.altair_chart(box_multi, use_container_width=True)

# =========================
# 연평균 회귀 (연도 단위) + 학습구간 슬라이더 + 미래예측(마지막 해만 빨간점)
# =========================
st.header("📈 연평균 선형 회귀 — X=연도, Y=선택지표(연평균)")
target_choices = [c for c in ["tavg", "temp", "tmean", "avg_temp", "평균기온", "tmax", "최고기온", "tmin", "최저기온"] if c in num_cols] or num_cols
if not target_choices:
    st.error("연평균 대상이 될 숫자형 컬럼이 필요합니다.")
    st.stop()
target_col = st.selectbox("연평균으로 사용할 기온 지표", options=target_choices, index=0)

df_year = compute_yearly_mean(df_daily, target_col=target_col, miss_threshold=miss_threshold)
st.subheader("연평균 테이블(결측 연도 포함)")
st.dataframe(df_year)

# 유효 연도만
df_fit = df_year.dropna(subset=["avg"]).copy()
if len(df_fit) < 3:
    st.warning("연평균 유효 연도가 충분하지 않습니다(최소 3년 권장).")
else:
    min_y, max_y = int(df_fit["year"].min()), int(df_fit["year"].max())

    # 🔧 학습 데이터 구간 슬라이더
    st.subheader("🔧 학습 데이터 구간(연도 범위) 선택")
    train_range = st.slider("학습에 사용할 연도 범위", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1)

    # 학습/테스트 분할: 선택 구간을 학습, 그 외(유효 연도 중) 테스트
    train_mask = (df_fit["year"] >= train_range[0]) & (df_fit["year"] <= train_range[1])
    train_df = df_fit[train_mask].copy()
    test_df  = df_fit[~train_mask].copy()

    if len(train_df) < 2:
        st.warning("학습 구간에 최소 2개 연도 이상이 필요합니다.")
    else:
        # 모델 학습
        X_train, y_train = train_df[["year"]], train_df["avg"]
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 평가 (테스트 연도 존재 시)
        if len(test_df) >= 1:
            y_pred = model.predict(test_df[["year"]])
            rmse = np.sqrt(mean_squared_error(test_df["avg"], y_pred))
            st.metric("RMSE (테스트: 학습구간 밖 연도)", f"{rmse:.3f}")
        else:
            st.info("테스트 구간(학습 범위 밖 유효 연도)이 없어 RMSE를 계산하지 않습니다.")

        a = float(model.coef_[0]); b = float(model.intercept_)
        st.caption(f"회귀식(학습구간 기반): **avg ≈ {a:.4f} × year + {b:.4f}**")

        # 시각화: 학습/테스트 구분 + 회귀선
        df_plot = df_fit.copy()
        df_plot["split"] = np.where((df_plot["year"] >= train_range[0]) & (df_plot["year"] <= train_range[1]), "train", "test")
        df_plot["pred"] = model.predict(df_plot[["year"]])

        pts = alt.Chart(df_plot).mark_circle(size=80, opacity=0.9).encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y("avg:Q", title=f"연평균 {target_col}"),
            color=alt.Color("split:N", scale=alt.Scale(domain=["train","test"], range=["#2E7D32", "#455A64"])),
            tooltip=["year:O", alt.Tooltip("avg:Q", format=".2f"), "split:N"]
        )
        regline = alt.Chart(df_plot).mark_line(color="black").encode(
            x="year:O", y="pred:Q"
        )
        base_chart = pts + regline

        # 🔮 미래 예측: 완전한 마지막 해 + 1 ~ 2100
        st.subheader("🔮 미래 예측 (마지막 해만 빨간 점 + 레이블)")

        max_dt = df_daily["date"].dropna().max()
        if pd.isna(max_dt):
            st.info("날짜 데이터가 없어 예측 범위를 계산할 수 없습니다.")
            st.altair_chart(base_chart, use_container_width=True)
        else:
            last_day = pd.Timestamp(max_dt.year, 12, 31)
            last_complete_year = max_dt.year if max_dt >= last_day else (max_dt.year - 1)
            start_pred_year = min(max(last_complete_year + 1, min_y), 2100)

            if start_pred_year > 2100:
                st.warning("예측 시작 연도가 2100을 초과합니다. 더 최근 데이터가 필요합니다.")
                st.altair_chart(base_chart, use_container_width=True)
            else:
                # 단일 연도 예측(그 연도만 표시)
                year_to_predict = st.number_input(
                    "단일 연도 예측",
                    min_value=int(start_pred_year), max_value=2100,
                    value=int(min(start_pred_year + 5, 2100)), step=1
                )
                single_df = None
                if st.button("해당 연도 예측"):
                    pred_single = float(model.predict(pd.DataFrame({"year": [year_to_predict]}))[0])
                    st.success(f"📌 {year_to_predict}년 예상 {target_col} = **{pred_single:.2f}**")
                    single_df = pd.DataFrame({"year": [year_to_predict], "pred": [pred_single], "label": [f"{pred_single:.2f}"]})

                # 구간 예측(슬라이더): 마지막 해만 빨간 점
                yr_min = int(start_pred_year); yr_max = 2100
                yr_range = st.slider("예측 구간(연도 범위)", min_value=yr_min, max_value=yr_max,
                                     value=(yr_min, min(yr_min+20, yr_max)), step=1)

                future_years = pd.DataFrame({"year": list(range(yr_range[0], yr_range[1] + 1))})
                future_years["pred"] = model.predict(future_years[["year"]])
                future_years["label"] = future_years["pred"].map(lambda v: f"{v:.2f}")

                # 점선 예측선
                chart_future_line = alt.Chart(future_years).mark_line(strokeDash=[5,5], color="gray").encode(
                    x=alt.X("year:O", title="연도"),
                    y=alt.Y("pred:Q", title=f"연평균 {target_col} (예측)")
                )

                # ✔ 마지막 해만 빨간 점 + 레이블
                last_year = int(future_years["year"].max())
                last_df = future_years[future_years["year"] == last_year]
                last_point = alt.Chart(last_df).mark_point(color="red", size=120).encode(
                    x=alt.X("year:O"), y=alt.Y("pred:Q")
                )
                last_label = alt.Chart(last_df).mark_text(dy=-14, color="red").encode(
                    x=alt.X("year:O"), y=alt.Y("pred:Q"), text="label:N"
                )

                charts = base_chart + chart_future_line + last_point + last_label

                # 단일 연도 예측 버튼을 눌렀다면 그 해도 빨간 점/라벨(요청과 충돌하지 않음: '구간'은 마지막 해만, '단일'은 단일만)
                if single_df is not None:
                    single_point = alt.Chart(single_df).mark_point(color="red", size=120).encode(
                        x=alt.X("year:O"), y=alt.Y("pred:Q")
                    )
                    single_label = alt.Chart(single_df).mark_text(dy=-14, color="red").encode(
                        x=alt.X("year:O"), y=alt.Y("pred:Q"), text="label:N"
                    )
                    charts = charts + single_point + single_label

                st.altair_chart(charts, use_container_width=True)

                with st.expander("예측 테이블", expanded=False):
                    st.dataframe(future_years)

# 푸터
st.markdown("---")
st.markdown("""
**교육 메모**  
- 히스토그램으로 전체 분포를, 월별 박스플랏으로 연도 간 계절 분포의 차이를 살펴보세요.  
- "월→여러 연도 박스플랏"으로 특정 월의 연도별 분포를 한눈에 비교할 수 있습니다.  
- 학습 구간 슬라이더로 회귀선이 어떻게 바뀌는지(추세 추정의 민감도)를 실습해 보세요.  
- 미래 예측은 **마지막 ‘완전한’ 연도 다음 해부터** 2100년까지 허용하며, 예측 시각화는 **예측 구간의 '마지막 해'만** 빨간 점과 레이블로 표시됩니다.
""")
