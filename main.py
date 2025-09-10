import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_page_config(page_title="🌡️ EDA·연평균 회귀·미래예측 + K-means 계절", layout="wide")

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
# 연평균 계산(연도별) — '완전한 연도'만 유지
# =========================
def compute_yearly_mean(df_daily, target_col, miss_threshold=0.02):
    """
    연도별로 target_col의 연평균을 계산.
    - 마지막 날짜가 그 해의 12/31(=완전한 연도)인지 확인.
    - 결측 비율은 '해당 연도의 전체 일수(365/366)' 대비로 계산(존재하지 않는 날짜도 결측으로 간주).
    - 결측 비율 > miss_threshold이면 avg=NaN.
    - 반환: (df_year, last_complete_year)
    """
    df = df_daily.copy()
    if "date" not in df.columns:
        raise ValueError("date 컬럼이 필요합니다.")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    max_dt = df["date"].dropna().max()
    if pd.isna(max_dt):
        return pd.DataFrame(columns=["year", "avg"]), None

    # 마지막 '완전한' 연도 판단
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
        n_total = len(merged)
        miss_ratio = merged["val"].isna().sum() / n_total
        avg_val = np.nan if miss_ratio > miss_threshold else merged["val"].mean(skipna=True)
        out.append({"year": y, "avg": avg_val})

    df_year = pd.DataFrame(out).sort_values("year").reset_index(drop=True)
    df_year = df_year[df_year["year"] <= last_complete_year].reset_index(drop=True)
    return df_year, last_complete_year

# =========================
# 사이드바: 옵션
# =========================
with st.sidebar:
    st.header("⚙️ 데이터 & 옵션")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7, help="요청사항: 1~7행은 메타/설명일 수 있어 기본 7행 스킵")
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

# 날짜 정규화
if "date" not in df_daily.columns and "날짜" in df_daily.columns:
    df_daily = df_daily.rename(columns={"날짜": "date"})
if "date" not in df_daily.columns:
    st.error("데이터에 'date' (또는 '날짜') 컬럼이 필요합니다.")
    st.dataframe(df_daily.head()); st.stop()
df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")

# =========================
# 컬럼 힌트/색상
# =========================
num_cols = df_daily.select_dtypes(include=np.number).columns.tolist()
heuristic_order = ["tavg", "temp", "tmean", "avg_temp", "tmax", "tmin", "평균기온(℃)", "최고기온(℃)", "최저기온(℃)"]
default_targets = [c for c in heuristic_order if c in num_cols]
default_show = default_targets[:2] if default_targets else (num_cols[:2] if len(num_cols) >= 2 else num_cols)
base_color_map = {
    "tmax": "red", "최고기온": "red",
    "tavg": "green", "temp": "green", "tmean": "green", "avg_temp": "green", "평균기온": "green",
    "tmin": "blue", "최저기온": "blue"
}

# =========================
# EDA (일 단위)
# =========================
st.header("📊 EDA — 일 단위")
c1, c2, c3 = st.columns(3)
with c1: st.metric("행(일 수)", f"{len(df_daily):,}")
with c2: st.metric("열(특성 수)", f"{df_daily.shape[1]:,}")
with c3: st.metric("결측 총합", f"{int(df_daily.isna().sum().sum()):,}")

with st.expander("데이터 타입 / 결측치 요약", expanded=False):
    st.dataframe(pd.DataFrame(df_daily.dtypes, columns=["dtype"]))
    miss = df_daily.isna().sum()
    miss_df = miss[miss > 0].to_frame("missing_count")
    st.dataframe(miss_df if not miss_df.empty else pd.DataFrame({"message": ["결측치 없음 ✅"]}))

st.subheader("① 라인 차트(일 단위)")
eda_cols = st.multiselect("표시할 기온(숫자형) 컬럼", options=num_cols, default=default_show)
if eda_cols:
    df_melt = df_daily[["date"] + eda_cols].melt("date", var_name="metric", value_name="value")
    present = df_melt["metric"].unique().tolist()
    domain, colors = [], []
    for m in present:
        if m in base_color_map: domain.append(m); colors.append(base_color_map[m])
    color_enc = alt.Color("metric:N") if not domain else alt.Color("metric:N", scale=alt.Scale(domain=domain, range=colors))
    line = alt.Chart(df_melt).mark_line().encode(
        x="date:T", y=alt.Y("value:Q", title="값"), color=color_enc,
        tooltip=["date:T", "metric:N", alt.Tooltip("value:Q", format=".2f")]
    ).properties(height=320)
    st.altair_chart(line, use_container_width=True)

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

st.subheader("③ 월별 박스플랏 — 모든 연도 합산")
df_month_all = df_daily.copy()
df_month_all["month"] = df_month_all["date"].dt.month
box_metric_all = st.selectbox("지표 선택", options=(default_targets or num_cols), index=0, key="box_metric_all")
select_mode = st.checkbox("월 선택 모드(체크 시 선택한 달만 표시)", value=False)
months_options = list(range(1, 13))
months_selected = months_options if not select_mode else st.multiselect("표시할 월 선택(1~12)", options=months_options, default=[1,7,12])
if box_metric_all:
    sub_all = df_month_all[(df_month_all["month"].isin(months_selected)) & (~df_month_all[box_metric_all].isna())]
    if not sub_all.empty:
        box_all = alt.Chart(sub_all).mark_boxplot(size=25).encode(
            x=alt.X("month:O", title="월"),
            y=alt.Y(f"{box_metric_all}:Q", title=f"{box_metric_all}"),
            tooltip=[alt.Tooltip(f"{box_metric_all}:Q", format=".2f"), "month:O"]
        ).properties(height=320)
        st.altair_chart(box_all, use_container_width=True)
    else:
        st.info("해당 조건의 데이터가 없습니다.")

st.subheader("④ 박스플랏 — 월 선택 → 여러 연도 선택")
df_years = df_daily.copy()
df_years["year"] = df_years["date"].dt.year
avail_years = sorted(df_years["year"].dropna().unique().tolist())
month_for_multi = st.selectbox("월 선택 (1~12)", options=list(range(1, 13)), index=0, key="box_month_multi")
years_for_multi = st.multiselect("연도 선택 (2개 이상 선택 권장)", options=avail_years,
                                 default=avail_years[-2:] if len(avail_years) >= 2 else avail_years)
box_metric_multi = st.selectbox("지표 선택", options=(default_targets or num_cols), index=0, key="box_metric_multi")
if years_for_multi and box_metric_multi:
    sub_multi = df_daily.copy()
    sub_multi["year"] = sub_multi["date"].dt.year
    sub_multi["month"] = sub_multi["date"].dt.month
    sub_multi = sub_multi[(sub_multi["year"].isin(years_for_multi)) & (sub_multi["month"] == month_for_multi) & (~sub_multi[box_metric_multi].isna())]
    if not sub_multi.empty:
        box_multi = alt.Chart(sub_multi).mark_boxplot(size=40).encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y(f"{box_metric_multi}:Q", title=f"{box_metric_multi}"),
            color="year:O",
            tooltip=[alt.Tooltip(f"{box_metric_multi}:Q", format=".2f"), "year:O"]
        ).properties(height=320)
        st.altair_chart(box_multi, use_container_width=True)
    else:
        st.info("해당 월/연도 데이터가 없습니다.")

# =========================
# 연평균 회귀 + 미래예측(마지막 해만 빨간점)
# =========================
st.header("📈 연평균 선형 회귀 — X=연도, Y=선택지표(연평균)")
target_choices = [c for c in ["tavg", "temp", "tmean", "avg_temp", "평균기온", "tmax", "최고기온", "tmin", "최저기온"] if c in num_cols] or num_cols
if not target_choices:
    st.error("연평균 대상이 될 숫자형 컬럼이 필요합니다."); st.stop()
target_col = st.selectbox("연평균으로 사용할 기온 지표", options=target_choices, index=0)
df_year, last_complete_year = compute_yearly_mean(df_daily, target_col=target_col, miss_threshold=miss_threshold)
st.subheader("연평균 테이블(완전한 연도 기준)")
st.dataframe(df_year)

df_fit = df_year.dropna(subset=["avg"]).copy()
if len(df_fit) >= 3:
    min_y, max_y = int(df_fit["year"].min()), int(df_fit["year"].max())
    st.subheader("🔧 학습 데이터 구간(연도 범위) 선택")
    train_range = st.slider("학습에 사용할 연도 범위", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1)

    train_mask = (df_fit["year"] >= train_range[0]) & (df_fit["year"] <= train_range[1])
    train_df, test_df = df_fit[train_mask].copy(), df_fit[~train_mask].copy()

    if len(train_df) >= 2:
        X_train, y_train = train_df[["year"]], train_df["avg"]
        model = LinearRegression().fit(X_train, y_train)
        if len(test_df) >= 1:
            rmse = float(np.sqrt(mean_squared_error(test_df["avg"], model.predict(test_df[["year"]]))))
            st.metric("RMSE (테스트: 학습구간 밖 연도)", f"{rmse:.3f}")
        a, b = float(model.coef_[0]), float(model.intercept_)
        st.caption(f"회귀식(학습구간 기반): **avg ≈ {a:.4f} × year + {b:.4f}**")

        df_plot = df_fit.copy()
        df_plot["split"] = np.where((df_plot["year"] >= train_range[0]) & (df_plot["year"] <= train_range[1]), "train", "test")
        df_plot["pred"] = model.predict(df_plot[["year"]])

        pts = alt.Chart(df_plot).mark_circle(size=80, opacity=0.9).encode(
            x=alt.X("year:O", title="연도"),
            y=alt.Y("avg:Q", title=f"연평균 {target_col}"),
            color=alt.Color("split:N", scale=alt.Scale(domain=["train","test"], range=["#2E7D32", "#455A64"])),
            tooltip=["year:O", alt.Tooltip("avg:Q", format=".2f"), "split:N"]
        )
        regline = alt.Chart(df_plot).mark_line(color="black").encode(x="year:O", y="pred:Q")
        base_chart = pts + regline

        st.subheader("🔮 미래 예측 (마지막 해만 빨간 점 + 레이블)")
        if last_complete_year is not None:
            start_pred_year = min(max(last_complete_year + 1, min_y), 2100)
            if start_pred_year <= 2100:
                year_to_predict = st.number_input("단일 연도 예측", min_value=int(start_pred_year), max_value=2100,
                                                  value=int(min(start_pred_year + 5, 2100)), step=1)
                single_df = None
                if st.button("해당 연도 예측"):
                    pred_single = float(model.predict(pd.DataFrame({"year": [year_to_predict]}))[0])
                    st.success(f"📌 {year_to_predict}년 예상 {target_col} = **{pred_single:.2f}**")
                    single_df = pd.DataFrame({"year": [year_to_predict], "pred": [pred_single], "label": [f"{pred_single:.2f}"]})

                yr_min, yr_max = int(start_pred_year), 2100
                yr_range = st.slider("예측 구간(연도 범위)", min_value=yr_min, max_value=yr_max,
                                     value=(yr_min, min(yr_min+20, yr_max)), step=1)
                future_years = pd.DataFrame({"year": list(range(yr_range[0], yr_range[1] + 1))})
                future_years["pred"] = model.predict(future_years[["year"]])
                future_years["label"] = future_years["pred"].map(lambda v: f"{v:.2f}")

                chart_future_line = alt.Chart(future_years).mark_line(strokeDash=[5,5], color="gray").encode(
                    x=alt.X("year:O", title="연도"), y=alt.Y("pred:Q", title=f"연평균 {target_col} (예측)")
                )
                last_year = int(future_years["year"].max())
                last_df = future_years[future_years["year"] == last_year]
                last_point = alt.Chart(last_df).mark_point(color="red", size=120).encode(x=alt.X("year:O"), y=alt.Y("pred:Q"))
                last_label = alt.Chart(last_df).mark_text(dy=-14, color="red").encode(x=alt.X("year:O"), y=alt.Y("pred:Q"), text="label:N")

                charts = base_chart + chart_future_line + last_point + last_label
                if single_df is not None:
                    single_point = alt.Chart(single_df).mark_point(color="red", size=120).encode(x=alt.X("year:O"), y=alt.Y("pred:Q"))
                    single_label = alt.Chart(single_df).mark_text(dy=-14, color="red").encode(x=alt.X("year:O"), y=alt.Y("pred:Q"), text="label:N")
                    charts = charts + single_point + single_label
                st.altair_chart(charts, use_container_width=True)
        else:
            st.info("완전한 연도가 없어 예측 범위를 설정하지 못했습니다.")
else:
    st.warning("연평균 유효 연도가 충분하지 않습니다(최소 3년 권장).")

# =========================
# 🌈 K-means로 비지도 계절 구분 (tmin, tmax)
# =========================
st.header("🌈 K-means 비지도 계절 구분 — tmin/tmax 기반")

# tmin/tmax 열 찾기
tmin_candidates = [c for c in ["tmin", "최저기온"] if c in df_daily.columns]
tmax_candidates = [c for c in ["tmax", "최고기온"] if c in df_daily.columns]
if not tmin_candidates or not tmax_candidates:
    st.info("tmin / tmax(또는 최저기온 / 최고기온) 컬럼이 필요합니다.")
else:
    tmin_col, tmax_col = tmin_candidates[0], tmax_candidates[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        k_clusters = st.slider("클러스터 수 K(권장 4)", 3, 6, 4, step=1)
    with col2:
        scaler_opt = st.selectbox("스케일링", ["표준화(Standard)", "MinMax", "없음"], index=0)
    with col3:
        use_complete_years_only = st.checkbox("완전한 연도만 사용(권장)", value=True)

    dfK = df_daily.copy()
    dfK["year"] = dfK["date"].dt.year
    if use_complete_years_only:
        # 완전한 연도만 필터링
        max_dt_all = dfK["date"].dropna().max()
        if pd.notna(max_dt_all):
            last_complete = max_dt_all.year if max_dt_all >= pd.Timestamp(max_dt_all.year, 12, 31) else (max_dt_all.year - 1)
            dfK = dfK[dfK["year"] <= last_complete]

    # 결측 제거
    dfK = dfK.dropna(subset=[tmin_col, tmax_col, "date"]).copy()
    if dfK.empty:
        st.info("클러스터링에 사용할 데이터가 없습니다.")
    else:
        X = dfK[[tmin_col, tmax_col]].to_numpy()
        if scaler_opt == "표준화(Standard)":
            X = StandardScaler().fit_transform(X)
        elif scaler_opt == "MinMax":
            X = MinMaxScaler().fit_transform(X)

        km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        dfK["cluster"] = labels

        # 계절 매핑: 각 클러스터의 평균 (tmin+tmax)/2 기준으로 정렬
        dfK["temp_mean"] = (dfK[tmin_col] + dfK[tmax_col]) / 2
        cluster_means = dfK.groupby("cluster")["temp_mean"].mean().sort_values().reset_index()
        # 낮은→높은: 겨울, 봄, 가을, 여름 (K=4 가정. K≠4인 경우 중간은 ‘중간1/2’로 표시)
        season_names_4 = ["겨울", "봄", "가을", "여름"]
        if k_clusters == 4:
            season_map = {row["cluster"]: season_names_4[i] for i, row in enumerate(cluster_means.itertuples(index=False))}
        else:
            mid_names = [f"중간{i+1}" for i in range(k_clusters-2)]
            dynamic_names = ["추움"] + mid_names + ["더움"]
            season_map = {row["cluster"]: dynamic_names[i] for i, row in enumerate(cluster_means.itertuples(index=False))}

        dfK["season_unsup"] = dfK["cluster"].map(season_map)

        st.subheader("① tmin–tmax 산점도 (클러스터 색) + 센트로이드")
        centroids = pd.DataFrame(km.cluster_centers_, columns=[tmin_col, tmax_col])
        if scaler_opt != "없음":
            # 역스케일링 불가(간단히 원 스케일로 표시하지 않음). 센트로이드는 상대 좌표 설명용 점만 표시.
            centroid_chart = alt.Chart(pd.DataFrame({"cx": [np.nan], "cy": [np.nan]})).mark_point()
            st.caption("※ 스케일링을 적용하면 센트로이드의 절대값은 원 단위가 아니므로 참고용입니다.")
        else:
            centroid_chart = alt.Chart(centroids).mark_point(shape='triangle-up', size=150, color='black')

        scatter = alt.Chart(dfK).mark_point(filled=True, opacity=0.6).encode(
            x=alt.X(f"{tmin_col}:Q", title=f"{tmin_col}"),
            y=alt.Y(f"{tmax_col}:Q", title=f"{tmax_col}"),
            color=alt.Color("season_unsup:N"),
            tooltip=["date:T", f"{tmin_col}:Q", f"{tmax_col}:Q", "season_unsup:N"]
        ).properties(height=360)
        st.altair_chart(scatter, use_container_width=True)

        st.subheader("② 날짜 타임라인(계절 색)")
        timeline = alt.Chart(dfK).mark_bar(height=8).encode(
            x=alt.X("date:T", title="날짜"),
            y=alt.value(10),
            color=alt.Color("season_unsup:N"),
            tooltip=["date:T", "season_unsup:N", alt.Tooltip(f"{tmin_col}:Q", format=".1f"), alt.Tooltip(f"{tmax_col}:Q", format=".1f")]
        ).properties(height=80)
        st.altair_chart(timeline, use_container_width=True)

        st.subheader("③ 연도별 계절 일수 추세 (완전한 연도 기준)")
        # 완전한 연도 기준으로 집계
        max_dt_all = df_daily["date"].dropna().max()
        if pd.isna(max_dt_all):
            st.info("연도 판별을 위한 날짜가 부족합니다.")
        else:
            last_complete = max_dt_all.year if max_dt_all >= pd.Timestamp(max_dt_all.year, 12, 31) else (max_dt_all.year - 1)
            dfKc = dfK[dfK["year"] <= last_complete].copy()
            if dfKc.empty:
                st.info("완전한 연도 구간 내 데이터가 없습니다.")
            else:
                # 시즌 이름 정렬을 위해 평균기온 기준 재정의(여름/겨울 고정)
                season_order = (dfKc.groupby("season_unsup")["temp_mean"].mean()
                                .sort_values()
                                .reset_index()["season_unsup"].tolist())
                # 연도-시즌별 카운트
                counts = dfKc.groupby(["year", "season_unsup"]).size().reset_index(name="days")
                # 라인차트
                line_season = alt.Chart(counts).mark_line(point=True).encode(
                    x=alt.X("year:O", title="연도"),
                    y=alt.Y("days:Q", title="일수"),
                    color=alt.Color("season_unsup:N", sort=season_order),
                    tooltip=["year:O", "season_unsup:N", "days:Q"]
                ).properties(height=360)
                st.altair_chart(line_season, use_container_width=True)

                # 여름 길이 추세 강조(계절명이 '여름'일 때)
                if "여름" in counts["season_unsup"].unique():
                    summer = counts[counts["season_unsup"] == "여름"].copy()
                    if len(summer["year"].unique()) >= 3:
                        # 단순 선형회귀로 추세선(일/년) & 일/10년 지표
                        lr = LinearRegression()
                        Xy = summer[["year"]].astype(int)
                        lr.fit(Xy, summer["days"])
                        slope_per_year = float(lr.coef_[0])
                        slope_per_decade = slope_per_year * 10.0

                        summer["pred"] = lr.predict(Xy)
                        trend = alt.Chart(summer).mark_line(color="red").encode(
                            x="year:O", y="pred:Q"
                        )
                        st.altair_chart(
                            alt.layer(
                                alt.Chart(summer).mark_line(point=True).encode(
                                    x="year:O", y="days:Q", tooltip=["year:O", "days:Q"]
                                ),
                                trend
                            ).properties(title="여름 일수 추세(회귀선)"),
                            use_container_width=True
                        )
                        st.metric("여름 일수 변화(추세선 기울기)", f"{slope_per_year:+.2f} 일/년  ≈  {slope_per_decade:+.1f} 일/10년")
                    else:
                        st.info("여름 일수 추세선을 그리기에 연도 수가 부족합니다.")
                else:
                    st.info("K가 4가 아니거나 데이터 특성상 '여름' 레이블이 존재하지 않을 수 있습니다.")
