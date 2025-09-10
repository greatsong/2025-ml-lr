# pages/1_회귀.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from lib import find_latest_csv, smart_read_csv, normalize_and_parse, compute_yearly_mean

st.set_page_config(page_title="📈 연평균 회귀", layout="wide")
st.title("📈 연평균 회귀")

with st.sidebar:
    st.header("데이터")
    skip_n = st.slider("상단 스킵 행 수", 0, 20, 7)
    src = st.radio("데이터 소스", ["기본(최신 CSV 자동)", "CSV 업로드"])
    miss_threshold = st.number_input("연간 결측 임계값(비율)", 0.0, 1.0, 0.02, 0.01)

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

num_cols = df.select_dtypes(include=np.number).columns.tolist()
target_choices = [c for c in ["평균기온(℃)","최고기온(℃)","최저기온(℃)"] if c in df.columns] or num_cols
target_col = st.selectbox("연평균 지표", options=target_choices, index=0)

df_year, last_complete_year = compute_yearly_mean(df, target_col=target_col, miss_threshold=miss_threshold)
st.subheader("연평균 테이블(완전한 연도)")
st.dataframe(df_year)

df_fit = df_year.dropna(subset=["avg"]).copy()
if len(df_fit) < 3:
    st.warning("유효 연도가 3년 이상 필요")
    st.stop()

min_y, max_y = int(df_fit["year"].min()), int(df_fit["year"].max())
train_range = st.slider("학습 범위", min_value=min_y, max_value=max_y, value=(min_y, max_y), step=1)
train_mask = (df_fit["year"] >= train_range[0]) & (df_fit["year"] <= train_range[1])
train_df, test_df = df_fit[train_mask].copy(), df_fit[~train_mask].copy()

if len(train_df) < 2:
    st.warning("학습 구간은 최소 2개 연도 필요")
    st.stop()

model = LinearRegression().fit(train_df[["year"]], train_df["avg"])
if len(test_df) >= 1:
    rmse = float(np.sqrt(mean_squared_error(test_df["avg"], model.predict(test_df[["year"]]))))
    st.metric("RMSE(테스트)", f"{rmse:.3f}")

df_plot = df_fit.copy()
df_plot["split"] = np.where((df_plot["year"] >= train_range[0]) & (df_plot["year"] <= train_range[1]), "train", "test")
df_plot["pred"] = model.predict(df_plot[["year"]])

base = alt.Chart(df_plot).mark_circle(size=80).encode(
    x="year:O", y=alt.Y("avg:Q", title=f"연평균 {target_col}"), color="split:N",
    tooltip=["year:O", alt.Tooltip("avg:Q", format=".2f"), "split:N"]
) + alt.Chart(df_plot).mark_line(color="black").encode(x="year:O", y="pred:Q")
st.altair_chart(base, use_container_width=True)

# 예측
if last_complete_year is None:
    st.info("완전한 마지막 연도 없음 → 예측 생략")
else:
    start_pred_year = min(max(int(last_complete_year) + 1, min_y), 2100)
    yr_min, yr_max = start_pred_year, 2100
    yr_range = st.slider("예측 구간", min_value=yr_min, max_value=yr_max,
                         value=(yr_min, min(yr_min+20, yr_max)), step=1)
    fut = pd.DataFrame({"year": list(range(yr_range[0], yr_range[1]+1))})
    fut["pred"] = model.predict(fut[["year"]])
    fut["label"] = fut["pred"].map(lambda v: f"{v:.2f}")

    line = alt.Chart(fut).mark_line(strokeDash=[5,5], color="gray").encode(x="year:O", y="pred:Q")
    last = fut.iloc[[-1]]
    last_pt = alt.Chart(last).mark_point(color="red", size=120).encode(x="year:O", y="pred:Q")
    last_tx = alt.Chart(last).mark_text(dy=-14, color="red").encode(x="year:O", y="pred:Q", text="label:N")
    st.altair_chart(base + line + last_pt + last_tx, use_container_width=True)

st.markdown("---")
st.caption("교육 메모: 회귀는 연평균만 사용하며, 예측은 **완전한 마지막 연도 다음 해부터** 가능합니다.")
