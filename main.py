import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 분류 모델
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# 회귀 모델
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="ML 교육용 웹앱: 분류+회귀+EDA", layout="wide")

# ===============================
# 사이드바: 교육 안내
# ===============================
with st.sidebar:
    st.title("🎓 교육용 안내")
    st.markdown("""
**학습 흐름**
1) 상단 탭에서 **분류(아이리스)** 또는 **회귀(기온 예측)** 선택  
2) **데이터 소스**: 내장 샘플 또는 CSV 업로드 (기온 CSV는 cp949 권장)  
3) **EDA**(요약/분포/관계/상관) 먼저 보기  
4) **모델/옵션** 고르고 **학습/평가**

**TIP**  
- EDA로 데이터의 패턴·이상치·스케일을 확인하세요.  
- 성능이 낮다면: 특성 선택, 스케일 조정, 모델 변경을 시도!
    """)

st.title("🧪 간단한 머신러닝 교육용 웹앱 (Streamlit Cloud)")

# ===============================
# 유틸 함수
# ===============================
def smart_read_csv(file_or_path, skip_top_rows=0):
    """cp949 → utf-8-sig → utf-8 → euc-kr 순으로 시도, 상단 N행 스킵 지원"""
    encodings = ["cp949", "utf-8-sig", "utf-8", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                file_or_path,
                encoding=enc,
                skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
            )
        except Exception as e:
            last_err = e
            continue
    # 마지막 시도 (인코딩 지정 없이)
    return pd.read_csv(
        file_or_path,
        skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
    )

def render_eda(df, title_suffix=""):
    st.subheader(f"🔎 EDA(탐색적 데이터 분석){' - ' + title_suffix if title_suffix else ''}")

    with st.expander("1) 데이터 개요 (행/열, 타입, 결측치)"):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.metric("행(Row) 수", f"{len(df):,}")
        with c2:
            st.metric("열(Column) 수", f"{df.shape[1]:,}")
        with c3:
            num_cols = df.select_dtypes(include=np.number).shape[1]
            st.metric("숫자형 특성 수", f"{num_cols:,}")

        st.write("**데이터 타입**")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

        st.write("**결측치 현황**")
        miss = df.isna().sum()
        miss_df = miss[miss>0].to_frame("missing_count")
        if miss_df.empty:
            st.success("결측치 없음 ✅")
        else:
            st.warning("결측치가 있습니다. 아래 열을 확인하세요.")
            st.dataframe(miss_df)

    with st.expander("2) 미리보기(Head/Tail)"):
        st.write("**Head(상위 5행)**")
        st.dataframe(df.head())
        st.write("**Tail(하위 5행)**")
        st.dataframe(df.tail())

    with st.expander("3) 숫자형 통계 요약"):
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            st.dataframe(num_df.describe().T)
        else:
            st.info("숫자형 컬럼이 없습니다.")

    with st.expander("4) 분포 탐색(히스토그램)"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col = st.selectbox("히스토그램으로 볼 숫자형 특성 선택", num_cols, key=f"hist_{title_suffix}")
            bins = st.slider("구간 개수(Bins)", 10, 60, 30, key=f"bins_{title_suffix}")
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(alt.repeat("column"), type='quantitative', bin=alt.Bin(maxbins=bins)),
                y='count()'
            ).properties(width=500, height=300).repeat(column=[col])
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("숫자형 컬럼이 없습니다.")

    with st.expander("5) 산점도(두 특성 관계)"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            x = st.selectbox("X축", num_cols, key=f"x_{title_suffix}")
            y = st.selectbox("Y축", [c for c in num_cols if c != x], key=f"y_{title_suffix}")
            color_col = st.selectbox("색상 기준(선택)", ["(없음)"] + df.columns.tolist(), key=f"color_{title_suffix}")
            enc = {"x": alt.X(x, type='quantitative'), "y": alt.Y(y, type='quantitative')}
            if color_col != "(없음)":
                enc["color"] = color_col
            chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(**enc).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("산점도를 그리려면 최소 2개의 숫자형 컬럼이 필요합니다.")

    with st.expander("6) 상관관계(숫자형) 히트맵"):
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            corr_reset = corr.reset_index().melt('index')
            corr_reset.columns = ['feature_x', 'feature_y', 'corr']
            heat = alt.Chart(corr_reset).mark_rect().encode(
                x=alt.X('feature_x:O', sort=None),
                y=alt.Y('feature_y:O', sort=None),
                tooltip=['feature_x', 'feature_y', alt.Tooltip('corr:Q', format=".2f")],
                color=alt.Color('corr:Q', scale=alt.Scale(domain=[-1,1]))
            ).properties(height=400)
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info("숫자형 컬럼이 2개 이상일 때 상관 히트맵을 그릴 수 있습니다.")

def build_xy(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def make_preprocessor(X, scale_numeric=True, onehot_categorical=True):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numeric_features:
        if scale_numeric:
            transformers.append(("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features))
        else:
            transformers.append(("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]), numeric_features))

    if categorical_features and onehot_categorical:
        transformers.append(("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features))

    if not transformers:
        return None

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre

# ===============================
# 내장 데이터 로더
# ===============================
def load_builtin_iris():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(data.target_names)))
    return df, "species"

def load_temperature_from_any(skip_top_rows=7):
    """data/daily_temp.csv → daily_temp.csv → 없으면 시뮬레이션 생성
       - cp949 우선, 상단 N행 스킵, date 파싱 시도
    """
    import os
    candidates = ["data/daily_temp.csv", "daily_temp.csv"]
    for path in candidates:
        if os.path.exists(path):
            df = smart_read_csv(path, skip_top_rows)
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                except Exception:
                    pass
            return df

    # 시뮬레이션 (교육용)
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-01", periods=365, freq="D")
    day_of_year = np.arange(365)
    base = 15 + 10*np.sin(2*np.pi*(day_of_year/365))  # 계절성
    weekday = [d.weekday() for d in days]
    weekday_effect = np.array([0,0.3,0.2,0.1,0,-0.1,-0.2])[weekday]
    noise = rng.normal(0, 1.5, size=365)
    temp = base + weekday_effect + noise
    df = pd.DataFrame({
        "date": days,
        "temp": temp.round(2),
        "humidity": np.clip(60 + 10*np.sin(2*np.pi*(day_of_year/365+0.1)) + rng.normal(0, 3, 365), 30, 95).round(1),
        "wind": np.clip(2 + rng.normal(0, 1, 365), 0, None).round(2),
    })
    return df

# ===============================
# 결측치 처리
# ===============================
def apply_missing_strategy(df, target_col=None, strategy="auto"):
    """
    strategy: auto | time | linear | ffill | bfill | median | drop
    - auto(권장): date 존재하면 'time', 없으면 수치=median, 범주=최빈값
    - time: date 기준 정렬 후 시간보간 → 남은 결측 ffill/bfill
    - linear: 수치형 선형보간 → 남은 결측 ffill/bfill, 범주=최빈값
    - ffill/bfill: 단순 전/후방 채움
    - median: 수치=중앙값, 범주=최빈값
    - drop: 결측 포함 행 삭제
    """
    df = df.copy()

    # date 파싱 보정
    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass

    # 자동 규칙 결정
    if strategy == "auto":
        if "date" in df.columns and df["date"].notna().any():
            strategy = "time"
        else:
            strategy = "median"

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if strategy == "time" and "date" in df.columns and df["date"].notna().any():
        df = df.sort_values("date")
        if num_cols:
            try:
                # 시간보간: 인덱스를 시간으로 설정 후 보간하는 접근
                df = df.set_index("date")
                df[num_cols] = df[num_cols].interpolate(method="time")
                df[num_cols] = df[num_cols].ffill().bfill()
                df = df.reset_index()
            except Exception:
                # 실패 시 일반 보간
                df[num_cols] = df[num_cols].interpolate()
                df[num_cols] = df[num_cols].ffill().bfill()
        # 범주형 최빈값
        for c in cat_cols:
            if df[c].isna().any():
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c].fillna(mode.iloc[0], inplace=True)

    elif strategy == "linear":
        if num_cols:
            df[num_cols] = df[num_cols].interpolate()
            df[num_cols] = df[num_cols].ffill().bfill()
        for c in cat_cols:
            if df[c].isna().any():
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c].fillna(mode.iloc[0], inplace=True)

    elif strategy == "ffill":
        df = df.ffill()

    elif strategy == "bfill":
        df = df.bfill()

    elif strategy == "median":
        for c in num_cols:
            if df[c].isna().any():
                med = df[c].median()
                df[c].fillna(med, inplace=True)
        for c in cat_cols:
            if df[c].isna().any():
                mode = df[c].mode(dropna=True)
                if not mode.empty:
                    df[c].fillna(mode.iloc[0], inplace=True)

    elif strategy == "drop":
        df = df.dropna()

    return df

# ===============================
# 탭: 분류 / 회귀
# ===============================
tab1, tab2 = st.tabs(["🌸 분류: 아이리스", "🌤️ 회귀: 기온 예측"])

# -------------------------------
# TAB 1: 분류(아이리스)
# -------------------------------
with tab1:
    st.header("🌸 아이리스 품종 분류 (Classification)")

    src = st.radio("데이터 소스 선택", ["내장(아이리스)", "CSV 업로드"], horizontal=True, key="cls_src")
    if src == "CSV 업로드":
        up = st.file_uploader("CSV 업로드(타깃 포함)", type=["csv"], key="cls_up")
        if up:
            df = smart_read_csv(up, skip_top_rows=0)
            target_col = st.selectbox("타깃(분류할 정답) 컬럼 선택", options=df.columns, key="cls_target")
        else:
            st.info("CSV를 업로드하면 자동으로 미리보기/EDA가 표시됩니다. (미업로드 시 내장 사용)")
            df, target_col = load_builtin_iris()
    else:
        df, target_col = load_builtin_iris()

    render_eda(df, "분류")

    st.subheader("⚙️ 모델 설정 & 학습")
    with st.expander("교수자 설명: 분류란?", expanded=True):
        st.markdown("""
**분류(Classification)**는 데이터가 어떤 **범주(클래스)**에 속하는지를 예측하는 문제입니다.  
예) 꽃의 길이/너비로 **품종**을 맞히기.
        """)

    X, y = build_xy(df, target_col)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, step=0.05, key="cls_test")
    with c2:
        scale_num = st.checkbox("숫자형 스케일링(StandardScaler)", True, key="cls_scale")
    with c3:
        onehot_cat = st.checkbox("문자형 원-핫인코딩", True, key="cls_ohe")

    model_name = st.selectbox("모델 선택", ["LogisticRegression", "KNN", "RandomForest"], key="cls_model")

    if model_name == "LogisticRegression":
        C = st.slider("정규화 세기(C, 작을수록 강함)", 0.01, 3.0, 1.0, 0.01, key="cls_C")
        clf = LogisticRegression(max_iter=1000, C=C)
    elif model_name == "KNN":
        k = st.slider("이웃 수(k)", 1, 30, 5, key="cls_k")
        clf = KNeighborsClassifier(n_neighbors=k)
    else:
        n = st.slider("트리 개수", 50, 500, 200, 50, key="cls_n")
        depth = st.slider("최대 깊이(0=제한없음)", 0, 20, 0, key="cls_depth")
        clf = RandomForestClassifier(
            n_estimators=n,
            max_depth=None if depth==0 else depth,
            random_state=42
        )

    pre = make_preprocessor(X, scale_numeric=scale_num, onehot_categorical=onehot_cat)
    pipe = Pipeline([("pre", pre), ("model", clf)]) if pre is not None else Pipeline([("model", clf)])

    if st.button("🟢 학습/평가 실행", key="cls_train"):
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("정확도(Accuracy)", f"{acc*100:.2f}%")

        # 혼동행렬
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in np.unique(y)],
                                columns=[f"pred_{c}" for c in np.unique(y)])
        st.write("**혼동행렬(Confusion Matrix)**")
        cm_long = cm_df.reset_index().melt("index")
        cm_long.columns = ["true", "pred", "count"]
        heat = alt.Chart(cm_long).mark_rect().encode(
            x="pred:O", y="true:O", color=alt.Color("count:Q"),
            tooltip=["true","pred","count"]
        ).properties(height=300)
        st.altair_chart(heat, use_container_width=True)

        # 분류 리포트(표)
        st.write("**분류 리포트 (Precision/Recall/F1)**")
        rep = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(rep).T)

# -------------------------------
# TAB 2: 회귀(기온 예측)
# -------------------------------
with tab2:
    st.header("🌤️ 기온 예측 (Regression)")

    st.markdown("**CSV 상단 불필요 행 스킵 및 결측치 처리 옵션**")
    skip_n = st.slider("상단 스킵할 행 수", 0, 20, 7,
                       help="요청사항: 1~7행은 불필요하므로 기본 7행 스킵")
    miss_strategy = st.selectbox(
        "결측치 처리 전략",
        ["auto(권장)", "time(시간보간)", "linear(선형보간)", "ffill(전방채움)", "bfill(후방채움)", "median(중앙값 대치)", "drop(결측행 삭제)"],
        index=0
    )
    miss_strategy_key = miss_strategy.split("(")[0]

    src = st.radio("데이터 소스 선택", ["내장(기온/시뮬레이션 또는 data/daily_temp.csv)", "CSV 업로드"], horizontal=True, key="reg_src")
    if src == "CSV 업로드":
        up = st.file_uploader("CSV 업로드(타깃 포함) - cp949 권장", type=["csv"], key="reg_up")
        if up:
            raw = smart_read_csv(up, skip_top_rows=skip_n)
        else:
            st.info("CSV를 업로드하면 미리보기/EDA가 표시됩니다. (미업로드 시 내장/시뮬레이션 사용)")
            raw = load_temperature_from_any(skip_top_rows=skip_n)
    else:
        raw = load_temperature_from_any(skip_top_rows=skip_n)

    # date 파싱 시도
    if "date" in raw.columns:
        try:
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        except Exception:
            pass

    # 타깃 자동 탐색(일반적인 이름 우선)
    target_candidates = [c for c in ["temp", "temperature", "target", "y"] if c in raw.columns]
    if target_candidates:
        target_reg = target_candidates[0]
    else:
        target_reg = st.selectbox("타깃(예측할 값) 컬럼 선택", options=raw.columns, key="reg_target_sel")

    # 결측치 전/후 비교
    st.caption("결측치 처리 전/후 비교")
    c_before = int(raw.isna().sum().sum())
    st.write(f"• 처리 전 결측치 총합: **{c_before}**")

    df_reg = apply_missing_strategy(raw, target_col=target_reg, strategy=miss_strategy_key)
    c_after = int(df_reg.isna().sum().sum())
    st.write(f"• 처리 후 결측치 총합: **{c_after}**")

    # EDA
    render_eda(df_reg, "회귀")

    # 모델 설정
    st.subheader("⚙️ 모델 설정 & 학습")
    with st.expander("교수자 설명: 회귀란?", expanded=True):
        st.markdown("""
**회귀(Regression)**는 **연속적인 수치**를 예측하는 문제입니다.  
예) 날짜/습도/바람 등의 정보로 **기온**을 예측하기.
        """)

    # 날짜 파생변수 옵션
    if "date" in df_reg.columns and np.issubdtype(df_reg["date"].dtype, np.datetime64):
        add_time_feats = st.checkbox("날짜에서 파생변수 추가(월/요일)", True, key="reg_timefe")
        if add_time_feats:
            tmp = df_reg.copy()
            tmp["month"] = tmp["date"].dt.month
            tmp["weekday"] = tmp["date"].dt.weekday
            df_reg = tmp

    Xr, yr = build_xy(df_reg, target_reg)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        test_size_r = st.slider("테스트 비율", 0.1, 0.4, 0.2, step=0.05, key="reg_test")
    with c2:
        scale_num_r = st.checkbox("숫자형 스케일링(StandardScaler)", False, key="reg_scale")
    with c3:
        onehot_cat_r = st.checkbox("문자형 원-핫인코딩", True, key="reg_ohe")

    model_name_r = st.selectbox("모델 선택", ["LinearRegression", "Ridge", "Lasso", "KNN", "RandomForest"], key="reg_model")

    if model_name_r == "LinearRegression":
        mdl = LinearRegression()
    elif model_name_r == "Ridge":
        alpha = st.slider("alpha(규제 세기)", 0.01, 10.0, 1.0, 0.01, key="ridge_a")
        mdl = Ridge(alpha=alpha, random_state=42)
    elif model_name_r == "Lasso":
        alpha = st.slider("alpha(규제 세기)", 0.001, 1.0, 0.01, 0.001, key="lasso_a")
        mdl = Lasso(alpha=alpha, random_state=42, max_iter=5000)
    elif model_name_r == "KNN":
        k = st.slider("이웃 수(k)", 1, 30, 5, key="knn_k")
        mdl = KNeighborsRegressor(n_neighbors=k)
    else:
        n = st.slider("트리 개수", 50, 500, 200, 50, key="rf_n")
        depth = st.slider("최대 깊이(0=제한없음)", 0, 20, 0, key="rf_d")
        mdl = RandomForestRegressor(
            n_estimators=n,
            max_depth=None if depth==0 else depth,
            random_state=42
        )

    pre_r = make_preprocessor(Xr, scale_numeric=scale_num_r, onehot_categorical=onehot_cat_r)
    pipe_r = Pipeline([("pre", pre_r), ("model", mdl)]) if pre_r is not None else Pipeline([("model", mdl)])

    if st.button("🟢 학습/평가 실행", key="reg_train"):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=test_size_r, random_state=42)
        pipe_r.fit(X_train, y_train)
        pred = pipe_r.predict(X_test)

        rmse = mean_squared_error(y_test, pred, squared=False)
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE(낮을수록 좋음)", f"{rmse:.3f}")
        c2.metric("MAE(낮을수록 좋음)", f"{mae:.3f}")
        c3.metric("R²(높을수록 좋음)", f"{r2:.3f}")

        # 예측 vs 실제 산점도
        st.write("**예측 vs 실제 산점도**")
        df_plot = pd.DataFrame({"y_true": y_test, "y_pred": pred})
        sc = alt.Chart(df_plot).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("y_true:Q", title="실제값"),
            y=alt.Y("y_pred:Q", title="예측값"),
            tooltip=["y_true","y_pred"]
        ).properties(height=350)
        # y=x 기준선
        line_min = float(df_plot.min().min())
        line_max = float(df_plot.max().max())
        base_line = pd.DataFrame({"y": [line_min, line_max]})
        line = alt.Chart(base_line).mark_line().encode(x="y:Q", y="y:Q")
        st.altair_chart(sc + line, use_container_width=True)

        # 잔차 분포
        st.write("**잔차(실제-예측) 분포**")
        resid = y_test - pred
        df_res = pd.DataFrame({"residual": resid})
        hist = alt.Chart(df_res).mark_bar().encode(
            x=alt.X("residual:Q", bin=alt.Bin(maxbins=40)),
            y="count()"
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)

        # 시계열 비교: date가 있으면
        if "date" in df_reg.columns and np.issubdtype(df_reg["date"].dtype, np.datetime64):
            st.write("**시계열 비교(테스트셋)**")
            part = df_reg.loc[y_test.index, ["date"]].copy()
            part["y_true"] = y_test.values
            part["y_pred"] = pred
            part = part.sort_values("date")
            line_true = alt.Chart(part).mark_line().encode(
                x="date:T", y=alt.Y("y_true:Q", title="값"), tooltip=["date","y_true"]
            )
            line_pred = alt.Chart(part).mark_line().encode(
                x="date:T", y="y_pred:Q", tooltip=["date","y_pred"]
            )
            st.altair_chart(line_true + line_pred, use_container_width=True)

# 푸터(교육 메모)
st.markdown("---")
st.markdown("""
**교육 메모**  
- **EDA 우선**: 데이터 구조·분포·관계를 먼저 파악하면 전처리와 모델 선택이 쉬워집니다.  
- **분류**는 클래스(범주) 예측, **회귀**는 연속값 예측입니다.  
- 결측치는 **시간보간(시계열)**, **중앙값 대치(범용)**처럼 맥락에 맞게 처리하세요.  
- 업로드 CSV도 동일 흐름으로 실습할 수 있도록 설계돼 있습니다.
""")
