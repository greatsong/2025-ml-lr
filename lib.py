# lib.py
import os, glob
import pandas as pd
import numpy as np

# ---------- 파일 로딩 ----------
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
    # 마지막 시도(인코딩 지정 안 함)
    return pd.read_csv(
        file_or_path,
        skiprows=range(skip_top_rows) if skip_top_rows > 0 else None
    )

# ---------- 전처리 ----------
def to_numeric_strict(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^-\d\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def normalize_and_parse(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # '날짜' -> 'date'
    if "date" not in df.columns and "날짜" in df.columns:
        df = df.rename(columns={"날짜": "date"})
    if "date" not in df.columns:
        return df  # 호출부에서 에러 처리

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]:
        if col in df.columns:
            df[col] = to_numeric_strict(df[col])
    return df

# ---------- 연평균(완전 연도) ----------
def compute_yearly_mean(df_daily, target_col, miss_threshold=0.02):
    """
    연도별 target_col 연평균 (완전 연도만 유효).
    완전 연도: 그 해 12/31 데이터가 존재하는 마지막 연도 이하.
    miss_threshold: 연간 결측비율 초과 시 NaN 처리.
    """
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
            out.append({"year": y, "avg": np.nan}); continue

        g2 = g.set_index("date")[[target_col]].rename(columns={target_col: "val"})
        merged = full.join(g2, how="left")
        miss_ratio = merged["val"].isna().sum() / len(merged)
        avg_val = np.nan if miss_ratio > miss_threshold else merged["val"].mean(skipna=True)
        out.append({"year": y, "avg": avg_val})

    df_year = pd.DataFrame(out).sort_values("year").reset_index(drop=True)
    df_year = df_year[df_year["year"] <= last_complete_year].reset_index(drop=True)
    return df_year, last_complete_year
