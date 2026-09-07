import pandas as pd
import certifi
from pymongo import MongoClient
import os
import yfinance as yf
from datetime import date, timedelta
import numpy as np

# -------------------------------------------------------
# 간소화된: 최근 저장일 기준으로 일별 업서트만 수행
# -------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise EnvironmentError("환경변수 MONGO_URI가 설정되지 않았습니다.")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client.get_database("bitcoindb")

# 기본 티커 (환경변수 우선)
TICKER = os.environ.get("TICKER") or "BTC-USD"


def ensure_collection_index(collection_name: str = "btc_price"):
    # `timestamp`에 대한 유니크 인덱스가 존재하는지 확인하고 없으면 생성합니다
    coll = db[collection_name]
    coll.create_index("timestamp", unique=True)


def _normalize_timestamp(ts, floor_unit: str = "T"):
    # pandas 타임스탬프를 일(D) 단위로 내림하여 naive datetime으로 반환합니다
    pts = pd.to_datetime(ts)
    if getattr(pts, "tz", None) is not None:
        try:
            pts = pts.tz_convert("UTC")
        except Exception:
            try:
                pts = pts.tz_localize("UTC")
            except Exception:
                pass
    pts = pts.floor(floor_unit)
    py = pts.to_pydatetime()
    if getattr(py, "tzinfo", None) is not None:
        py = py.replace(tzinfo=None)
    return py


def _as_float(x, default=0.0):
    # 값 x를 float로 변환합니다. pandas Series/ndarray인 경우 마지막 요소를 사용합니다.
    if x is None:
        return default
    if isinstance(x, (pd.Series, pd.DataFrame)):
        try:
            x = x.iloc[-1]
        except Exception:
            return default
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            x = x[-1]
        except Exception:
            return default
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x, default=0):
    # 값 x를 int로 변환합니다. pandas Series/ndarray인 경우 마지막 요소를 사용합니다.
    if x is None:
        return default
    if isinstance(x, (pd.Series, pd.DataFrame)):
        try:
            x = x.iloc[-1]
        except Exception:
            return default
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            x = x[-1]
        except Exception:
            return default
    try:
        return int(x)
    except Exception:
        return default


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance가 반환하는 MultiIndex 컬럼을 평탄화합니다 (단일 티커 안전 처리)
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

# 컬렉션에서 가장 최근 타임스탬프를 찾아 날짜(date)로 반환합니다
def get_latest_date(collection_name: str = None):
    coll_name = collection_name or os.environ.get("COLLECTION_NAME") or "btc_price"
    coll = db[coll_name]
    doc = coll.find_one(sort=[("timestamp", -1)])
    if not doc:
        return None
    ts = doc.get("timestamp")
    if hasattr(ts, "date"):
        return ts.date()
    try:
        return pd.to_datetime(ts).date()
    except Exception:
        return None

# 컬렉션에서 가장 최근 타임스탬프를 찾아 다음 날부터 어제까지의 일별 데이터를 업서트합니다
def upsert_from_last(collection_name: str | None = None):
    coll_name = collection_name or os.environ.get("COLLECTION_NAME") or "btc_price"
    latest = get_latest_date(coll_name)
    if latest is None:
        start_date = pd.to_datetime(os.environ.get("BACKFILL_START") or "2015-01-01").date()
    else:
        start_date = latest + timedelta(days=1)

    end_date = date.today() - timedelta(days=1)
    if start_date > end_date:
        print(f"[{coll_name}] 최신(또는 어제)까지 이미 업서트되어 있음: {start_date} > {end_date}")
        return

    start_iso = start_date.isoformat()
    end_iso = (end_date + timedelta(days=1)).isoformat()
    print(f"[{coll_name}] 일별 데이터 다운로드: {start_iso} ~ {end_date.isoformat()}")

    df = yf.download(TICKER, start=start_iso, end=end_iso, interval="1d", progress=False)
    df = _flatten_columns(df)
    if df is None or df.empty:
        print(f"[{coll_name}] 다운로드된 데이터가 없습니다.")
        return

    ensure_collection_index(coll_name)
    coll = db[coll_name]
    inserted = 0
    updated = 0
    for idx, row in df.iterrows():
        ts = _normalize_timestamp(idx, "D")
        doc = {
            "timestamp": ts,
            "open": _as_float(row.get("Open", 0.0)),
            "high": _as_float(row.get("High", 0.0)),
            "low": _as_float(row.get("Low", 0.0)),
            "close": _as_float(row.get("Close", 0.0)),
            "volume": _as_int(row.get("Volume", 0)),
        }
        res = coll.update_one({"timestamp": ts}, {"$set": doc}, upsert=True)
        if res.upserted_id is not None:
            inserted += 1
        else:
            updated += 1

    print(f"[{coll_name}] 업서트 완료. 삽입: {inserted}, 업데이트: {updated}")


if __name__ == "__main__":
    # 기본 동작: 최근 저장일 기준으로 어제까지 일별 데이터를 업서트합니다
    print("\n===== 최근 저장일 기준 일별 업서트 시작 =====")
    upsert_from_last()
    print("\n===== 완료 =====")
    client.close()