# Yahoo Finance(yfinance)를 사용해 비트코인의 일별/분별 OHLCV 데이터를 수집
# GitHub Actions에서 실행되고, 수집한 데이터가 MongoDB에 저장

# 수치 연산 라이브러리 설정
import pandas as pd # 테이블 형태의 데이터를 다루는 라이브러리
import certifi # 몽고DB에 접속 환경 설정 라이브러리
from pymongo import MongoClient # 몽고DB 클러스터 접속 라이브러리
import os # GitHub Actions의 시크릿 값 환경설정 라이브러리
import yfinance as yf # Yahoo Finance에서 금융 데이터를 가져오는 라이브러리
from datetime import date, timedelta # 날짜 연산 라이브러리
import numpy as np # 수치 연산 라이브러리

# -------------------------------------------------------
# MongoDB 연결
# -------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    # 중요: 워크플로/로컬 실행 전 `MONGO_URI` 환경변수를 설정했는지 확인
    raise EnvironmentError("환경변수 MONGO_URI가 설정되지 않았습니다.")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())

print("[INFO] MongoDB에 연결되었습니다.")

db = client["bitcoindb"] # 데이터베이스 클라이언트를 bitcoindb로 설정하여 저장

# 야후파이낸스에서 어떤 데이터를 가져올지 결정하는 TICKER 환경변수 설정
TICKER = os.environ.get("TICKER") or "BTC-USD"

# bitcoindb 클라이언트에 btc_price로 하여 가격 데이터 저장, timestamp가 중복되지 않게 설정
def ensure_collection_index(collection_name: str = "btc_price"): 
    try:
        coll = db[collection_name]
        coll.create_index("timestamp", unique=True)
        print(f"[{collection_name}] timestamp 유니크 인덱스 확인/생성 완료")
    except Exception as e:
        print(f"[{collection_name}] 인덱스 생성 실패:", e)

# 타임스탬프 저장방식 통일화 및 정규화
def _normalize_timestamp(ts, floor_unit: str = "T"):
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

# 어떤 값이 들어오든 안전하게 float(실수)로 바꿔주는 함수
def _as_float(x, default=0.0):
    """값을 안전하게 float로 변환합니다.
    pandas Series나 ndarray가 들어오면 마지막 요소를 사용합니다.
    실패 시 `default`를 반환합니다.
    """
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

# 데이터를 안전하게 원하는 숫자 타입으로 바꿔주는 함수
def _as_int(x, default=0):
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

# yfinance가 반환하는 DataFrame의 컬럼이 다중형태일 때 이를 단순한 형태로 변환
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance의 MultiIndex 컬럼을 평탄화합니다.
    예: ('Open','BTC-USD') -> 'Open'
    이렇게 하면 이후 `row.get('Open')` 방식으로 안전하게 접근할 수 있습니다.
    """
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# 2015년부터 특정 연도까지 쌓여있던 모든 과거 데이터를 한꺼번에 가져와서 저장
def backfill_until_august(start_date=None, year: int | None = None, collection_name: str | None = None):
    coll_name = collection_name or os.environ.get("COLLECTION_NAME") or "btc_price"
    year = year or int(os.environ.get("BACKFILL_YEAR") or date.today().year)
    if start_date is None:
        start_date = os.environ.get("BACKFILL_START") or "2015-01-01"

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()

    end_date = date(year, 8, 31)
    if start_date > end_date:
        print(f"[backfill] 시작일({start_date})이 종료일({end_date})보다 뒤에 있어 작업을 건너뜁니다.")
        return

    # yfinance의 날짜 범위 처리 방식에 맞춰 종료일을 하루 밀어주는 작업
    start_iso = start_date.isoformat()
    end_iso = (end_date + timedelta(days=1)).isoformat()

    print(f"[backfill] {TICKER} 일별 데이터 다운로드: {start_iso} ~ {end_date.isoformat()}")

    # 저장 대상 컬렉션을 준비
    ensure_collection_index(coll_name)
    coll = db[coll_name]

    # 야후파이낸스의 데이터를 우선 다운로드해야 함
    try:
        df = yf.download(TICKER, start=start_iso, end=end_iso, interval="1d", progress=False)
        df = _flatten_columns(df)
    except Exception as e:
        # 일괄 다운로드가 실패하면 더 이상 진행하지 않고 예외를 다시 발생시켜 실행을 중단합니다.
        print(f"[backfill] yf.download 실패: {e}")
        raise

    inserted = 0
    updated = 0

    # 일괄 다운로드가 성공했는지 실패했는지에 경로에 따라 조회
    if df is None or df.empty:
        print("[backfill] 일괄 다운로드 결과 없음 — 날짜별로 개별 조회를 시도합니다.")
        cur = pd.to_datetime(start_date)
        last = pd.to_datetime(end_date)
        while cur <= last:
            day_start = cur.date().isoformat()
            day_end = (cur + timedelta(days=1)).date().isoformat()
            try:
                tdf = yf.download(TICKER, start=day_start, end=day_end, interval="1d", progress=False)
                tdf = _flatten_columns(tdf)
            except Exception as e:
                print(f"[backfill] {day_start} 다운로드 실패: {e}")
                tdf = None

            if tdf is None or tdf.empty:
                # 날짜에 데이터가 없을 수 있음 — 건너뜀
                cur += timedelta(days=1)
                continue

            for idx, row in tdf.iterrows():
                ts = _normalize_timestamp(idx, "D")
                doc = {
                    "timestamp": ts,
                    "open": _as_float(row.get("Open", 0.0)),
                    "high": _as_float(row.get("High", 0.0)),
                    "low": _as_float(row.get("Low", 0.0)),
                    "close": _as_float(row.get("Close", 0.0)),
                    "volume": _as_int(row.get("Volume", 0)),
                }
                try:
                    res = coll.update_one({"timestamp": ts}, {"$set": doc}, upsert=True)
                    if res.upserted_id is not None:
                        inserted += 1
                    else:
                        updated += 1
                except Exception as e:
                    print(f"[backfill] 문서 업서트 실패 ({ts}): {e}")

            cur += timedelta(days=1)
    else:
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
            try:
                res = coll.update_one({"timestamp": ts}, {"$set": doc}, upsert=True)
                if res.upserted_id is not None:
                    inserted += 1
                else:
                    updated += 1
            except Exception as e:
                print(f"[backfill] 문서 업서트 실패 ({ts}): {e}")

    print(f"[backfill] 완료. 삽입: {inserted}, 업데이트: {updated}")


# -------------------------------------------------------
# 메인
# -------------------------------------------------------
if __name__ == "__main__":
    # 스크립트 직접 실행 시 기본 동작 표시
    print("\n===== 백필(8월까지) 업로드 시작 =====")
    # 현재 연도의 8월 31일까지 일별 데이터를 업로드합니다.
    if os.environ.get("BACKFILL_UPTO_AUGUST", "1") not in ("0", "false", "False"):
        backfill_until_august()

    print("\n===== 최신 가격 업로드 시작 =====")
    upload_latest_price()
    print("\n===== 완료 =====")
    # DB 연결 종료
    client.close()