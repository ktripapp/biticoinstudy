# BGeometrics Scalar API에서 온체인 시계열을 가져와 MongoDB에 저장합니다.

from datetime import date, datetime
import os
import sys
import requests
import certifi
from pymongo import MongoClient
from pymongo.errors import ConfigurationError


# 설정
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise EnvironmentError("환경변수 MONGO_URI가 설정되지 않았습니다.")

BGEOMETRICS_TOKEN = os.environ.get("BGEOMETRICS_TOKEN")
BGEOMETRICS_BASE_URL = os.environ.get("BGEOMETRICS_BASE_URL", "https://api.bitcoin-data.com/v1")

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes")

ENDPOINTS = [
    "sopr",
    "mvrv",
    "nupl",
    "aviv",
    "active-addresses",
    "nvm-ratio",
    "receiver-addresses",
    "sender-addresses",
]

SINGLE_COLLECTION = os.environ.get("SINGLE_COLLECTION", "onchain")


def compute_default_end():
    return date.today()


def subtract_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year - years)


def build_params(startday: str, endday: str):
    params = {"startday": startday, "endday": endday}
    if BGEOMETRICS_TOKEN:
        params["token"] = BGEOMETRICS_TOKEN
    return params


def normalize_timeseries_item(item):
    if isinstance(item, dict):
        date_keys = [k for k in item.keys() if k in ("date", "day", "x")]
        value_keys = [k for k in item.keys() if k in ("value", "v", "y")]
        d = item.get(date_keys[0]) if date_keys else item.get("date") or item.get("day")
        v = item.get(value_keys[0]) if value_keys else None
        return d, v, item
    return None, None, item


def extract_numeric_value(raw_item):
    """Try to extract a numeric value from a raw item. Returns float or None."""
    if raw_item is None:
        return None
    # direct numeric
    if isinstance(raw_item, (int, float)):
        try:
            return float(raw_item)
        except Exception:
            return None
    # string that might be numeric
    if isinstance(raw_item, str):
        try:
            return float(raw_item)
        except Exception:
            return None
    # dict-like: check common keys then any numeric field
    if isinstance(raw_item, dict):
        for k in ("value", "v", "y", "val", "price"):
            if k in raw_item:
                try:
                    return float(raw_item[k])
                except Exception:
                    pass
        for k, v in raw_item.items():
            if isinstance(v, (int, float)):
                try:
                    return float(v)
                except Exception:
                    continue
    return None


def fetch_endpoint(session: requests.Session, base_url: str, endpoint: str, startday: str, endday: str):
    import time
    import random

    url = f"{base_url.rstrip('/')}/{endpoint}"
    params = build_params(startday, endday)

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            # Network-level error: retry
            if attempt == max_retries:
                raise
            backoff = (2 ** (attempt - 1)) + random.random()
            print(f"Network error on attempt {attempt}/{max_retries} for {endpoint}: {e}. Backing off {backoff:.1f}s")
            time.sleep(backoff)
            continue

        if resp.status_code == 429:
            # Rate limited: respect Retry-After if present, otherwise exponential backoff
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after is not None else (2 ** (attempt - 1)) + random.random()
            except Exception:
                wait = (2 ** (attempt - 1)) + random.random()
            print(f"Received 429 for {endpoint} (attempt {attempt}/{max_retries}). Waiting {wait:.1f}s before retrying.")
            if attempt == max_retries:
                print(f"Max retries reached for {endpoint} after rate limiting.")
                return None
            time.sleep(wait)
            continue

        if 500 <= resp.status_code < 600:
            # Server error: retry
            if attempt == max_retries:
                resp.raise_for_status()
            backoff = (2 ** (attempt - 1)) + random.random()
            print(f"Server error {resp.status_code} on attempt {attempt}/{max_retries} for {endpoint}. Backing off {backoff:.1f}s")
            time.sleep(backoff)
            continue

        # For other HTTP errors, raise immediately
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            raise

        return resp.json()

    return None


def backfill_until_august(collection_prefix: str = "bgeometrics"):
    end_date = compute_default_end()
    start_date = subtract_years(end_date, 4)
    startday = start_date.strftime("%Y-%m-%d")
    endday = end_date.strftime("%Y-%m-%d")

    print(f"백필 기간: {startday} 부터 {endday} (엔드포인트: {', '.join(ENDPOINTS)})")

    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    # 안전하게 기본 DB를 얻고, 없으면 'bitcoindb'로 폴백
    try:
        db = client.get_default_database()
        if db is None:
            raise ConfigurationError("No default database")
    except ConfigurationError:
        db = client["bitcoindb"]

    # Log DRY_RUN state
    print(f"DRY_RUN={DRY_RUN}")

    # Check MongoDB connectivity (ping) before fetching data
    try:
        client.admin.command('ping')
        print("MongoDB ping: OK")
    except Exception as e:
        print("MongoDB ping failed:", e)
        # If not dry run, abort early so workflow fails loud and clear
        if not DRY_RUN:
            raise

    session = requests.Session()

    coll = db[SINGLE_COLLECTION]
    # date 단일 인덱스(유니크)
    try:
        coll.create_index("date", unique=True)
    except Exception:
        pass

    try:
        for ep in ENDPOINTS:
            print(f"-> Fetching endpoint: {ep}")
            data = fetch_endpoint(session, BGEOMETRICS_BASE_URL, ep, startday, endday)
            if data is None:
                print(f"Skipping endpoint {ep} due to repeated errors/rate limiting.")
                continue

            items = None
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                items = data["data"]
            else:
                # 배열이 아닌 경우, 모든 항목을 단일 문서의 raw 필드로 저장
                doc = {"fetched_at": datetime.utcnow(), f"raw.{ep}": data}
                # 날짜를 알 수 없으면 insert_one로 저장하지 않고 로그 출력
                print(f"[{ep}] non-array response: stored under raw.{ep} in documents when date exists")
                continue

            ops = 0
            for it in items:
                d, v, raw = normalize_timeseries_item(it)
                if not d:
                    continue
                # 우선 raw에서 숫자를 추출하고, normalize에서 추출된 v보다 우선시
                numeric = extract_numeric_value(raw)
                if numeric is None and v is not None:
                    numeric = extract_numeric_value(v)

                filter_q = {"date": str(d)}
                set_fields = {ep: numeric, f"raw.{ep}": raw, "fetched_at": datetime.utcnow()}
                update = {"$set": set_fields}
                if DRY_RUN:
                    print(f"[DRY RUN] would upsert {filter_q} -> set {set_fields}")
                else:
                    coll.update_one(filter_q, update, upsert=True)
                ops += 1

            print(f"[{ep}] merged/updated {ops} documents into {coll.full_name}")
        # Post-run summary
        try:
            if DRY_RUN:
                print("DRY_RUN enabled — no DB writes performed.")
            else:
                total = coll.count_documents({})
                print(f"Post-run: collection '{coll.full_name}' contains {total} documents")
                sample = coll.find_one({}, projection={"_id": 0})
                if sample:
                    print("Sample document:", sample)
        except Exception as e:
            print("Failed to produce post-run summary:", e)

    finally:
        client.close()


if __name__ == "__main__":
    # 환경 변수로 비활성화 가능
    if os.environ.get("BACKFILL_UPTO_AUGUST", "1") not in ("0", "false", "False"):
        backfill_until_august()
    else:
        print("BACKFILL_UPTO_AUGUST is disabled; exiting.")
