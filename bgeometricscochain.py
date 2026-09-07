# BGeometrics Scalar API에서 온체인 시계열을 가져와 MongoDB에 저장합니다.

from datetime import date, datetime
import os
import random
import time
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


def exponential_backoff(attempt: int) -> float:
    """attempt(1부터 시작)에 따른 지수 백오프 대기 시간(초)."""
    return (2 ** (attempt - 1)) + random.random()


def build_params(startday: str, endday: str):
    params = {"startday": startday, "endday": endday}
    if BGEOMETRICS_TOKEN:
        # BGeometrics(api.bitcoin-data.com) 문서는 헤더 인증을 기본으로 안내하지만,
        # 과거에는 쿼리 파라미터 token도 허용했습니다. 두 방식 모두 보내서
        # 어느 쪽이 맞는지 API 쪽에서 알아서 인식하게 하고, 로그로 어떤 방식이
        # 실제로 통과했는지 나중에 확인할 수 있게 합니다.
        params["token"] = BGEOMETRICS_TOKEN
    return params


def build_headers():
    headers = {}
    if BGEOMETRICS_TOKEN:
        headers["Authorization"] = f"Bearer {BGEOMETRICS_TOKEN}"
        headers["x-api-key"] = BGEOMETRICS_TOKEN
    return headers


DATE_TEXT_KEYS = ("date", "day", "d", "x")
EPOCH_KEYS = ("unixTs", "unix_ts", "timestamp", "time")
DATE_KEY_CANDIDATES = DATE_TEXT_KEYS + EPOCH_KEYS
VALUE_KEY_CANDIDATES = ("value", "v", "y", "val")


def normalize_timeseries_item(item, endpoint: str = None):
    if isinstance(item, dict):
        date_keys = [k for k in item.keys() if k in DATE_KEY_CANDIDATES]
        value_keys = [k for k in item.keys() if k in VALUE_KEY_CANDIDATES]

        d = item.get(date_keys[0]) if date_keys else None
        # 엔드포인트 이름 자체가 값 필드 키인 경우가 많음 (예: {"d": "...", "sopr": 1.02})
        if endpoint and endpoint in item:
            v = item.get(endpoint)
        elif value_keys:
            v = item.get(value_keys[0])
        else:
            v = None

        # 텍스트 날짜가 없고 epoch 형태만 있는 경우 날짜 문자열로 변환
        if d is None:
            for k in EPOCH_KEYS:
                if item.get(k) is not None:
                    try:
                        d = datetime.utcfromtimestamp(int(item[k])).strftime("%Y-%m-%d")
                    except Exception:
                        pass
                    break

        return d, v, item
    return None, None, item


def _safe_float(x):
    """실패하면 조용히 None을 반환하는 float 변환."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_numeric_value(raw_item):
    """Try to extract a numeric value from a raw item. Returns float or None."""
    if raw_item is None:
        return None
    if isinstance(raw_item, (int, float, str)):
        return _safe_float(raw_item)
    if isinstance(raw_item, dict):
        # 우선 흔히 쓰이는 값 필드 이름부터 확인
        for k in ("value", "v", "y", "val", "price"):
            if k in raw_item:
                v = _safe_float(raw_item[k])
                if v is not None:
                    return v
        # 없으면 dict 안의 첫 번째 숫자형 필드를 사용
        for v in raw_item.values():
            if isinstance(v, (int, float)):
                return float(v)
    return None


def fetch_endpoint(session: requests.Session, base_url: str, endpoint: str, startday: str, endday: str):
    url = f"{base_url.rstrip('/')}/{endpoint}"
    params = build_params(startday, endday)
    headers = build_headers()

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            # Network-level error: retry
            if attempt == max_retries:
                raise
            backoff = exponential_backoff(attempt)
            print(f"Network error on attempt {attempt}/{max_retries} for {endpoint}: {e}. Backing off {backoff:.1f}s")
            time.sleep(backoff)
            continue

        if resp.status_code == 429:
            # Rate limited: respect Retry-After if present, otherwise exponential backoff
            retry_after = resp.headers.get("Retry-After")
            wait = _safe_float(retry_after) if retry_after is not None else None
            if wait is None:
                wait = exponential_backoff(attempt)
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
            backoff = exponential_backoff(attempt)
            print(f"Server error {resp.status_code} on attempt {attempt}/{max_retries} for {endpoint}. Backing off {backoff:.1f}s")
            time.sleep(backoff)
            continue

        # For other HTTP errors, raise immediately
        resp.raise_for_status()
        return resp.json()

    return None


def extract_items(data, endpoint: str):
    """API 응답에서 시계열 항목 리스트를 찾아 반환. 못 찾으면 None."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("result"), dict) and isinstance(data["result"].get("data"), list):
            # {"result": {"data": [...]}} 같은 중첩 형태도 지원
            return data["result"]["data"]
        if isinstance(data.get(endpoint), list):
            # {"sopr": [...]} 같은 형태도 지원
            return data[endpoint]
    return None


def backfill_until_august():
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

    # Log DRY_RUN state and whether key env vars are actually populated (마스킹)
    print(f"DRY_RUN={DRY_RUN}")
    print(f"BGEOMETRICS_TOKEN set: {bool(BGEOMETRICS_TOKEN)}")
    print(f"BGEOMETRICS_BASE_URL={BGEOMETRICS_BASE_URL}")
    print(f"MONGO_URI set: {bool(MONGO_URI)} (length={len(MONGO_URI) if MONGO_URI else 0})")

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

            # 진단용: 실제 응답 구조를 항상 로그로 남긴다.
            # (키 이름이 예상과 다르면 여기서 바로 드러남)
            preview = str(data)[:500]
            print(f"[{ep}] response type={type(data).__name__}, preview={preview}")

            items = extract_items(data, ep)

            if items is None:
                print(f"[{ep}] WARNING: unrecognized response shape, skipped (nothing saved). "
                      f"응답 구조가 예상과 달라 이 엔드포인트는 저장되지 않았습니다. 위 preview를 확인하세요.")
                continue

            ops, skipped = 0, 0
            for it in items:
                d, v, raw = normalize_timeseries_item(it, endpoint=ep)
                if not d:
                    skipped += 1
                    continue
                # 우선 엔드포인트/명시 값 필드(v)를 사용하고, 없으면 raw 전체에서 숫자를 탐색
                numeric = extract_numeric_value(v) if v is not None else None
                if numeric is None:
                    numeric = extract_numeric_value(raw)

                filter_q = {"date": str(d)}
                set_fields = {ep: numeric, f"raw.{ep}": raw, "fetched_at": datetime.utcnow()}
                update = {"$set": set_fields}
                if DRY_RUN:
                    print(f"[DRY RUN] would upsert {filter_q} -> set {set_fields}")
                else:
                    coll.update_one(filter_q, update, upsert=True)
                ops += 1

            print(f"[{ep}] merged/updated {ops} documents into {coll.full_name} "
                  f"({skipped} items skipped due to missing date field)")
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