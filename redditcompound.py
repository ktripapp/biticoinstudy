"""Reddit 감성 데이터 수집 → VADER 분석 → MongoDB 일별 업서트"""

import gzip
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import certifi
import requests
from pymongo import MongoClient
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CONFIG = {
    "subreddit": "CryptoCurrency",
    "start_date": "2017-08-18",   # MongoDB가 비어 있을 때 최초 시작일
    "batch_size": 100,
    "request_delay": 1.0,
    "max_retries": 5,
    "compress": False,
    "collect": "both",            # submissions / comments / both
}

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
DB_NAME = "bitcoindb"
COLLECTION_NAME = "redditcompound"


def timestamp(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def get_mongo():
    """local_secrets.py 또는 환경변수에서 MongoDB URI를 읽어 연결합니다."""
    uri = None
    try:
        import local_secrets
        uri = getattr(local_secrets, "MONGO_URI", None)
    except ImportError:
        pass

    uri = uri or os.getenv("MONGO_URI") or os.getenv("MONGO_URI_SECRET")
    if not uri:
        raise RuntimeError("MONGO_URI가 없습니다.")

    client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client, client[DB_NAME][COLLECTION_NAME]


def get_latest_date(collection):
    doc = collection.find_one(sort=[("date", -1)], projection={"date": 1})
    if not doc or not doc.get("date"):
        return None
    value = doc["date"]
    return datetime.fromisoformat(value).date() if isinstance(value, str) else value.date()


def request_json(session, url, params):
    """API 요청 및 재시도."""
    for attempt in range(CONFIG["max_retries"] + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()

            if attempt == CONFIG["max_retries"]:
                print(f"HTTP {response.status_code}: {response.text[:200]}")
                return None

            wait = 60 if response.status_code == 429 else 10 * (attempt + 1)
            print(f"HTTP {response.status_code}. {wait}초 후 재시도...")
            time.sleep(wait)
        except requests.RequestException as e:
            if attempt == CONFIG["max_retries"]:
                print(f"요청 실패: {e}")
                return None
            wait = 10 * (attempt + 1)
            print(f"요청 오류. {wait}초 후 재시도: {e}")
            time.sleep(wait)
    return None


def save_jsonl(records, path):
    """원본 데이터를 JSONL로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if CONFIG["compress"] else open
    mode = "at"
    if CONFIG["compress"]:
        path = path.with_suffix(path.suffix + ".gz")

    with opener(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def analyze_sentiment(record, data_type, analyzer):
    """게시물 또는 댓글의 VADER compound 점수를 계산합니다."""
    if data_type == "submissions":
        text = f"{record.get('title', '')} {record.get('selftext', '')}".strip()
    else:
        text = (record.get("body") or "").strip()

    score = analyzer.polarity_scores(text)["compound"] if text else 0.0
    record["_sent_compound"] = score
    return score


def collect_data(data_type, start_ts, end_ts, output_dir, analyzer):
    """Arctic Shift에서 데이터를 수집하고 날짜별 감성점수를 동시에 집계합니다."""
    endpoint = "posts" if data_type == "submissions" else "comments"
    url = f"{ARCTIC_BASE}/{endpoint}/search"
    output_path = output_dir / f"{data_type}_{CONFIG['subreddit']}.jsonl"

    totals = defaultdict(float)
    counts = defaultdict(int)
    current = start_ts
    total = 0

    session = requests.Session()
    session.headers["User-Agent"] = "RedditSentimentResearch/1.0"

    with tqdm(desc=f"{data_type}", unit="개") as bar:
        while current < end_ts:
            params = {
                "subreddit": CONFIG["subreddit"],
                "after": current,
                "before": end_ts,
                "limit": CONFIG["batch_size"],
                "sort": "asc",
                "sort_type": "created_utc",
            }
            data = request_json(session, url, params)
            records = data.get("data", []) if data else []
            if not records:
                break

            for record in records:
                score = analyze_sentiment(record, data_type, analyzer)
                created = record.get("created_utc")
                if created is not None:
                    day = datetime.fromtimestamp(int(created), tz=timezone.utc).date().isoformat()
                    totals[day] += score
                    counts[day] += 1

            save_jsonl(records, output_path)
            total += len(records)
            bar.update(len(records))

            last_ts = int(records[-1]["created_utc"])
            if last_ts < current:  # API 비정상 응답 방지
                break
            current = last_ts + 1

            if len(records) < CONFIG["batch_size"]:
                break
            time.sleep(CONFIG["request_delay"])

    session.close()
    print(f"{data_type}: {total:,}개 수집")
    return totals, counts


def upload_daily_sentiment(collection, totals, counts):
    """게시물과 댓글을 합친 날짜별 평균 감성점수를 MongoDB에 저장합니다."""
    uploaded = 0
    for day in sorted(totals):
        count = counts[day]
        if count == 0:
            continue

        document = {
            "date": day,
            "compound": round(totals[day] / count, 4),
            "count": count,
            "uploaded_at": datetime.now(timezone.utc),
        }
        collection.update_one({"date": day}, {"$set": document}, upsert=True)
        uploaded += 1

    print(f"MongoDB 업서트 완료: {uploaded}일")


def main():
    client = None
    try:
        client, collection = get_mongo()
        latest = get_latest_date(collection)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        start_date = latest + timedelta(days=1) if latest else datetime.strptime(CONFIG["start_date"], "%Y-%m-%d").date()

        if start_date > yesterday:
            print("수집할 새로운 데이터가 없습니다.")
            return

        start_ts = timestamp(start_date.isoformat())
        end_ts = timestamp((yesterday + timedelta(days=1)).isoformat())
        output_dir = Path(__file__).resolve().parent / "bitcoin_reddit_data" / f"reddit_{start_date:%Y%m%d}_{yesterday:%Y%m%d}"

        print(f"수집 기간: {start_date} ~ {yesterday} UTC")
        print(f"대상: r/{CONFIG['subreddit']}")

        analyzer = SentimentIntensityAnalyzer()
        totals = defaultdict(float)
        counts = defaultdict(int)

        types = ["submissions", "comments"] if CONFIG["collect"] == "both" else [CONFIG["collect"]]
        for data_type in types:
            daily_totals, daily_counts = collect_data(data_type, start_ts, end_ts, output_dir, analyzer)
            for day, value in daily_totals.items():
                totals[day] += value
                counts[day] += daily_counts[day]

        upload_daily_sentiment(collection, totals, counts)
        print("전체 작업 완료")

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
