"""Reddit 감성 데이터 수집 → VADER 분석 → MongoDB 일별 업서트 (비트코인 키워드 필터링 적용)"""

import gzip
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import certifi
import requests
from pymongo import MongoClient
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reddit 데이터 수집 설정
CONFIG = {
    "subreddit": "CryptoCurrency", # 수집 대상 Reddit 서브레딧
    "start_date": "2026-01-01",   # MongoDB가 비어 있을 때 최초 시작일
    "batch_size": 100,  # API 한 번 요청 시 가져올 최대 데이터 개수
    "request_delay": 1.0,  # API 요청 간 대기 시간(초)
    "max_retries": 5,  # API 요청 실패 시 최대 재시도 횟수
    "compress": False, # JSONL 파일 gzip 압축 여부
    "collect": "both",  # 수집 대상: submissions(게시물) / comments(댓글) / both(둘 다)
}

# 비트코인 관련 여부를 판단할 키워드 목록
BITCOIN_KEYWORDS = [
    "bitcoin", "btc", "satoshi", "sats", "hodl", "halving",
    "bitcoin cash", "wrapped bitcoin", "wbtc",
    "btc price", "bitcoin price", "bitcoin etf",
    "bitcoin mining", "bitcoin network", "bitcoin dominance",
]
# 위 키워드들을 하나의 정규식으로 결합 (단어 경계 \b 사용, 대소문자 무시)
BITCOIN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BITCOIN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Arctic Shift Reddit 데이터 수집 API 기본 주소
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"

# MongoDB 데이터베이스 및 컬렉션 이름
DB_NAME = "bitcoindb"
COLLECTION_NAME = "redditcompound"

# 날짜 문자열을 UTC 기준 Unix 타임스탬프로 변환
def timestamp(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def get_mongo():
    """환경변수에서 MongoDB URI를 읽어 연결합니다."""
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

# MongoDB 컬렉션에서 가장 최근에 저장된 날짜를 조회
def get_latest_date(collection):
    doc = collection.find_one(sort=[("date", -1)], projection={"date": 1}) # 'date' 필드를 기준으로 가장 최신 데이터 조회
    if not doc or not doc.get("date"):
        return None # 조회된 문서가 없거나 date 값이 없으면 None 반환
    value = doc["date"]  # 조회된 MongoDB 문서에서 date 값 추출
    return datetime.fromisoformat(value).date() if isinstance(value, str) else value.date()  # date 값이 문자열이면 ISO 형식 문자열을 Python date 객체로 변환

# API 요청을 수행하고, 오류 발생 시 설정된 횟수만큼 재시도
def request_json(session, url, params):
    """API 요청 및 재시도."""
    for attempt in range(CONFIG["max_retries"] + 1):
        try:
            response = session.get(url, params=params, timeout=30) # 지정된 URL에 GET 요청, 30초 이상 지연되면 요청을 중단
            if response.status_code == 200:  # HTTP 상태 코드가 200이면 요청 성공
                return response.json()  # API 응답 데이터를 JSON 형식으로 변환하여 반환

            if attempt == CONFIG["max_retries"]:  # 설정된 최대 재시도 횟수까지 모두 실패한 경우
                print(f"HTTP {response.status_code}: {response.text[:200]}") # HTTP 오류 코드와 응답 내용 일부를 출력
                return None  # 더 이상 재시도하지 않고 None 반환

            wait = 60 if response.status_code == 429 else 10 * (attempt + 1)   # HTTP 429 오류는 API 요청 제한(Rate Limit)을 의미
            print(f"HTTP {response.status_code}. {wait}초 후 재시도...")  # 오류 상태 코드와 재시도 전 대기 시간을 출력
            time.sleep(wait)  # 지정된 시간 동안 대기 후 다음 반복에서 재요청
        except requests.RequestException as e:   # 네트워크 연결 오류, Timeout 등의 requests 예외 처리
            if attempt == CONFIG["max_retries"]:  # 최대 재시도 횟수까지 모두 실패한 경우
                print(f"요청 실패: {e}")   # 최종 요청 실패 원인을 출력
                return None    # 더 이상 재시도하지 않고 None 반환
            wait = 10 * (attempt + 1)   # 재시도 횟수가 증가할수록 대기 시간 증가
            print(f"요청 오류. {wait}초 후 재시도: {e}")   # 오류 내용과 재시도 전 대기 시간을 출력
            time.sleep(wait)  # 지정된 시간 동안 대기
    return None   # None 반환

# API 등에서 받아온 원본 데이터를 JSONL 파일로 차곡차곡 저장
def save_jsonl(records, path):
    """원본 데이터를 JSONL로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True) # 저장 폴더 생성
    opener = gzip.open if CONFIG["compress"] else open # 압축 여부에 따라 파일 열기
    mode = "at" # 텍스트 추가 모드
    if CONFIG["compress"]: # 압축 저장 여부 확인
        path = path.with_suffix(path.suffix + ".gz") # .gz 확장자 추가

    with opener(path, mode, encoding="utf-8") as f:  # 파일 열기
        for record in records: # 데이터를 하나씩 가져오기
            f.write(json.dumps(record, ensure_ascii=False) + "\n") # JSONL로 한 줄씩 저장
    return path # 저장된 파일 경로 반환


def extract_text(record, data_type):
    """게시물 또는 댓글에서 감성분석/키워드 판별에 사용할 텍스트를 추출합니다."""
    if data_type == "submissions":  # 게시물인지 확인
        return f"{record.get('title', '')} {record.get('selftext', '')}".strip()  # 게시물의 제목과 본문을 합침
    return (record.get("body") or "").strip()  # 댓글의 본문을 가져옴


def is_bitcoin_related(text):
    """텍스트에 비트코인 관련 키워드가 포함되어 있는지 확인합니다."""
    if not text:
        return False
    # casefold()를 사용해 대소문자 및 일부 유니코드 변이에 강건하게 매칭합니다.
    text_cf = text.casefold()
    for kw in BITCOIN_KEYWORDS:
        if re.search(r"\b" + re.escape(kw.casefold()) + r"\b", text_cf):
            return True
    return False


def analyze_sentiment(record, text, analyzer):  # 감성분석 점수를 계산하는 함수
    """게시물 또는 댓글의 VADER compound 점수를 계산합니다."""
    score = analyzer.polarity_scores(text)["compound"] if text else 0.0  # VADER로 compound 감성점수 계산
    record["_sent_compound"] = score  # 계산된 감성점수를 원본 데이터에 추가
    return score  # 감성점수 반환

# 데이터를 수집하고 감성점수를 집계하는 함수 (비트코인 관련 게시물/댓글만 대상)
def collect_data(data_type, start_ts, end_ts, output_dir, analyzer):
    """Arctic Shift에서 데이터를 수집하고, 비트코인 관련 게시물/댓글만 걸러 날짜별 감성점수를 집계합니다."""
    endpoint = "posts" if data_type == "submissions" else "comments" # 데이터 유형에 따라 게시물 또는 댓글 API 선택
    url = f"{ARCTIC_BASE}/{endpoint}/search" # API 요청 주소 생성
    output_path = output_dir / f"{data_type}_{CONFIG['subreddit']}_bitcoin.jsonl" # 수집 데이터를 저장할 파일 경로 설정

    totals = defaultdict(float)  # 날짜별 감성점수 합계 (비트코인 관련만)
    counts = defaultdict(int)    # 날짜별 게시물/댓글 수 (비트코인 관련만)
    current = start_ts           # 현재 조회 시작 시각
    total = 0                    # 서브레딧 전체 수집 개수
    matched = 0                  # 비트코인 관련으로 필터링된 개수

    session = requests.Session() # HTTP 요청 세션 생성
    session.headers["User-Agent"] = "RedditSentimentResearch/1.0" # API 요청에 사용할 User-Agent 설정

    with tqdm(desc=f"{data_type}", unit="개") as bar:  # 데이터 수집 진행 상황 표시
        while current < end_ts:  # 종료 시간까지 반복 수집
            params = {
                "subreddit": CONFIG["subreddit"], # 수집할 서브레딧 지정
                "after": current,  # 조회 시작 시각
                "before": end_ts,  # 조회 종료 시각
                "limit": CONFIG["batch_size"],  # 한 번에 가져올 데이터 수
                "sort": "asc",  # 오름차순 정렬
                "sort_type": "created_utc",  # 생성 시각 기준 정렬
            }
            data = request_json(session, url, params)  # API 요청 및 JSON 응답 파싱
            records = data.get("data", []) if data else []  # 응답에서 데이터 추출
            if not records:  # 데이터가 없으면 종료
                break

            bitcoin_records = []  # 이번 배치에서 비트코인 관련으로 판별된 레코드만 모음
            for record in records:
                text = extract_text(record, data_type)  # 감성분석/필터링용 텍스트 추출
                if not is_bitcoin_related(text):  # 비트코인 관련 키워드가 없으면 건너뜀
                    continue

                score = analyze_sentiment(record, text, analyzer)  # 비트코인 관련 텍스트만 감성분석
                created = record.get("created_utc")
                if created is not None:
                    day = datetime.fromtimestamp(int(created), tz=timezone.utc).date().isoformat()
                    totals[day] += score
                    counts[day] += 1
                bitcoin_records.append(record)

            if bitcoin_records:
                save_jsonl(bitcoin_records, output_path)  # 비트코인 관련 레코드만 파일로 저장
            matched += len(bitcoin_records)
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
    print(f"{data_type}: 전체 {total:,}개 중 비트코인 관련 {matched:,}개 수집")
    return totals, counts


def upload_daily_sentiment(collection, totals, counts):
    """비트코인 관련 게시물과 댓글을 합친 날짜별 평균 감성점수를 MongoDB에 저장합니다."""
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
        print(f"대상: r/{CONFIG['subreddit']} (비트코인 관련 키워드 필터링 적용)")
        print(f"필터 키워드: {', '.join(BITCOIN_KEYWORDS)}")

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