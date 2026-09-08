"""
Reddit 비트코인 데이터 수집기
Arctic Shift 혹은 Pushshift API를 사용하여 특정 서브레딧(r/Bitcoin 등)의 게시물과 댓글을 기간 단위로 수집합니다.

지원 엔드포인트:
    - Arctic Shift API (https://arctic-shift.photon-reddit.com)  ← 현재 권장 및 가장 안정적
    - Pushshift API (https://api.pushshift.io) ← 백업용(최근 접근 제한이 있을 수 있음)

사용법 예시:
    pip install -r requirements.txt
    python redditcompound.py
"""

import requests
import json
import time
import os
import gzip
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import certifi
from pymongo import MongoClient


# ─────────────────────────────────────────────
# ⚙️  설정
# ─────────────────────────────────────────────
CONFIG = {
    # 수집 대상 서브레딧
    "subreddit": "CryptoCurrency",

    # 수집 기간 (UTC 기준) — CONFIG["start_date"]에 "YYYY-MM-DD" 형식으로 반드시 지정해야 합니다.
    # end_date는 main()에서 항상 "어제 날짜"로 자동 설정됩니다 (지정해도 무시됨).
    # "start_date": "2024-02-16",

    # 저장 경로 — main() 실행 시 날짜 범위를 기준으로 자동 계산되어 덮어써지므로 여기서 지정할 필요 없음.

    # 한 번 요청당 가져올 항목 수 (Arctic Shift 최대 100) 한꺼번에 많이 가져와지지 않음.
    "batch_size": 100,

    # 요청 간격 (초) — rate limit 방지
    "request_delay": 1.0,

    # 재시도 횟수, 안가져와질 때 시도를 계속 해야 함.
    "max_retries": 5,

    # 데이터를 gzip으로 압축 저장 여부
    "compress": False,

    # CSV가 이미 존재하면 이어서 붙일지 여부
    "append_csv": False,
    # JSONL 저장 여부 (True -> JSONL(.jsonl or .jsonl.gz), False -> CSV)
    "save_jsonl": True,
    # 체크포인트(JSON) 사용 여부
    "use_checkpoints": False,

    # 수집할 항목: "submissions"(게시물), "comments"(댓글), "both"
    "collect": "both",
    # 감성분석 활성화
    "do_sentiment": True,

    # 체크포인트를 무시하고 지정한 start_date부터 강제로 수집하려면 True로 설정
    # (기본 False — 기존 체크포인트가 있으면 그 시점부터 재개합니다)
    "ignore_checkpoints": False,
    # 생성 여부: 병합(merged) CSV 파일 생성 여부. False면 생성하지 않음.
    "create_merged_csv": True,
}

# 확장 키워드(빈 문자열이면 서브레딧 전체를 의미합니다)
CONFIG["queries"] = [
    "",                 # 빈 문자열 = 서브레딧 전체 (default)
    "bitcoin",
    "btc",
    "bitcoin price",
    "btcusd",
]

# Arctic Shift API 엔드포인트
ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"

# Pushshift API 엔드포인트 (백업용)
PUSHSHIFT_BASE = "https://api.pushshift.io/reddit/search"


# ─────────────────────────────────────────────
# 🛠️  유틸리티 함수
# ─────────────────────────────────────────────

def to_utc_timestamp(date_str: str) -> int:
    """날짜 문자열(YYYY-MM-DD)을 UTC Unix timestamp로 변환."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def format_ts(ts: int) -> str:
    """Unix timestamp를 읽기 쉬운 날짜 문자열로 변환."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def save_jsonl(records: list, filepath: Path, compress: bool = True):
    """레코드 리스트를 JSONL (또는 .jsonl.gz) 파일에 저장."""
    if compress:
        filepath = filepath.with_suffix('.jsonl.gz')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(filepath, 'at', encoding='utf-8') as f:
            for r in records:
                sent_raw = r.get('_sent_compound') if r is not None else None
                try:
                    sent_val = float(sent_raw) if sent_raw is not None else None
                except Exception:
                    sent_val = None
                if sent_val is not None and sent_val == 0.0:
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'a', encoding='utf-8') as f:
            for r in records:
                sent_raw = r.get('_sent_compound') if r is not None else None
                try:
                    sent_val = float(sent_raw) if sent_raw is not None else None
                except Exception:
                    sent_val = None
                if sent_val is not None and sent_val == 0.0:
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return filepath


def load_checkpoint(path: Path) -> int:
    """체크포인트 파일에서 마지막 수집된 timestamp를 읽습니다."""
    if not CONFIG.get('use_checkpoints', False):
        return 0
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            return data.get('last_timestamp', 0)
    return 0


def save_checkpoint(path: Path, timestamp: int):
    """현재까지 수집된 마지막 timestamp를 저장합니다."""
    if not CONFIG.get('use_checkpoints', False):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'last_timestamp': timestamp, 'updated': format_ts(timestamp)}, f)


def sanitize_query(q: str) -> str:
    """파일명/체크포인트용으로 쿼리 문자열을 안전하게 변환합니다."""
    if not q:
        return "all"
    s = q.strip().lower()
    # 안전한 문자만 남기고 공백을 '_'로 바꿈
    for ch in " ":
        s = s.replace(ch, "_")
    # 제거할 문자
    keep = []
    for c in s:
        if c.isalnum() or c in "-_":
            keep.append(c)
    out = "".join(keep)
    return out or "q"


def get_mongo_client_and_collection(uri: str | None, db_name: str = "bitcoindb", coll_name: str = "redditcompound"):
    """MongoDB에 연결을 시도하고(핑 테스트 포함) (client, collection) 또는 (None, None)을 반환합니다.

    실패 시 원인 확인을 위해 상세 에러를 출력합니다.
    """
    if not uri:
        return None, None
    try:
        client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        # quick connectivity test
        client.admin.command('ping')
        db = client[db_name]
        coll = db.get_collection(coll_name)
        print(f"Connected to MongoDB: {db_name}.{coll_name}")
        return client, coll
    except Exception as e:
        print(f"MongoDB connection/test failed: {e}")
        return None, None


# ─────────────────────────────────────────────
# 🌐  Arctic Shift API 호출
# ─────────────────────────────────────────────

class ArcticShiftCollector:
    """
    Arctic Shift API를 사용해 Reddit 데이터를 수집합니다.
    문서: https://arctic-shift.photon-reddit.com/api-docs
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "RedditBitcoinResearcher/1.0 (academic data collection)"
        })
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # sentiment analyzer
        self.analyzer = None
        if self.cfg.get("do_sentiment", False):
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"⚠️ Sentiment analyzer 초기화 실패: {e}")
                self.analyzer = None

    def _get(self, url: str, params: dict, retries: int = 0) -> dict | None:
        """GET 요청 with retry 로직."""
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 60 * (retries + 1)
                print(f"\n⚠️  Rate limited. {wait}초 대기 후 재시도...")
                time.sleep(wait)
                return self._get(url, params, retries + 1)
            if resp.status_code != 200:
                print(f"\n❌ HTTP {resp.status_code}: {resp.text[:200]}")
                if retries < self.cfg["max_retries"]:
                    time.sleep(10 * (retries + 1))
                    return self._get(url, params, retries + 1)
                return None
            return resp.json()
        except requests.RequestException as e:
            print(f"\n❌ 요청 오류: {e}")
            if retries < self.cfg["max_retries"]:
                time.sleep(10 * (retries + 1))
                return self._get(url, params, retries + 1)
            return None

    def collect_submissions(self, query: str = ""):
        """게시물(submissions) 수집. `query`가 비어있으면 서브레딧 전체를 수집합니다."""
        qname = sanitize_query(query)
        print(f"\n📄 게시물(Submissions) 수집 시작... 쿼리: '{query or 'ALL'}'")

        start_ts = to_utc_timestamp(self.cfg["start_date"])
        # Treat end_date as inclusive: convert to midnight UTC of the next day
        end_ts   = to_utc_timestamp(self.cfg["end_date"]) + 24 * 3600
        subreddit = self.cfg["subreddit"]

        # 체크포인트 로드 (중단된 경우 이어서 수집)
        ckpt_path = self.output_dir / f"submissions_checkpoint_{qname}.json"
        resume_ts = load_checkpoint(ckpt_path)
        if resume_ts > start_ts and not self.cfg.get("ignore_checkpoints", False):
            print(f"  ↩️  체크포인트 감지: {format_ts(resume_ts)} 부터 재개")
            start_ts = resume_ts
        elif resume_ts > start_ts and self.cfg.get("ignore_checkpoints", False):
            print(f"  ⚠️ 체크포인트({format_ts(resume_ts)}) 존재하지만 ignore_checkpoints=True 이므로 무시하고 {format_ts(start_ts)}부터 시작합니다")

        out_path = self.output_dir / f"submissions_{subreddit}_{qname}.jsonl"
        total_collected = 0
        current_after = start_ts

        url = f"{ARCTIC_BASE}/posts/search"

        with tqdm(desc="  게시물", unit="개") as pbar:
            while current_after < end_ts:
                params = {
                    "subreddit": subreddit,
                    "after":     current_after,
                    "before":    end_ts,
                    "limit":     self.cfg["batch_size"],
                    "sort":      "asc",
                    "sort_type": "created_utc",
                }

                if query:
                    params["q"] = query

                data = self._get(url, params)
                if not data:
                    break

                posts = data.get("data", [])
                if not posts:
                    break

                # 감성분석 추가 (선택적)
                if self.analyzer is not None:
                    for p in posts:
                        title = p.get("title", "") or ""
                        selftext = p.get("selftext", "") or ""
                        text = (title + " \n" + selftext).strip()
                        try:
                            if text:
                                scores = self.analyzer.polarity_scores(text)
                            else:
                                scores = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
                        except Exception:
                            scores = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
                        p["_sent_compound"] = scores.get("compound")
                        p["_sent_pos"] = scores.get("pos")
                        p["_sent_neu"] = scores.get("neu")
                        p["_sent_neg"] = scores.get("neg")
                        # label
                        c = p["_sent_compound"]
                        if c >= 0.05:
                            lab = "positive"
                        elif c <= -0.05:
                            lab = "negative"
                        else:
                            lab = "neutral"
                        p["_sent_label"] = lab

                # 저장
                saved = save_jsonl(posts, out_path, self.cfg["compress"])
                total_collected += len(posts)
                pbar.update(len(posts))

                # 다음 페이지: 마지막 항목의 created_utc + 1
                last_ts = posts[-1]["created_utc"]
                save_checkpoint(ckpt_path, last_ts)
                current_after = last_ts + 1

                if len(posts) < self.cfg["batch_size"]:
                    break  # 마지막 페이지

                time.sleep(self.cfg["request_delay"])

        saved = locals().get('saved', out_path)
        print(f"  ✅ 게시물 수집 완료: 총 {total_collected:,}개 → {saved}")
        return total_collected

    def collect_comments(self):
        """댓글(comments) 수집."""
        print("\n💬 댓글(Comments) 수집 시작...")

        start_ts  = to_utc_timestamp(self.cfg["start_date"])
        # Treat end_date as inclusive: convert to midnight UTC of the next day
        end_ts    = to_utc_timestamp(self.cfg["end_date"]) + 24 * 3600
        subreddit = self.cfg["subreddit"]

        ckpt_path = self.output_dir / "comments_checkpoint.json"
        resume_ts = load_checkpoint(ckpt_path)
        if resume_ts > start_ts and not self.cfg.get("ignore_checkpoints", False):
            print(f"  ↩️  체크포인트 감지: {format_ts(resume_ts)} 부터 재개")
            start_ts = resume_ts
        elif resume_ts > start_ts and self.cfg.get("ignore_checkpoints", False):
            print(f"  ⚠️ 체크포인트({format_ts(resume_ts)}) 존재하지만 ignore_checkpoints=True 이므로 무시하고 {format_ts(start_ts)}부터 시작합니다")

        out_path = self.output_dir / f"comments_{subreddit}.jsonl"
        total_collected = 0
        current_after = start_ts

        url = f"{ARCTIC_BASE}/comments/search"

        with tqdm(desc="  댓글", unit="개") as pbar:
            while current_after < end_ts:
                params = {
                    "subreddit": subreddit,
                    "after":     current_after,
                    "before":    end_ts,
                    "limit":     self.cfg["batch_size"],
                    "sort":      "asc",
                    "sort_type": "created_utc",
                }

                data = self._get(url, params)
                if not data:
                    break

                comments = data.get("data", [])
                if not comments:
                    break

                # 감성분석 추가 (선택적)
                if self.analyzer is not None:
                    for c in comments:
                        body = c.get("body", "") or ""
                        text = body.strip()
                        try:
                            if text:
                                scores = self.analyzer.polarity_scores(text)
                            else:
                                scores = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
                        except Exception:
                            scores = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
                        c["_sent_compound"] = scores.get("compound")
                        c["_sent_pos"] = scores.get("pos")
                        c["_sent_neu"] = scores.get("neu")
                        c["_sent_neg"] = scores.get("neg")
                        # label
                        cc = c["_sent_compound"]
                        if cc >= 0.05:
                            lab = "positive"
                        elif cc <= -0.05:
                            lab = "negative"
                        else:
                            lab = "neutral"
                        c["_sent_label"] = lab

                saved = save_jsonl(comments, out_path, self.cfg["compress"])
                total_collected += len(comments)
                pbar.update(len(comments))

                last_ts = comments[-1]["created_utc"]
                save_checkpoint(ckpt_path, last_ts)
                current_after = last_ts + 1

                if len(comments) < self.cfg["batch_size"]:
                    break

                time.sleep(self.cfg["request_delay"])

        saved = locals().get('saved', out_path)
        print(f"  ✅ 댓글 수집 완료: 총 {total_collected:,}개 → {saved}")
        return total_collected


# ─────────────────────────────────────────────
# 🌐  Pushshift API (백업용)
# ─────────────────────────────────────────────

class PushshiftCollector:
    """
    Pushshift API를 이용한 백업 수집기입니다.
    참고: 2023년 이후 공개 Pushshift API는 접근이 제한될 수 있습니다. (권한이 필요할 수 있음)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "RedditBitcoinResearcher/1.0"
        })
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict) -> list:
        url = f"{PUSHSHIFT_BASE}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            print(f"Pushshift 오류: {e}")
            return []

    def collect(self, data_type: str = "submission"):
        """data_type: 'submission' 또는 'comment'"""
        subreddit = self.cfg["subreddit"]
        start_ts  = to_utc_timestamp(self.cfg["start_date"])
        # Treat end_date as inclusive: convert to midnight UTC of the next day
        end_ts    = to_utc_timestamp(self.cfg["end_date"]) + 24 * 3600

        out_path = self.output_dir / f"pushshift_{data_type}s_{subreddit}.jsonl"
        total = 0
        current_after = start_ts

        print(f"\n[Pushshift] {data_type} 수집 중...")
        while current_after < end_ts:
            params = {
                "subreddit": subreddit,
                "after":     current_after,
                "before":    end_ts,
                "size":      self.cfg["batch_size"],
                "sort":      "asc",
                "sort_type": "created_utc",
            }
            items = self._get(data_type, params)
            if not items:
                break

            save_jsonl(items, out_path, self.cfg["compress"])
            total += len(items)
            current_after = items[-1]["created_utc"] + 1
            print(f"  {total:,}개 수집됨... ({format_ts(current_after)})")
            time.sleep(self.cfg["request_delay"])

        print(f"[Pushshift] 완료: {total:,}개")
        return total


# ─────────────────────────────────────────────
# 📊  수집 후 통계 출력
# ─────────────────────────────────────────────

def print_stats(output_dir: str, compress: bool):
    """수집된 데이터 파일 통계를 출력합니다."""
    output_dir = Path(output_dir)
    if not CONFIG.get('save_jsonl', False):
        ext = '.csv'
    else:
        ext = '.jsonl.gz' if compress else '.jsonl'
    files = list(output_dir.glob(f"*{ext}"))

    print("\n" + "═" * 50)
    print("📊 수집 결과 요약")
    print("═" * 50)

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        # 라인 수 (레코드 수) 계산
        count = 0
        try:
            if f.suffix == '.csv':
                with open(f, 'r', encoding='utf-8', newline='') as fh:
                    reader = csv.reader(fh)
                    # subtract header
                    count = sum(1 for _ in reader) - 1
            else:
                opener = gzip.open if compress else open
                with opener(f, 'rt', encoding='utf-8') as fh:
                    for _ in fh:
                        count += 1
        except Exception:
            count = -1
        print(f"  📁 {f.name}")
        print(f"      레코드 수: {count:,}개  |  파일 크기: {size_mb:.1f} MB")

    print("═" * 50)
    print(f"  저장 위치: {output_dir.resolve()}")
    print("═" * 50)


# ─────────────────────────────────────────────
# 🚀  메인 실행
# ─────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   Reddit Bitcoin 데이터 수집기               ║")
    print("║   Arctic Shift / Pushshift 방식              ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  대상 서브레딧: r/{CONFIG['subreddit']}")
    print(f"  수집 항목:    {CONFIG['collect']}")
    print()

    # MongoDB에서 최신 업로드 날짜를 조회해 수집 시작일을 자동으로 계산합니다.
    MONGO_URI = None
    collection = None
    try:
        import local_secrets
        MONGO_URI = getattr(local_secrets, 'MONGO_URI', None)
        if MONGO_URI:
            print('Loaded MongoDB URI from local_secrets')
    except Exception:
        MONGO_URI = os.environ.get('MONGO_URI') or os.environ.get('MONGO_URI_SECRET')
        if MONGO_URI:
            print('Loaded MongoDB URI from environment')

    latest_uploaded_date = None
    client = None
    if MONGO_URI:
        client, collection = get_mongo_client_and_collection(MONGO_URI, 'bitcoindb', 'redditcompound')
        if collection is not None:
            try:
                latest_doc = collection.find_one(sort=[('date', -1)], projection={'date': 1})
                if latest_doc and 'date' in latest_doc and latest_doc['date'] is not None:
                    d = latest_doc['date']
                    if isinstance(d, str):
                        try:
                            latest_uploaded_date = datetime.fromisoformat(d).date()
                        except Exception:
                            latest_uploaded_date = None
                    elif hasattr(d, 'date'):
                        latest_uploaded_date = d.date()
                    else:
                        latest_uploaded_date = None
                    print(f'Latest uploaded date in MongoDB: {latest_uploaded_date}')
                else:
                    print('MongoDB에 업로드된 날짜 정보가 없습니다 (컬렉션 비어있음).')
            except Exception as e:
                print('MongoDB 조회 중 오류 발생:', e)
                collection = None

    # 수집할 날짜 범위 결정: 우선 MongoDB에 저장된 최신 업로드 날짜를 사용하고, 없으면 CONFIG의 값 사용
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    if latest_uploaded_date:
        start_date = latest_uploaded_date + timedelta(days=1)
    elif CONFIG.get('start_date'):
        start_date = datetime.strptime(CONFIG['start_date'], '%Y-%m-%d').date()
    else:
        print('❌ MongoDB에서 최신 업로드를 확인할 수 없고, CONFIG["start_date"]가 설정되어 있지 않습니다. 수동으로 시작일을 지정하세요.')
        return

    if start_date > yesterday:
        print(f'No new days to collect. Start date: {start_date}, yesterday: {yesterday}')
        return

    # update CONFIG dates
    CONFIG['start_date'] = start_date.strftime('%Y-%m-%d')
    CONFIG['end_date'] = yesterday.strftime('%Y-%m-%d')

    # Build absolute output directory dynamically from date range under repo folder
    folder_name = f"reddit_{CONFIG['start_date'].replace('-','')}_{CONFIG['end_date'].replace('-','')}"
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / 'bitcoin_reddit_data' / folder_name
    CONFIG['output_dir'] = str(out_dir)

    print(f"  수집 기간:    {CONFIG['start_date']} -> {CONFIG['end_date']}")
    print(f"  저장 경로:    {CONFIG['output_dir']}")

    collector = ArcticShiftCollector(CONFIG)
    total_posts    = 0
    total_comments = 0

    if CONFIG["collect"] in ("submissions", "both"):
        total_posts = collector.collect_submissions()

    if CONFIG["collect"] in ("comments", "both"):
        total_comments = collector.collect_comments()

    print(f"\n🎉 전체 수집 완료!")
    print(f"   게시물: {total_posts:,}개")
    print(f"   댓글:   {total_comments:,}개")

    print_stats(CONFIG["output_dir"], CONFIG["compress"])

    # ─────────────────────────────────────────────
    # 📊 JSONL 집계 + MongoDB 업서트
    # 아래 블록에서 .jsonl 파일을 직접 읽어 날짜별로 감성점수를 집계하고 업서트
    # ─────────────────────────────────────────────
    if CONFIG.get('create_merged_csv', False):
        try:
            out_dir = Path(CONFIG['output_dir'])
            ext = '.jsonl.gz' if CONFIG.get('compress', False) else '.jsonl'
            jsonl_files = list(out_dir.glob(f"*{ext}"))

            if not jsonl_files:
                print('  ℹ️ JSONL 파일이 없습니다 (CSV 저장 모드이거나 수집된 데이터가 없을 수 있습니다). CSV 기반 병합을 시도합니다.')
            else:
                sums = {}
                counts = {}
                for jf in jsonl_files:
                    opener = gzip.open if (jf.suffix == '.gz' or jf.name.endswith('.jsonl.gz')) else open
                    with opener(jf, 'rt', encoding='utf-8') as fh:
                        for line in fh:
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue

                            sent_raw = obj.get('_sent_compound') if obj is not None else None
                            try:
                                sent_val = float(sent_raw) if sent_raw is not None else None
                            except Exception:
                                sent_val = None
                            if sent_val is None or sent_val == 0.0:
                                continue

                            # 날짜 결정: 'date' 문자열 우선, 없으면 'created_utc' 타임스탬프 사용
                            date_val = None
                            if 'date' in obj and obj.get('date'):
                                try:
                                    dt = pd.to_datetime(obj.get('date'), errors='coerce')
                                    if pd.notnull(dt):
                                        date_val = dt.date().isoformat()
                                except Exception:
                                    date_val = str(obj.get('date'))
                            elif 'created_utc' in obj and obj.get('created_utc') is not None:
                                try:
                                    ts = int(obj.get('created_utc'))
                                    date_val = datetime.utcfromtimestamp(ts).date().isoformat()
                                except Exception:
                                    date_val = None
                            if not date_val:
                                continue

                            sums[date_val] = sums.get(date_val, 0.0) + float(sent_val)
                            counts[date_val] = counts.get(date_val, 0) + 1

                if not sums:
                    print('  ⚠️ JSONL 집계 결과가 없습니다 (모든 항목이 필터링되었거나 빈 파일).')
                elif collection is None:
                    print('  ⚠️ MongoDB 컬렉션이 없어 JSONL 집계 결과를 업로드하지 못했습니다. (MONGO_URI 확인 필요)')
                else:
                    upserted = 0
                    for date_str, total in sums.items():
                        cnt = counts.get(date_str, 0)
                        if cnt == 0:
                            continue
                        mean = total / cnt

                        try:
                            d = datetime.fromisoformat(date_str).date()
                        except Exception:
                            pd_dt = pd.to_datetime(date_str, errors='coerce')
                            if pd.isna(pd_dt):
                                continue
                            d = pd_dt.date()

                        # 기존 코드와 동일한 방식: midnight UTC datetime으로 매칭(기존 ISODate 문서 호환),
                        # 저장은 문자열 date로 유지
                        date_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                        date_str_norm = date_dt.date().isoformat()

                        # 소수점 4자리로 반올림하여 업서트
                        doc = {
                            'date': date_str_norm,
                            'compound': round(float(mean), 4),
                            'count': int(cnt),
                            'uploaded_at': datetime.utcnow()
                        }
                        filter_q = {'$or': [{'date': date_dt}, {'date': date_str_norm}]}
                        try:
                            res = collection.update_one(filter_q, {'$set': doc}, upsert=True)
                            upserted_id = str(res.upserted_id) if getattr(res, 'upserted_id', None) else None
                            # 로그에 compound(소수점4자리)와 count를 포함
                            try:
                                comp_str = f"{doc['compound']:.4f}"
                            except Exception:
                                comp_str = str(doc.get('compound'))
                            # 출력에서 matched/modified/upserted_id는 생략하고, 날짜/compound/count만 표시
                            print(f"Upserted(JSONL) date={date_str_norm}, compound={comp_str}, count={doc['count']}")
                            upserted += 1
                        except Exception as e:
                            print('몽고DB 업sert 실패(JSONL):', e)
                    print(f"  ✅ JSONL 집계 기반 MongoDB 업로드 완료(시도): {upserted}개 문서 (컬렉션: redditcompound)")
        except Exception as e:
            print('JSONL 집계/업로드 과정에서 오류 발생:', e)

    # CSV 기반 병합/변환 코드는 제거되었습니다. 이 스크립트의 기본 워크플로우는 JSONL(.jsonl/.jsonl.gz) 기반입니다.
    if CONFIG.get('create_merged_csv', False):
        print('  ℹ️ CSV 병합 관련 코드는 제거되었습니다. JSONL 집계만 수행됩니다.')

    # Close MongoDB client if opened
    try:
        if 'client' in locals() and client is not None:
            client.close()
            print('MongoDB client closed')
    except Exception:
        pass


if __name__ == "__main__":
    main()