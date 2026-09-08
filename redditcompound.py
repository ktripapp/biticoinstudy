"""
Reddit Bitcoin 데이터 수집기
Arctic Shift / Pushshift API 방식으로 r/Bitcoin의 수년치 게시물과 댓글을 수집합니다.

지원 엔드포인트:
  - Arctic Shift API (https://arctic-shift.photon-reddit.com)  ← 현재 가장 안정적
  - Pushshift API (https://api.pushshift.io) ← 현재 접근 제한 있음

사용법:
  pip install requests tqdm
  python reddit_bitcoin_collector.py
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

    # 한 번 요청당 가져올 항목 수 (Arctic Shift 최대 100)
    "batch_size": 100,

    # 요청 간격 (초) — rate limit 방지
    "request_delay": 1.0,

    # 재시도 횟수
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
    # If configured to not save JSONL, write CSV directly and return CSV path
    if not CONFIG.get('save_jsonl', False):
        csv_path = filepath.with_suffix('.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Only keep `date` and `_sent_compound` columns.
        fieldnames = ['date', '_sent_compound']
        # Always append to existing CSV during a run so each batch adds rows.
        # If the file doesn't exist, create and write header.
        mode = 'a' if csv_path.exists() else 'w'
        write_header = (mode == 'w')
        with open(csv_path, mode, newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in records:
                # skip records whose sentiment compound is explicitly 0.0
                sent_raw = r.get('_sent_compound') if r is not None else None
                try:
                    sent_val = float(sent_raw) if sent_raw is not None else None
                except Exception:
                    sent_val = None
                if sent_val is not None and sent_val == 0.0:
                    continue

                # determine date: prefer 'date' string, else convert 'created_utc' timestamp
                date_val = None
                if 'date' in r and r.get('date'):
                    try:
                        # try to normalize to YYYY-MM-DD
                        dt = pd.to_datetime(r.get('date'), errors='coerce')
                        if pd.notnull(dt):
                            date_val = dt.date().isoformat()
                    except Exception:
                        date_val = str(r.get('date'))
                elif 'created_utc' in r and r.get('created_utc') is not None:
                    try:
                        ts = int(r.get('created_utc'))
                        date_val = datetime.utcfromtimestamp(ts).date().isoformat()
                    except Exception:
                        date_val = None
                # sentiment value
                sent = sent_val if sent_val is not None else (r.get('_sent_compound') if r.get('_sent_compound') is not None else None)
                writer.writerow({'date': date_val, '_sent_compound': sent})
        return csv_path

    # fallback: save JSONL as before
    if compress:
        filepath = filepath.with_suffix('.jsonl.gz')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(filepath, 'at', encoding='utf-8') as f:
            for r in records:
                # skip records with explicit 0.0 sentiment
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
                # skip records with explicit 0.0 sentiment
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
    """Try to connect to MongoDB, run a ping test, and return (client, collection) or (None, None).

    Prints detailed error on failure to help CI/Actions debugging.
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


def jsonl_to_csv(jsonl_path: Path, csv_path: Path, compress: bool = True, append: bool = False):
    """JSONL(.gz) 파일을 CSV로 변환합니다. 모든 최상위 키의 union을 헤더로 사용합니다.
    주의: 큰 파일은 메모리/IO 비용이 큽니다."""
    opener = gzip.open if compress else open

    # 1) 키 수집
    keys = set()
    with opener(jsonl_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            keys.update(obj.keys())

    keys = sorted(keys)

    # 2) 실제 쓰기
    # If append==True and csv exists, try to append while keeping header compatibility.
    if append and csv_path.exists():
        # read existing header
        with open(csv_path, "r", encoding="utf-8", newline='') as exf:
            reader = csv.reader(exf)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []

        existing_keys = existing_header
        new_keys = keys
        union_keys = sorted(list(dict.fromkeys(existing_keys + new_keys)))

        # if headers match exactly, simply append rows
        if existing_keys == union_keys:
            with opener(jsonl_path, "rt", encoding="utf-8") as fh, open(csv_path, "a", newline='', encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=union_keys)
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    row = {}
                    for k in union_keys:
                        v = obj.get(k, "")
                        if isinstance(v, (dict, list)):
                            row[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            row[k] = v
                    writer.writerow(row)
            return

        # Headers differ: rewrite full CSV with union header to keep consistency
        import shutil
        tmp_path = csv_path.with_suffix('.tmp.csv')
        with open(tmp_path, "w", newline='', encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=union_keys)
            writer.writeheader()
            # copy existing rows
            with open(csv_path, "r", encoding="utf-8", newline='') as exf:
                reader = csv.DictReader(exf)
                for r in reader:
                    row = {k: r.get(k, "") for k in union_keys}
                    writer.writerow(row)
            # append new rows
            with opener(jsonl_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    row = {}
                    for k in union_keys:
                        v = obj.get(k, "")
                        if isinstance(v, (dict, list)):
                            row[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            row[k] = v
                    writer.writerow(row)
        # replace original
        shutil.move(str(tmp_path), str(csv_path))
        return

    # default: write new CSV (overwrite)
    with opener(jsonl_path, "rt", encoding="utf-8") as fh, open(csv_path, "w", newline='', encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # ensure only top-level simple values; convert lists/dicts to JSON string
            row = {}
            for k in keys:
                v = obj.get(k, "")
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = v
            writer.writerow(row)


def convert_all_jsonl_to_csv(output_dir: Path, compress: bool = True, append: bool = False) -> list:
    """output_dir 내 JSONL(.jsonl 또는 .jsonl.gz) 파일을 찾아 같은 이름으로 .csv를 생성합니다.
    반환값: 생성된 CSV 파일 경로 리스트
    """
    out_dir = Path(output_dir)
    created = []
    ext = ".jsonl.gz" if compress else ".jsonl"
    for f in out_dir.glob(f"*{ext}"):
        csv_name = f.with_suffix("")  # remove .gz
        csv_name = csv_name.with_suffix(".csv")
        try:
            jsonl_to_csv(f, csv_name, compress=compress, append=append)
            created.append(csv_name)
        except Exception as e:
            print(f"CSV 변환 실패: {f.name} -> {e}")
    return created


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
    Pushshift API 백업 수집기.
    ※ 2023년 이후 공개 Pushshift API는 접근이 제한됩니다.
       Reddit moderator 계정이 있으면 academic access 신청 가능.
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

    # Determine date range to collect: prefer MongoDB-derived date; fallback to CONFIG if provided
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
    # save_jsonl=True 설정 시에는 CSV가 아닌 .jsonl(.gz) 파일만 생성되므로,
    # 아래 블록에서 .jsonl 파일을 직접 읽어 날짜별로 감성점수를 집계하고 업서트합니다.
    # (CSV 기반 병합 블록은 save_jsonl=False일 때를 위한 보조 경로로 이어서 유지됩니다.)
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

                        doc = {
                            'date': date_str_norm,
                            'compound': round(float(mean), 10),
                            'count': int(cnt),
                            'uploaded_at': datetime.utcnow()
                        }
                        filter_q = {'$or': [{'date': date_dt}, {'date': date_str_norm}]}
                        try:
                            res = collection.update_one(filter_q, {'$set': doc}, upsert=True)
                            upserted_id = str(res.upserted_id) if getattr(res, 'upserted_id', None) else None
                            print(f"Upserted(JSONL) date={date_str_norm} -> matched={res.matched_count}, modified={res.modified_count}, upserted_id={upserted_id}")
                            upserted += 1
                        except Exception as e:
                            print('몽고DB 업sert 실패(JSONL):', e)
                    print(f"  ✅ JSONL 집계 기반 MongoDB 업로드 완료(시도): {upserted}개 문서 (컬렉션: redditcompound)")
        except Exception as e:
            print('JSONL 집계/업로드 과정에서 오류 발생:', e)

    # 병합 CSV 생성은 설정에 따라 수행합니다. 기본값(False)일 경우 생략합니다.
    # (save_jsonl=False로 CSV가 실제 생성된 경우를 위한 보조 경로입니다.)
    if CONFIG.get('create_merged_csv', False):
        try:
            out_dir = Path(CONFIG['output_dir'])
            csv_files = list(out_dir.glob('comments_*.csv')) + list(out_dir.glob('submissions_*.csv'))
            parts = []
            for f in csv_files:
                try:
                    df = pd.read_csv(f, dtype={'date': str})
                except Exception as e:
                    print(f'CSV 읽기 실패, 건너뜀: {f.name} -> {e}')
                    continue
                if '_sent_compound' in df.columns:
                    sent_col = '_sent_compound'
                elif 'compound' in df.columns:
                    sent_col = 'compound'
                else:
                    print(f"컬럼 '_sent_compound'/'compound' 없음, 건너뜀: {f.name}")
                    continue

                if 'count' in df.columns:
                    count_col = 'count'
                else:
                    count_col = None

                keep_cols = ['date', sent_col]
                if count_col:
                    keep_cols.append(count_col)

                df = df.loc[:, keep_cols]
                df = df.dropna(subset=['date', sent_col])
                df[sent_col] = pd.to_numeric(df[sent_col], errors='coerce')
                df = df[df[sent_col] != 0.0]

                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date.astype(str)
                df = df.dropna(subset=['date'])

                if count_col:
                    df['count'] = pd.to_numeric(df[count_col], errors='coerce').fillna(1).astype(int)
                else:
                    df['count'] = 1

                df = df.rename(columns={sent_col: '_sent_compound'})
                parts.append(df.loc[:, ['date', '_sent_compound', 'count']])

            if parts:
                merged = pd.concat(parts, ignore_index=True)
                agg_sent = merged.groupby('date', as_index=False)['_sent_compound'].mean()
                agg_count = merged.groupby('date', as_index=False)['count'].sum()
                agg = pd.merge(agg_sent, agg_count, on='date')
                agg = agg.rename(columns={'_sent_compound': 'compound', 'count': 'count'})
                agg['compound'] = agg['compound'].round(10)

                # 집계 결과를 MongoDB에 업서트하거나, MongoDB가 없으면 로컬 CSV로 저장합니다.
                if collection is not None:
                    print(f"Aggregated rows: {len(agg)}, columns: {list(agg.columns)}")
                    print("Sample aggregated rows:\n", agg.head().to_string(index=False))
                    upserted = 0
                    for _, row in agg.iterrows():
                        try:
                            date_str = row['date']
                            try:
                                date_obj = datetime.fromisoformat(date_str).date()
                            except Exception:
                                date_obj = pd.to_datetime(date_str, errors='coerce').date()

                            # normalize date as midnight UTC datetime for matching, but store date as string
                            date_dt = datetime(date_obj.year, date_obj.month, date_obj.day, tzinfo=timezone.utc)
                            date_str = date_dt.date().isoformat()
                            # Build document using string date to keep existing string representation
                            doc = {
                                'date': date_str,
                                'compound': float(row['compound']),
                                'count': int(row['count']),
                                'uploaded_at': datetime.utcnow()
                            }
                            # Match either existing ISODate(datetime) or string date values, then write string
                            filter_q = {'$or': [{'date': date_dt}, {'date': date_str}]}
                            res = collection.update_one(filter_q, {'$set': doc}, upsert=True)
                            try:
                                matched = int(res.matched_count)
                                modified = int(res.modified_count)
                                upserted_id = str(res.upserted_id) if getattr(res, 'upserted_id', None) else None
                            except Exception:
                                matched = getattr(res, 'matched_count', None)
                                modified = getattr(res, 'modified_count', None)
                                upserted_id = getattr(res, 'upserted_id', None)
                            print(f"Upserted date={date_str} -> matched={matched}, modified={modified}, upserted_id={upserted_id}")
                            upserted += 1
                        except Exception as e:
                            print('몽고DB 업sert 실패:', e)
                    print(f"  ✅ MongoDB에 업로드 완료(시도): {upserted}개 문서 (컬렉션: redditcompound)")
                else:
                    try:
                        out_path = out_dir / 'reddit_compound.csv'
                        agg.to_csv(out_path, index=False)
                        print(f"  ✅ 병합 CSV 생성 완료: {out_path} ({len(agg)}개 행)")
                    except Exception as e:
                        print('병합 결과 CSV 저장 실패:', e)
            else:
                print('  ⚠️ 병합할 파일 기반 데이터가 없습니다.')
        except Exception as e:
            print('병합(파일 기반) 과정에서 오류 발생:', e)
    else:
        # 사용자가 병합을 원치 않으므로 아무 파일도 생성하지 않습니다.
        print('  ℹ️ 설정에 따라 병합 CSV 생성을 건너뜁니다.')

    # Close MongoDB client if opened
    try:
        if 'client' in locals() and client is not None:
            client.close()
            print('MongoDB client closed')
    except Exception:
        pass


if __name__ == "__main__":
    main()