from __future__ import annotations

import argparse
import ast
import csv
import gzip
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests


DEFAULT_REVIEWS_URL = "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
DEFAULT_RAW_PATH = Path("data/raw/steam_reviews.json.gz")
DEFAULT_OUTPUT_PATH = Path("data/processed/reviews_agg.csv")

GAME_ID_KEYS = ("game_id", "product_id", "app_id", "appid", "id")
TEXT_KEYS = ("text", "review", "review_text", "content")
RECOMMEND_KEYS = ("recommended", "recommend", "voted_up")
HOURS_KEYS = ("hours", "hours_played", "playtime_forever")

STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "dont",
    "down",
    "even",
    "for",
    "from",
    "game",
    "games",
    "get",
    "got",
    "had",
    "has",
    "have",
    "into",
    "its",
    "just",
    "like",
    "more",
    "most",
    "much",
    "not",
    "one",
    "only",
    "out",
    "play",
    "played",
    "playing",
    "really",
    "still",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "this",
    "time",
    "too",
    "very",
    "was",
    "well",
    "were",
    "when",
    "while",
    "will",
    "with",
    "would",
    "you",
    "your",
}

POSITIVE_TERMS = {
    "addictive",
    "amazing",
    "atmospheric",
    "awesome",
    "beautiful",
    "best",
    "charming",
    "clever",
    "cozy",
    "deep",
    "emotional",
    "engaging",
    "excellent",
    "fantastic",
    "fun",
    "funny",
    "good",
    "great",
    "immersive",
    "masterpiece",
    "memorable",
    "polished",
    "relaxing",
    "replayable",
    "rewarding",
    "satisfying",
    "solid",
    "wonderful",
}

NEGATIVE_TERMS = {
    "annoying",
    "bad",
    "boring",
    "broken",
    "buggy",
    "clunky",
    "crash",
    "crashes",
    "dull",
    "frustrating",
    "grind",
    "grindy",
    "mediocre",
    "poor",
    "repetitive",
    "tedious",
    "terrible",
    "unfair",
    "unfinished",
    "unplayable",
    "worst",
}

OPINION_TERMS = POSITIVE_TERMS | NEGATIVE_TERMS | {
    "challenging",
    "difficult",
    "easy",
    "fast",
    "hard",
    "heavy",
    "intense",
    "slow",
    "tactical",
}


@dataclass
class ReviewAggregate:
    review_count: int = 0
    recommended_count: int = 0
    not_recommended_count: int = 0
    positive_term_hits: int = 0
    negative_term_hits: int = 0
    total_hours: float = 0.0
    hours_count: int = 0
    keyword_counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Steam review opinions by game_id.")
    parser.add_argument("--url", default=DEFAULT_REVIEWS_URL)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional max reviews for smoke tests.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Prune counters after this many reviews.")
    parser.add_argument("--max-terms-per-game", type=int, default=300)
    parser.add_argument("--top-keywords", type=int, default=12)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Arquivo {destination} ja existe. Pulando download.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url}...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    print(f"Download salvo em {destination}.")


def parse_record(line: str) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(line)
    except (SyntaxError, ValueError):
        parsed = json.loads(line)
    return parsed if isinstance(parsed, dict) else {}


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", html.unescape(str(value or "")))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def tokenize(text: str) -> list[str]:
    normalized = normalize_ascii(text).lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9'-]{2,}", normalized)
    return [token.strip("-'") for token in tokens if token not in STOPWORDS and len(token.strip("-'")) >= 3]


def iter_ngrams(tokens: list[str], max_n: int = 3) -> Iterable[tuple[str, ...]]:
    for n in range(1, max_n + 1):
        if len(tokens) < n:
            continue
        for index in range(0, len(tokens) - n + 1):
            yield tuple(tokens[index : index + n])


def opinion_keyword_counts(text: str) -> Counter[str]:
    tokens = tokenize(text)
    counts: Counter[str] = Counter()
    for ngram in iter_ngrams(tokens):
        contains_opinion = any(token in OPINION_TERMS for token in ngram)
        if len(ngram) == 1 and not contains_opinion:
            continue

        phrase = " ".join(ngram)
        weight = 3 if contains_opinion else 1
        if len(ngram) >= 2:
            weight += 1
        counts[phrase] += weight
    return counts


def recommendation_value(record: dict[str, Any]) -> bool | None:
    value = first_present(record, RECOMMEND_KEYS)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "recommended", "recommend", "positive", "1"}:
            return True
        if normalized in {"false", "no", "not recommended", "negative", "0"}:
            return False
    return None


def hours_value(record: dict[str, Any]) -> float | None:
    value = first_present(record, HOURS_KEYS)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def update_aggregate(aggregate: ReviewAggregate, record: dict[str, Any], text: str) -> None:
    aggregate.review_count += 1

    recommendation = recommendation_value(record)
    if recommendation is True:
        aggregate.recommended_count += 1
    elif recommendation is False:
        aggregate.not_recommended_count += 1

    hours = hours_value(record)
    if hours is not None and hours >= 0:
        aggregate.total_hours += hours
        aggregate.hours_count += 1

    tokens = tokenize(text)
    aggregate.positive_term_hits += sum(token in POSITIVE_TERMS for token in tokens)
    aggregate.negative_term_hits += sum(token in NEGATIVE_TERMS for token in tokens)
    aggregate.keyword_counts.update(opinion_keyword_counts(text))


def prune_counter(counter: Counter[str], max_terms: int) -> None:
    if len(counter) <= max_terms:
        return
    most_common = counter.most_common(max_terms)
    counter.clear()
    counter.update(dict(most_common))


def prune_aggregates(aggregates: dict[str, ReviewAggregate], max_terms_per_game: int) -> None:
    for aggregate in aggregates.values():
        prune_counter(aggregate.keyword_counts, max_terms_per_game)


def sentiment_score(aggregate: ReviewAggregate) -> float:
    rec_total = aggregate.recommended_count + aggregate.not_recommended_count
    rec_score = aggregate.recommended_count / rec_total if rec_total else None

    lex_total = aggregate.positive_term_hits + aggregate.negative_term_hits
    if lex_total:
        lex_score = 0.5 + 0.5 * ((aggregate.positive_term_hits - aggregate.negative_term_hits) / lex_total)
    else:
        lex_score = None

    if rec_score is not None and lex_score is not None:
        return round((0.65 * rec_score) + (0.35 * lex_score), 4)
    if rec_score is not None:
        return round(rec_score, 4)
    if lex_score is not None:
        return round(lex_score, 4)
    return 0.5


def process_reviews(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    chunk_size: int = 50000,
    max_terms_per_game: int = 300,
    top_keywords: int = 12,
) -> int:
    aggregates: dict[str, ReviewAggregate] = defaultdict(ReviewAggregate)
    processed = 0
    skipped = 0

    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as file:
        for line in file:
            if limit is not None and processed >= limit:
                break
            try:
                record = parse_record(line)
                game_id = first_present(record, GAME_ID_KEYS)
                text = first_present(record, TEXT_KEYS)
                if not game_id or not text:
                    skipped += 1
                    continue

                update_aggregate(aggregates[str(game_id)], record, str(text))
                processed += 1

                if processed % chunk_size == 0:
                    prune_aggregates(aggregates, max_terms_per_game)
                    print(f"Processadas {processed} reviews; jogos agregados: {len(aggregates)}")
            except Exception:
                skipped += 1

    prune_aggregates(aggregates, max_terms_per_game)
    write_aggregates(aggregates, output_path, top_keywords=top_keywords)
    print(f"Reviews processadas: {processed}; ignoradas: {skipped}; jogos: {len(aggregates)}")
    return len(aggregates)


def write_aggregates(
    aggregates: dict[str, ReviewAggregate],
    output_path: Path,
    top_keywords: int = 12,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "game_id",
        "review_count",
        "recommended_count",
        "not_recommended_count",
        "recommendation_ratio",
        "avg_hours",
        "positive_term_hits",
        "negative_term_hits",
        "sentiment_score",
        "review_keywords",
    ]

    rows = sorted(aggregates.items(), key=lambda item: item[1].review_count, reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for game_id, aggregate in rows:
            rec_total = aggregate.recommended_count + aggregate.not_recommended_count
            recommendation_ratio = aggregate.recommended_count / rec_total if rec_total else ""
            avg_hours = aggregate.total_hours / aggregate.hours_count if aggregate.hours_count else ""
            keywords = ", ".join(term for term, _ in aggregate.keyword_counts.most_common(top_keywords))
            writer.writerow(
                {
                    "game_id": game_id,
                    "review_count": aggregate.review_count,
                    "recommended_count": aggregate.recommended_count,
                    "not_recommended_count": aggregate.not_recommended_count,
                    "recommendation_ratio": round(recommendation_ratio, 4) if recommendation_ratio != "" else "",
                    "avg_hours": round(avg_hours, 2) if avg_hours != "" else "",
                    "positive_term_hits": aggregate.positive_term_hits,
                    "negative_term_hits": aggregate.negative_term_hits,
                    "sentiment_score": sentiment_score(aggregate),
                    "review_keywords": keywords,
                }
            )


def main() -> None:
    args = parse_args()
    download_file(args.url, args.raw_path, force=args.force_download)
    process_reviews(
        input_path=args.raw_path,
        output_path=args.output_path,
        limit=args.limit,
        chunk_size=args.chunk_size,
        max_terms_per_game=args.max_terms_per_game,
        top_keywords=args.top_keywords,
    )


if __name__ == "__main__":
    main()
