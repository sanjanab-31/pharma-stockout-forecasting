"""ingest.py - asynchronous API ingestion layer (report Sections 3.2.2 and 5.2.1).

Each data source is polled concurrently, records are validated against a simple schema,
duplicate pushes are discarded with a hash key, and valid records are published to a topic.
For a self-contained demo the five supply-chain APIs are served by `MockSupplyChainAPI`
from the generated dataset; publishing goes to in-memory topics, or to Apache Kafka when
the KAFKA_BOOTSTRAP environment variable is set and kafka-python is installed.

    python ingest.py --day 400
"""
import argparse, asyncio, hashlib, json, os, random
from collections import defaultdict
import pandas as pd
from config import DATA

SCHEMAS = {  # topic -> required fields
    "inventory": ["facility_id", "drug_id", "day", "on_hand"],
    "dispensing": ["facility_id", "drug_id", "day", "demand", "dispensed"],
    "orders": ["facility_id", "drug_id", "day", "on_order", "order_qty"],
    "shipments": ["facility_id", "drug_id", "day", "eta_days", "intransit_14d"],
    "shortages": ["drug_id", "day", "upstream_shortage"],
}


class MockSupplyChainAPI:
    """Serves one day of records per source, with network latency and occasional duplicate pushes."""
    def __init__(self, panel: pd.DataFrame):
        self.panel = panel

    async def get(self, topic: str, day: int):
        await asyncio.sleep(random.uniform(0.01, 0.05))                     # simulated network latency
        rows = self.panel[self.panel.day == day]
        if topic == "shortages":
            rows = rows.drop_duplicates("drug_id")
        recs = rows[SCHEMAS[topic]].to_dict("records")
        return recs + random.sample(recs, k=min(5, len(recs)))             # a few duplicate pushes


class Publisher:
    def __init__(self):
        self.topics = defaultdict(list); self.producer = None
        if os.environ.get("KAFKA_BOOTSTRAP"):
            try:
                from kafka import KafkaProducer
                self.producer = KafkaProducer(bootstrap_servers=os.environ["KAFKA_BOOTSTRAP"],
                                              value_serializer=lambda v: json.dumps(v, default=str).encode())
            except Exception as e:
                print(f"[ingest] Kafka unavailable ({e}); using in-memory topics")

    def send(self, topic, rec):
        if self.producer: self.producer.send(topic, rec)
        else: self.topics[topic].append(rec)


def valid(rec, topic):
    return all(k in rec and rec[k] is not None and rec[k] == rec[k] for k in SCHEMAS[topic])   # rec[k]==rec[k] rejects NaN


async def poll_cycle(api, pub, day, seen):
    stats = {}
    async def fetch(topic):
        recs = await api.get(topic, day); ok = bad = dup = 0
        for rec in recs:
            if not valid(rec, topic): bad += 1; continue                 # -> dead-letter queue in production
            key = hashlib.sha1(json.dumps([topic, rec], sort_keys=True, default=str).encode()).hexdigest()
            if key in seen: dup += 1; continue
            seen.add(key); pub.send(topic, rec); ok += 1
        stats[topic] = (ok, bad, dup)
    await asyncio.gather(*(fetch(t) for t in SCHEMAS))                   # all sources polled concurrently
    return stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--day", type=int, default=400); a = ap.parse_args()
    random.seed(0)
    api = MockSupplyChainAPI(pd.read_pickle(DATA / "raw_panel.pkl")); pub = Publisher()
    stats = asyncio.run(poll_cycle(api, pub, a.day, set()))
    print(f"[ingest] polling cycle for day {a.day}:")
    for t, (ok, bad, dup) in stats.items(): print(f"   {t:<11} published {ok:5d} | invalid {bad:3d} | duplicates dropped {dup}")


if __name__ == "__main__":
    main()
