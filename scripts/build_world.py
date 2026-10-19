"""도메인 팩으로 월드 만들기.

  .venv/bin/python -m scripts.build_world domains/ev_charging.yaml
  .venv/bin/python -m scripts.build_world domains/drafts/insulin_pump.yaml --axes dm_status,urgency
"""
import argparse
import time

from world import config
from world.domain import build_world, load_pack
from world.segment import segment
from world.store import connect, row_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--axes", help="세그먼트 표에 쓸 축, 쉼표 구분 (기본: 팩의 axes_default)")
    ap.add_argument("--db", default=config.DB_PATH)
    a = ap.parse_args()
    t = time.time()
    pack = load_pack(a.pack)
    con = connect(a.db)
    print(f"pack={pack['name']}  L0 rows={con.execute('SELECT count(*) FROM l0').fetchone()[0]:,}  row_weight={row_weight(con):.1f}")
    w, results = build_world(con, pack)
    for r in results:
        line = f"  {r['method']:12s} {r['attr']:16s} {r['status']}"
        for c in r["checks"]:
            line += f"  [{c['check']} {'ok' if c['passed'] else 'FAIL'} {c['detail']}]"
        print(line)
    axes = a.axes.split(",") if a.axes else pack.get("axes_default", [])
    if axes:
        df = segment(con, w, axes, filter=pack.get("market_scope"))
        print(df.to_string(index=False))
    print(f"world_id={w}  elapsed={time.time()-t:.1f}s  db={a.db}")


if __name__ == "__main__":
    main()
