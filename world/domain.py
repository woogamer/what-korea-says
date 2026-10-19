"""도메인 팩: 업종별 정의(L1 열·믿음·검사·기본 축)를 YAML 한 파일로.

코어(store/ops/reconcile/segment)는 업종을 모른다. 팩이 "이 시장에 어떤 열을 어떤 통계로 붙일지"를 적고,
build_world 가 그 순서대로 연산을 실행한다. 새 업종 = 새 YAML.
"""
from pathlib import Path

import yaml

from . import ops
from .store import add_belief, create_world


def load_pack(path) -> dict:
    pack = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k in ("name", "attributes"):
        if k not in pack:
            raise ValueError(f"pack needs '{k}'")
    return pack


def build_world(con, pack: dict, world_name: str | None = None) -> tuple[str, list[dict]]:
    """팩 → 월드. 믿음을 먼저 등록하고 attributes 순서대로 연산. (world_id, 연산 결과 목록)."""
    w = create_world(con, world_name or f"{pack['name']} v1")
    belief_ids = {}
    for b in pack.get("beliefs", []):
        belief_ids[b["id"]] = add_belief(
            con, w, b["proposition"], b["grade"], b.get("source_ref", ""),
            variables=b.get("variables"), n=b.get("n"), sample_condition=b.get("sample_condition"))
    results = []
    for a in pack["attributes"]:
        a = dict(a)
        method = a.pop("method")
        a["attr"] = a.pop("name")
        if "belief" in a:
            a["source_beliefs"] = [belief_ids[a.pop("belief")]]
        oid, r = ops.run(con, w, method, a, actor=f"pack:{pack['name']}")
        status = con.execute("SELECT status FROM op_log WHERE op_id=?", [oid]).fetchone()[0]
        results.append({"attr": a["attr"], "method": method, "status": status, "checks": r.get("checks", [])})
    return w, results
