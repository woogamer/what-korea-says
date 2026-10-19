"""월드를 바꾸는 연산. 모두 op_log에 기록되고 propose → approve → apply 를 거친다.

apply 단계에서 연산이 만든 열/가중치는 비활성으로 생성되고, params["checks"]의
정합성 검사를 통과해야 활성화된다(실패 = rejected, 월드는 그대로).
"""
import json
from datetime import datetime

import pandas as pd

from . import reconcile
from .grades import validate
from .store import (active_attrs, build_view, create_world, get_world, new_id,
                    view_name, L0_COLUMNS)

OPS = {}


def op(name):
    def deco(fn):
        OPS[name] = fn
        return fn
    return deco


# ---------- 생명주기 ----------

def propose(con, world_id, op_type, params, actor="analyst") -> str:
    if op_type not in OPS:
        raise ValueError(f"unknown op {op_type}")
    seq = con.execute("SELECT COALESCE(max(seq),0)+1 FROM op_log WHERE world_id=?",
                      [world_id]).fetchone()[0]
    oid = new_id("op")
    con.execute("INSERT INTO op_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                [oid, world_id, seq, op_type, json.dumps(params, ensure_ascii=False),
                 "proposed", actor, None, None, datetime.now()])
    return oid


def approve(con, op_id):
    con.execute("UPDATE op_log SET status='approved' WHERE op_id=? AND status='proposed'", [op_id])


def apply(con, op_id) -> dict:
    row = con.execute("SELECT world_id, op_type, params, status FROM op_log WHERE op_id=?",
                      [op_id]).fetchone()
    world_id, op_type, params, status = row[0], row[1], json.loads(row[2]), row[3]
    if status != "approved":
        raise PermissionError(f"op {op_id} is {status}, needs approval")
    result = OPS[op_type](con, world_id, params, op_id)
    # 검사는 새 열/가중치가 활성화된 상태에서 돌려야 하므로 임시 활성화 후 실패 시 되돌린다
    _activate(con, world_id, result, op_id)
    checks = [reconcile.run_check(con, world_id, c) for c in params.get("checks", [])]
    passed = all(c["passed"] for c in checks)
    result["checks"] = checks
    if not passed:
        _deactivate(con, world_id, result)
    con.execute("UPDATE op_log SET status=?, result=? WHERE op_id=?",
                ["committed" if passed else "rejected", json.dumps(result, ensure_ascii=False), op_id])
    build_view(con, world_id)
    return result


def run(con, world_id, op_type, params, actor="analyst") -> tuple[str, dict]:
    """분석자가 직접 실행: propose + approve + apply."""
    oid = propose(con, world_id, op_type, params, actor)
    approve(con, oid)
    return oid, apply(con, oid)


def _activate(con, world_id, result, op_id):
    for attr_id in result.get("attr_ids", []):
        name = con.execute("SELECT name FROM attr_def WHERE world_id=? AND attr_id=?",
                           [world_id, attr_id]).fetchone()[0]
        con.execute("UPDATE attr_def SET active=false WHERE world_id=? AND name=?", [world_id, name])
        con.execute("UPDATE attr_def SET active=true WHERE world_id=? AND attr_id=?", [world_id, attr_id])
    if result.get("weight_ref"):
        result["prev_weight_ref"] = get_world(con, world_id)["weight_ref"]
        con.execute("UPDATE world SET weight_ref=? WHERE world_id=?", [result["weight_ref"], world_id])
    con.execute("UPDATE world SET head_op=? WHERE world_id=?", [op_id, world_id])
    build_view(con, world_id)


def _deactivate(con, world_id, result):
    for attr_id in result.get("attr_ids", []):
        prev = result.get("prev_attr_ids", {}).get(attr_id)
        con.execute("UPDATE attr_def SET active=false WHERE world_id=? AND attr_id=?", [world_id, attr_id])
        if prev:
            con.execute("UPDATE attr_def SET active=true WHERE world_id=? AND attr_id=?", [world_id, prev])
    if result.get("weight_ref"):
        con.execute("UPDATE world SET weight_ref=? WHERE world_id=?",
                    [result.get("prev_weight_ref"), world_id])
    con.execute("UPDATE world SET head_op=(SELECT op_id FROM op_log WHERE world_id=? "
                "AND status='committed' ORDER BY seq DESC LIMIT 1) WHERE world_id=?",
                [world_id, world_id])


# ---------- 내부 ----------

def _new_attr(con, world_id, name, dtype, domain, grade, method, source_beliefs, op_id):
    validate(grade)
    if name in L0_COLUMNS:
        raise ValueError(f"{name} is an L0 column")
    prev = con.execute("SELECT attr_id, COALESCE(max(version),0) FROM attr_def WHERE world_id=? "
                       "AND name=? AND active GROUP BY attr_id", [world_id, name]).fetchone()
    ver = con.execute("SELECT COALESCE(max(version),0)+1 FROM attr_def WHERE world_id=? AND name=?",
                      [world_id, name]).fetchone()[0]
    attr_id = f"{name}@v{ver}_{world_id}"
    con.execute("INSERT INTO attr_def VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [world_id, attr_id, name, ver, dtype, json.dumps(domain or [], ensure_ascii=False),
                 grade, method, json.dumps(source_beliefs or []), op_id, False])
    return attr_id, (prev[0] if prev else None)


def _result(attr_id, prev, **extra):
    r = {"attr_ids": [attr_id], "prev_attr_ids": {attr_id: prev} if prev else {}}
    r.update(extra)
    return r


# ---------- 연산 ----------

@op("extend")
def extend(con, world_id, p, op_id):
    """열 추가. 값은 전부 [미]."""
    attr_id, prev = _new_attr(con, world_id, p["attr"], p.get("dtype", "categorical"),
                              p.get("domain"), "[미]", "extend", [], op_id)
    return _result(attr_id, prev)


@op("impute_stat")
def impute_stat(con, world_id, p, op_id):
    """조건부 확률표로 개인 단위 전개. 재현 가능(seed). 표에 없는 조합은 NULL([미]).

    p = {attr, by:[col...], table:[{by:{col:val}, p:{value:prob}}], grade, source_beliefs, seed}
    """
    by, seed = p["by"], int(p.get("seed", 0))
    rows = []
    for cell in p["table"]:
        total = sum(cell["p"].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities sum to {total} in {cell['by']}")
        lo = 0.0
        for val, pr in cell["p"].items():
            rows.append({**{f"by_{c}": cell["by"][c] for c in by}, "value": val, "lo": lo, "hi": lo + pr, "p": pr})
            lo += pr
    tbl = pd.DataFrame(rows)
    con.register("_impute_tbl", tbl)
    attr_id, prev = _new_attr(con, world_id, p["attr"], "categorical",
                              sorted({r["value"] for r in rows}), p.get("grade", "[도]"),
                              "stat_conditional", p.get("source_beliefs"), op_id)
    v = view_name(world_id)
    build_view(con, world_id)
    join = " AND ".join(f"v.{c} = t.by_{c}" for c in by) or "TRUE"   # by=[] 이면 주변분포를 전원에게(축 차이 없음)
    u = f"((hash(v.uuid || '{seed}:{p['attr']}') % 1000003) / 1000003.0)"  # 열마다 다른 난수열
    con.execute(f"""
        INSERT INTO attr_val
        SELECT '{attr_id}', v.uuid, t.value, NULL, t.p, '{p.get("grade", "[도]")}'
        FROM {v} v JOIN _impute_tbl t ON {join}
        WHERE {u} >= t.lo AND {u} < t.hi""")
    con.unregister("_impute_tbl")
    n = con.execute("SELECT count(*) FROM attr_val WHERE attr_id=?", [attr_id]).fetchone()[0]
    return _result(attr_id, prev, n_filled=n)


@op("impute_rule")
def impute_rule(con, world_id, p, op_id):
    """SQL 식으로 열 계산. p = {attr, expr, grade, dtype, source_beliefs}"""
    dtype = p.get("dtype", "categorical")
    grade = p.get("grade", "[도]")
    attr_id, prev = _new_attr(con, world_id, p["attr"], dtype, p.get("domain"), grade,
                              "rule_sql", p.get("source_beliefs"), op_id)
    v = view_name(world_id)
    build_view(con, world_id)
    val = f"NULL, CAST(({p['expr']}) AS DOUBLE)" if dtype == "numeric" else f"CAST(({p['expr']}) AS VARCHAR), NULL"
    con.execute(f"INSERT INTO attr_val SELECT '{attr_id}', uuid, {val}, 1.0, '{grade}' "
                f"FROM {v} WHERE ({p['expr']}) IS NOT NULL")
    n = con.execute("SELECT count(*) FROM attr_val WHERE attr_id=?", [attr_id]).fetchone()[0]
    return _result(attr_id, prev, n_filled=n)


@op("perturb")
def perturb(con, world_id, p, op_id):
    """시나리오용 외생 변화 = impute_rule, 기본 등급 [가]. fork한 월드에서 쓴다."""
    return impute_rule(con, world_id, {**p, "grade": p.get("grade", "[가]")}, op_id)


@op("reweight")
def reweight(con, world_id, p, op_id):
    """레이킹(iterative proportional fitting)으로 filter 안 행의 가중치를 목표 주변분포에 맞춤.

    p = {targets:{col:{val:share}}, filter, iters, grade}. 목표에 없는 값은 가중치 0.
    """
    v = view_name(world_id)
    build_view(con, world_id)
    cols = list(p["targets"])
    where = p.get("filter") or "TRUE"
    df = con.execute(f"SELECT uuid, w, {', '.join(cols)} FROM {v} WHERE {where}").df()
    total = df["w"].sum()
    for _ in range(int(p.get("iters", 20))):
        for c in cols:
            tgt = p["targets"][c]
            cur = df.groupby(c)["w"].sum()
            factor = {val: (share * total / cur[val] if val in cur and cur[val] > 0 else 0.0)
                      for val, share in tgt.items()}
            df["w"] = df[c].map(factor).fillna(0.0) * df["w"]
    ver = con.execute("SELECT COALESCE(max(version),0)+1 FROM weight WHERE world_id=?",
                      [world_id]).fetchone()[0]
    con.register("_new_w", df[["uuid", "w"]])
    con.execute(f"INSERT INTO weight SELECT '{world_id}', {ver}, uuid, w FROM {v} WHERE NOT ({where})")
    con.execute(f"INSERT INTO weight SELECT '{world_id}', {ver}, uuid, w FROM _new_w")
    con.unregister("_new_w")
    achieved = {c: df.groupby(c)["w"].sum().div(df["w"].sum()).round(4).to_dict() for c in cols}
    return {"weight_ref": f"{world_id}:{ver}", "n_reweighted": len(df), "achieved": achieved}


@op("anchor")
def anchor(con, world_id, p, op_id):
    """실측 개인을 가장 가까운 L0 행에 정박하고 observed 값을 L2 열로 기록.

    p = {records:[{source_ref, source_type, fields:{col:val}, observed:{col:val}}],
         grade:'[사-대면]', seed, holdout:[col...]}
    holdout 열은 채점 전용(보정 카드): 월드에는 기록되지만 응답자 렌더링에서 제외된다(method='anchor_holdout').
    fields 중 age는 ±2세 허용, 나머지는 정확 일치. 일치 필드 수가 최대인 행 중 seed로 하나.
    observed 열은 새 버전을 만들어(기존 값 복사) 정박된 행만 덮어쓴다.
    """
    grade, seed = validate(p.get("grade", "[사]")), int(p.get("seed", 0))
    holdout = set(p.get("holdout", []))
    v = view_name(world_id)
    build_view(con, world_id)
    taken = {r[0] for r in con.execute("SELECT uuid FROM anchor WHERE world_id=?", [world_id]).fetchall()}
    anchors, observed_cols = [], {}
    for rec in p["records"]:
        terms, params = [], []
        for col, val in rec["fields"].items():
            if col == "age":
                terms.append("CASE WHEN abs(age - ?) <= 2 THEN 1 ELSE 0 END"); params.append(val)
            else:
                terms.append(f"CASE WHEN {col} = ? THEN 1 ELSE 0 END"); params.append(val)
        score = " + ".join(terms) if terms else "0"
        excl = f"AND uuid NOT IN ({','.join(repr(u) for u in taken)})" if taken else ""
        row = con.execute(
            f"SELECT uuid, ({score}) AS s FROM {v} WHERE TRUE {excl} "
            f"ORDER BY s DESC, hash(uuid || '{seed}') LIMIT 1", params).fetchone()
        uid, s = row
        taken.add(uid)
        aid = new_id("a")
        con.execute("INSERT INTO anchor VALUES (?,?,?,?,?,?,?,?,?)",
                    [world_id, aid, rec.get("source_type", "interview"), rec["source_ref"], uid,
                     s / max(len(terms), 1), json.dumps(list(rec["fields"]), ensure_ascii=False),
                     json.dumps(rec.get("observed", {}), ensure_ascii=False), op_id])
        anchors.append({"anchor_id": aid, "uuid": uid, "match_score": s / max(len(terms), 1)})
        for col, val in rec.get("observed", {}).items():
            observed_cols.setdefault(col, []).append((uid, val))
    attr_ids, prev_ids = [], {}
    for col, pairs in observed_cols.items():
        existing = next((a for a in active_attrs(con, world_id) if a["name"] == col), None)
        dtype = existing["dtype"] if existing else "categorical"
        attr_id, prev = _new_attr(con, world_id, col, dtype, existing and json.loads(existing["domain"]),
                                  existing["grade"] if existing else "[미]", "anchor_holdout" if col in holdout else "anchor",
                                  p.get("source_beliefs"), op_id)
        if prev:
            con.execute("INSERT INTO attr_val SELECT ?, uuid, v_str, v_num, confidence, grade "
                        "FROM attr_val WHERE attr_id=?", [attr_id, prev])
        for uid, val in pairs:
            vs, vn = (None, float(val)) if dtype == "numeric" else (str(val), None)
            con.execute("INSERT OR REPLACE INTO attr_val VALUES (?,?,?,?,?,?)",
                        [attr_id, uid, vs, vn, 1.0, grade])
        attr_ids.append(attr_id)
        if prev:
            prev_ids[attr_id] = prev
    return {"attr_ids": attr_ids, "prev_attr_ids": prev_ids, "anchors": anchors}


@op("assume")
def assume(con, world_id, p, op_id):
    """반사실 가정: 지정한 행들의 속성 하나를 값 하나로 덮어쓴다(등급 [가]). 같은 응답자에게 조건만 바꿔 묻는 감도 실험용. fork한 월드에서 쓴다.

    p = {attr, uuids:[...], value, grade:'[가]', dtype}
    """
    grade = validate(p.get("grade", "[가]"))
    existing = next((a for a in active_attrs(con, world_id) if a["name"] == p["attr"]), None)
    dtype = p.get("dtype", existing["dtype"] if existing else "categorical")
    attr_id, prev = _new_attr(con, world_id, p["attr"], dtype, existing and json.loads(existing["domain"]),
                              existing["grade"] if existing else grade, "assume", p.get("source_beliefs"), op_id)
    if prev:
        con.execute("INSERT INTO attr_val SELECT ?, uuid, v_str, v_num, confidence, grade FROM attr_val WHERE attr_id=?", [attr_id, prev])
    vs, vn = (None, float(p["value"])) if dtype == "numeric" else (str(p["value"]), None)
    con.executemany("INSERT OR REPLACE INTO attr_val VALUES (?,?,?,?,?,?)", [[attr_id, u, vs, vn, 1.0, grade] for u in p["uuids"]])
    return _result(attr_id, prev, n_assumed=len(p["uuids"]))


@op("fork")
def fork(con, world_id, p, op_id):
    """복제본 월드. attr_def·belief·anchor 정의만 복사(값은 공유, copy-on-write)."""
    src = get_world(con, world_id)
    new = create_world(con, p["name"], parent=world_id, weight_ref=src["weight_ref"])
    con.execute("INSERT INTO attr_def SELECT ?, attr_id, name, version, dtype, domain, grade, method, "
                "source_beliefs, op_id, active FROM attr_def WHERE world_id=? AND active", [new, world_id])
    con.execute("INSERT INTO belief SELECT ?, belief_id, proposition, variables, grade, source_ref, n, "
                "sample_condition, status, superseded_by, op_id FROM belief WHERE world_id=?", [new, world_id])
    con.execute("INSERT INTO anchor SELECT ?, anchor_id, source_type, source_ref, uuid, match_score, "
                "matched_on, observed, op_id FROM anchor WHERE world_id=?", [new, world_id])
    build_view(con, new)
    return {"new_world_id": new}
