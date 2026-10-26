"""Оценка качества на размеченных договорах (evals/contracts + evals/gold.json).

Метрики:
- precision / recall / F1 по рискам (совпадение пункта и кода риска);
- обнаружение prompt injection и полнота на договорах с injection;
- доля выдуманных цитат, пойманных проверкой (rejected);
- откуда взята статья закона: агент через MCP или запасной вариант из чек-листа,
  и совпадает ли она с ожидаемыми статьями чек-листа;
- время и число вызовов LLM.
"""

import json
import time
from datetime import date
from pathlib import Path

from contract_review_agent.graph import INJECTION_CODE, build_graph, legal_kb_tools
from contract_review_agent.llm import make_llm, model_name

EVALS = Path(__file__).resolve().parents[2] / "evals"


def score(findings: list[dict], gold: dict) -> dict:
    predicted = {(f["clause_id"], f["code"]) for f in findings
                 if f["status"] == "verified" and f["code"] != INJECTION_CODE}
    matched, tp = set(), 0
    for risk in gold["risks"]:
        hit = next(((risk["clause"], c) for c in risk["codes"] if (risk["clause"], c) in predicted), None)
        if hit:
            tp += 1
            matched.add(hit)
    fp = sorted(predicted - matched)
    fn = [r for r in gold["risks"] if not any((r["clause"], c) in matched for c in r["codes"])]
    injected = sorted({f["clause_id"] for f in findings if f["code"] == INJECTION_CODE})
    return {"tp": tp, "fp": len(fp), "fn": len(fn), "gold": len(gold["risks"]),
            "false_positives": [f"{c}:{k}" for c, k in fp],
            "missed": [f"{r['clause']}:{'|'.join(r['codes'])}" for r in fn],
            "injection_expected": gold["injection_clauses"], "injection_found": injected}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


async def main(args) -> None:
    base = EVALS if args.set == "dev" else EVALS / args.set
    gold = json.loads((base / "gold.json").read_text(encoding="utf-8"))
    names = [n for n in gold if not args.only or args.only in n]
    llm = None if args.rules_only else make_llm()
    label = "rules-only" if args.rules_only else model_name()
    rows, laws = [], []
    async with legal_kb_tools() as tools:
        graph = build_graph(llm, tools)
        for name in names:
            text = (base / "contracts" / f"{name}.txt").read_text(encoding="utf-8")
            t0 = time.time()
            result = await graph.ainvoke({"contract": text})
            s = score(result["findings"], gold[name])
            s.update(name=name, seconds=round(time.time() - t0, 1),
                     llm_calls=result["stats"].get("llm_calls", 0),
                     llm_errors=result["stats"].get("llm_errors", 0),
                     rejected=len(result["rejected"]), retries=result["stats"].get("retries", 0))
            laws += [f["law"] for f in result["findings"] if f.get("law")]
            rows.append(s)
            p, r, f1 = prf(s["tp"], s["fp"], s["fn"])
            print(f"{name:34s} P={p:.2f} R={r:.2f} F1={f1:.2f}  {s['seconds']:>6.1f} с"
                  f"  пропущено={s['missed']}  лишнее={s['false_positives']}", flush=True)

    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    p, r, f1 = prf(tp, fp, fn)
    inj = [r for r in rows if r["injection_expected"]]
    inj_tp, inj_fp, inj_fn = (sum(x[k] for x in inj) for k in ("tp", "fp", "fn"))
    summary = {
        "set": args.set, "model": label, "date": date.today().isoformat(), "contracts": len(rows), "gold_risks": tp + fn,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "injection_detected": f"{sum(x['injection_found'] == x['injection_expected'] for x in inj)}/{len(inj)}",
        "recall_on_injected": round(prf(inj_tp, inj_fp, inj_fn)[1], 3) if inj else None,
        "false_alarm_injection": sum(bool(x["injection_found"]) for x in rows if not x["injection_expected"]),
        "quotes_rejected": sum(x["rejected"] for x in rows),
        "retries": sum(x["retries"] for x in rows),
        "law_by_agent": f"{sum(l['source'] == 'agent' for l in laws)}/{len(laws)}",
        "law_expected": f"{sum(l['expected'] for l in laws)}/{len(laws)}",
        "llm_calls": sum(x["llm_calls"] for x in rows), "llm_errors": sum(x["llm_errors"] for x in rows),
        "seconds_per_contract": round(sum(x["seconds"] for x in rows) / len(rows), 1),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    if not args.only:
        out = EVALS / "results" / f"{args.set}_{label.replace(':', '_').replace('/', '_')}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "contracts": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"Сохранено: {out}")
