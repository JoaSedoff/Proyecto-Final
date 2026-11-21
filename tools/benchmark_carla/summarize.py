import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "resultados/fid.jsonl")]
g = defaultdict(list)
for r in rows:
    g[(r["dt"], r["substep_dt"], r["max_substeps"])].append(r)
print("dt    sub    n  covers  speedup  cruces/h  rojo  choquesV/h choquesW/h  detencion_med  p95  pasados/muestras  volcados  vel")
for k in sorted(g):
    rs = g[k]
    n = len(rs)
    hrs = sum(r["sim_s"] for r in rs) / 3600
    cr = sum(r["crossings"] for r in rs)
    cv = sum(r["collisions_vehicle"] for r in rs)
    cw = sum(r["collisions_any"] - r["collisions_vehicle"] for r in rs)
    rr = sum(r["red_runs"] for r in rs)
    sp = [r["stop_pos_med"] for r in rs if r["stop_pos_med"] is not None]
    p95 = [r["stop_pos_p95"] for r in rs if r["stop_pos_p95"] is not None]
    past = sum(r["stop_past_line"] for r in rs)
    smp = sum(r["stop_samples"] for r in rs)
    print(f"{k[0]:<5} {k[1]:<6} {n}  {k[1]*k[2]:.3f}  {sum(r['speedup'] for r in rs)/n:6.2f}  {cr/hrs:7.1f}  {rr:3d}  {cv/hrs:8.1f}  {cw/hrs:8.1f}"
          f"   {sum(sp)/max(1,len(sp)):6.2f}  {max(p95) if p95 else float('nan'):5.2f}   {past}/{smp}   {sum(r['flipped'] for r in rs)}"
          f"   {sum(r['mean_speed_mps'] or 0 for r in rs)/n:.2f}")
