# Sinter Burden Optimizer — v30 Dashboard

## Files
- `app.py` — Streamlit dashboard
- `optimizer.py` — cleaned v30 optimization engine supplied by the user
- `requirements.txt` — deployment dependencies

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important design decisions
- `optimizer.py` is the source of truth for the optimization logic and v30 chemistry/constraint definitions.
- `TARGETS` is explicitly exposed by `optimizer.py`; the dashboard does not assume a missing `opt.TARGETS`.
- IOL Fines = 8% and BF Returns = 17% remain strict backend mandates.
- Moisture is part of the material master and is used for wet/as-received costing and the coke heat-balance calculation.
- Dry and wet tables each show burden kg/t, burden %, cost Rs/t and cost %.
- O&M is editable and is added to displayed total cost.
- Coke/FeO/heat parameters are editable and passed to the v30 solver.
- Manual Burden Control is intentionally independent of the optimizer result: the latest optimizer output is frozen as the baseline, and slider changes calculate a separate practical what-if result without rerunning or modifying the LP result.
- Manual IOL Fines, BF Returns and recycle quantities stay at the optimized baseline so the strict mandates are not accidentally violated.
