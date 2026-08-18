
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import math
import io

import optimizer as opt

st.set_page_config(
    page_title="Sinter Burden Optimizer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# BACKEND INTERFACE — all optimization logic comes from optimizer.py
# ---------------------------------------------------------------------
TARGETS = opt.TARGETS
FE_LOWER = opt.FE_LOWER
FE_UPPER = opt.FE_UPPER
get_default_chemistry = opt.get_default_chemistry
load_chemistry_from_excel = opt.load_chemistry_from_excel
compute_achieved = opt.compute_achieved
solve_blend_with_compensation = opt.solve_blend_with_compensation
compute_coke_heat_balance_diagnostic = opt.compute_coke_heat_balance_diagnostic

# ---------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------
st.markdown("""
<style>
:root{
 --bg:#071017; --panel:#0d1b23; --panel2:#10232d; --line:#24404d;
 --text:#edf6f8; --muted:#8da5ae; --blue:#2f8ac0; --cyan:#29c7c9;
 --green:#16c784; --amber:#e6a63a; --red:#ef6262;
}
html,body,[class*="css"]{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
.block-container{padding-top:4.15rem;padding-bottom:2rem;max-width:1500px;}
[data-testid="stSidebar"]{background:#08131a;border-right:1px solid #1d3540;}
.panel{
 background:linear-gradient(180deg,#10232c,#0b171e);
 border:1px solid var(--line); border-radius:10px; padding:14px 16px;
 margin-bottom:12px;
}
.panel-title{font-size:.72rem;letter-spacing:.14em;font-weight:800;color:#79c8ec;text-transform:uppercase;}
.panel-sub{font-size:.72rem;color:var(--muted);margin-top:3px;}
.kpi{background:#0d1d25;border:1px solid #294653;border-radius:9px;padding:12px 14px;min-height:92px;}
.kpi-label{font-size:.62rem;letter-spacing:.13em;color:#8fb3c1;font-weight:800;}
.kpi-value{font-size:1.35rem;font-weight:800;margin-top:7px;color:#f4fbfc;}
.kpi-note{font-size:.65rem;color:#829aa4;margin-top:3px;}
.badge-ok{color:#35d995;border:1px solid #176d52;background:#09231c;padding:5px 9px;border-radius:999px;font-size:.68rem;font-weight:800;}
.badge-warn{color:#f0bd55;border:1px solid #745a21;background:#251e0c;padding:5px 9px;border-radius:999px;font-size:.68rem;font-weight:800;}
.badge-bad{color:#ff7777;border:1px solid #773737;background:#2a1114;padding:5px 9px;border-radius:999px;font-size:.68rem;font-weight:800;}
.notice{padding:9px 12px;border:1px solid #294653;background:#0c1b22;border-radius:8px;color:#a9c1ca;font-size:.78rem;}
.section-gap{height:5px;}
.small{font-size:.72rem;color:var(--muted);}
h1,h2,h3{letter-spacing:.02em;}
div[data-testid="stDataEditor"]{border:1px solid #27424e;border-radius:8px;overflow:hidden;}
div[data-testid="stDataFrame"]{border-radius:8px;}
button[kind="primary"]{border-radius:7px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
def init_state():
    if "df" not in st.session_state:
        st.session_state.df = get_default_chemistry().copy()
    if "source_name" not in st.session_state:
        st.session_state.source_name = "Built-in v30 Chemistry"
    if "result" not in st.session_state:
        st.session_state.result = None
    if "previous_cost" not in st.session_state:
        st.session_state.previous_cost = None
    if "inputs_changed" not in st.session_state:
        st.session_state.inputs_changed = False
    if "manual_base" not in st.session_state:
        st.session_state.manual_base = None
    if "manual_adjusted" not in st.session_state:
        st.session_state.manual_adjusted = None
    if "manual_req" not in st.session_state:
        st.session_state.manual_req = {}
    if "om_cost" not in st.session_state:
        st.session_state.om_cost = float(opt.DEFAULT_OM_COST_RS_T)
    if "production_t" not in st.session_state:
        st.session_state.production_t = 1000.0
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0
    if "dry_wet_page" not in st.session_state:
        st.session_state.dry_wet_page = "Dry & Wet"
    if "manual_reset_counter" not in st.session_state:
        st.session_state.manual_reset_counter = 0
    if "chem_editor_version" not in st.session_state:
        st.session_state.chem_editor_version = 0

init_state()

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def quality_checks(achieved):
    return {
        "Fe": FE_LOWER <= float(achieved.get("Fe", -999)) <= FE_UPPER,
        "SiO2": float(achieved.get("SiO2", 999)) <= TARGETS["SiO2_max"],
        "Al2O3": float(achieved.get("Al2O3", 999)) <= TARGETS["Al2O3_max"],
        "Al2O3/SiO2": float(achieved.get("Al2O3/SiO2", 999)) <= TARGETS["Al2O3_SiO2_max"],
        "Basicity": TARGETS["Basicity_min"] <= float(achieved.get("Basicity", -999)) <= TARGETS["Basicity_max"],
        "MgO": TARGETS["MgO_min"] <= float(achieved.get("MgO", -999)) <= TARGETS["MgO_max"],
        "CaO": TARGETS["CaO_min"] <= float(achieved.get("CaO", -999)) <= TARGETS["CaO_max"],
    }

def total_cost_with_om(rm_cost, om=None):
    return float(rm_cost or 0) + float(st.session_state.om_cost if om is None else om)

def material_cost(blend, df, wet=False):
    total = 0.0
    for m, q in blend.items():
        if m not in df.index:
            continue
        q = float(q or 0)
        if wet:
            moisture = float(df.loc[m].get("Moisture_Pct", 0) or 0) / 100
            q = q / (1-moisture) if moisture < 1 else q
        total += q * float(df.loc[m,"Price_Rs_t"]) / 1000
    return total

def burden_mass(blend, df, wet=False):
    total = 0.0
    for m,q in blend.items():
        q=float(q or 0)
        if wet:
            moisture=float(df.loc[m].get("Moisture_Pct",0) or 0)/100
            q=q/(1-moisture) if moisture < 1 else q
        total += q
    return total

def basis_table(blend, df, wet=False, include_om=True):
    rows=[]
    total_b=burden_mass(blend,df,wet)
    rm_cost=material_cost(blend,df,wet)
    total_cost=rm_cost+st.session_state.om_cost if include_om else rm_cost
    for m,q in blend.items():
        if m not in df.index or float(q or 0) <= 0:
            continue
        moisture=float(df.loc[m].get("Moisture_Pct",0) or 0)/100
        b=float(q)
        if wet:
            b=b/(1-moisture) if moisture < 1 else b
        c=b*float(df.loc[m,"Price_Rs_t"])/1000
        rows.append({
            "Material":m,
            "Group":str(df.loc[m,"Group"]),
            "Burden (kg/t)":b,
            "% Burden":(b/total_b*100 if total_b else 0),
            "Cost (₹/t)":c,
            "% Cost":(c/total_cost*100 if total_cost else 0),
        })
    if include_om:
        rows.append({
            "Material":"O&M",
            "Group":"Operating",
            "Burden (kg/t)":np.nan,
            "% Burden":np.nan,
            "Cost (₹/t)":float(st.session_state.om_cost),
            "% Cost":(float(st.session_state.om_cost)/total_cost*100 if total_cost else 0),
        })
    rows.append({
        "Material":"TOTAL",
        "Group":"",
        "Burden (kg/t)":total_b,
        "% Burden":100.0,
        "Cost (₹/t)":total_cost,
        "% Cost":100.0,
    })
    return pd.DataFrame(rows), rm_cost, total_b, total_cost

def clean_numeric_df(df):
    out=df.copy()
    for c in ["Fe","SiO2","Al2O3","CaO","MgO","LOI","Moisture_Pct","Tech_Min","Tech_Max","Available_Tonnes","Price_Rs_t"]:
        if c in out.columns:
            out[c]=pd.to_numeric(out[c],errors="coerce").fillna(0.0)
    return out

def active_df():
    return st.session_state.df.copy()

def reset_manual_baseline(blend):
    st.session_state.manual_base = blend.copy() if blend else None
    st.session_state.manual_adjusted = blend.copy() if blend else None
    st.session_state.manual_req = {}
    for k in list(st.session_state.keys()):
        if str(k).startswith("manual_"):
            # preserve dictionaries above; widget keys are manual_slider_
            pass
    for k in list(st.session_state.keys()):
        if str(k).startswith("manual_slider_"):
            del st.session_state[k]
    st.session_state.manual_reset_counter += 1

def run_optimizer():
    df=active_df()
    before=st.session_state.result["cost"] if st.session_state.result else None
    # Only the optimization engine is called here. Manual adjustment never calls this.
    r=solve_blend_with_compensation(
        df,
        float(st.session_state.production_t),
        TARGETS,
        baseline_blend=None,
        iol_nominal=opt.IOL_FINES_NOMINAL_PCT,
        bf_nominal=opt.BF_RETURNS_NOMINAL_PCT,
        iol_fb_min=opt.IOL_FINES_FALLBACK_MIN,
        iol_fb_max=opt.IOL_FINES_FALLBACK_MAX,
        bf_fb_min=opt.BF_RETURNS_FALLBACK_MIN,
        bf_fb_max=opt.BF_RETURNS_FALLBACK_MAX,
        coke_cv=st.session_state.coke_cv,
        coke_fc=st.session_state.coke_fc,
        latent_heat=st.session_state.latent_heat,
        calcination_heat=st.session_state.calc_heat,
        melting_heat=st.session_state.melt_heat,
        loss_fraction=st.session_state.loss_frac,
        firing_ratio_max=st.session_state.firing_ratio,
        enforce_firing_ratio_max=False,
        coke_min_rate=st.session_state.coke_min,
        coke_max_rate=st.session_state.coke_max,
        feo_min=st.session_state.feo_min,
        feo_target=st.session_state.feo_target,
        feo_max=st.session_state.feo_max,
        feo_ref_surplus=st.session_state.feo_ref_surplus,
        feo_ref_pct=st.session_state.feo_ref_pct,
        feo_thermal_slope=st.session_state.feo_slope,
        ref_coke_cv=opt.DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
        ref_coke_fc=opt.DEFAULT_REFERENCE_COKE_FC_PCT,
        manual_override=st.session_state.manual_coke_override,
        manual_coke_rate=st.session_state.manual_coke_rate,
    )
    st.session_state.previous_cost=before
    st.session_state.result={
        "status":r[0],"blend":r[1],"cost":r[2],"achieved":r[3],
        "diagnostics":r[4],"fallback":r[5],"df":df.copy()
    }
    st.session_state.inputs_changed=False
    reset_manual_baseline(r[1])
    st.session_state.run_count += 1

def apply_chemistry_editor(edited):
    edited=edited.copy()
    edited.index=edited["Material"]
    for m in edited.index:
        if m not in st.session_state.df.index:
            continue
        for c in ["Fe","SiO2","Al2O3","CaO","MgO","LOI","Moisture_Pct","Tech_Min","Tech_Max"]:
            if c in edited.columns:
                st.session_state.df.loc[m,c]=float(edited.loc[m,c])
    st.session_state.inputs_changed=True

def apply_commercial_editor(edited):
    edited=edited.copy()
    changed=False
    for _,r in edited.iterrows():
        m=r["Material"]
        if m not in st.session_state.df.index: continue
        vals={
            "Price_Rs_t":float(r["Price (₹/t)"]),
            "Available_Tonnes":float(r["RM Stock (t)"]),
            "Tech_Max":float(r["Tech Max (kg/t)"]),
            "Moisture_Pct":float(r["Moisture (%)"]),
        }
        for c,v in vals.items():
            if float(st.session_state.df.loc[m,c]) != v:
                st.session_state.df.loc[m,c]=v; changed=True
        # Availability is represented by Available_Tonnes=0 in the solver.
        av=bool(r["Available"])
        if not av and float(st.session_state.df.loc[m,"Available_Tonnes"]) != 0:
            st.session_state.df.loc[m,"Available_Tonnes"]=0; changed=True
        elif av and float(st.session_state.df.loc[m,"Available_Tonnes"]) == 0:
            # do not invent stock; user must enter a positive stock after enabling
            pass
    if changed:
        st.session_state.inputs_changed=True

def quality_status(achieved):
    if achieved is None: return "READY","ok"
    ok=all(quality_checks(achieved).values())
    return ("PASS","ok") if ok else ("REVIEW","bad")

def manual_adjusted_blend(base, df, requested):
    """Independent what-if calculator. Baseline is the latest optimizer output.
    Total burden is preserved by proportional redistribution among adjustable
    non-mandated materials. No optimizer solve is performed.
    """
    adjusted={m:float(v) for m,v in base.items()}
    adjustable=[m for m in base if m in df.index and df.loc[m,"Group"] in ("Iron_ore","Flux","Fuel")]
    fixed=[m for m in base if m not in adjustable]
    for m,v in requested.items():
        if m in adjustable: adjusted[m]=float(v)
    changed=next(iter(requested),None)
    # requested values are interpreted simultaneously; preserve total by scaling
    # the remaining adjustable pool to absorb the net delta.
    target_total=sum(float(v) for v in base.values())
    new_sum=sum(float(adjusted[m]) for m in adjustable)
    old_sum=sum(float(base[m]) for m in adjustable)
    delta=new_sum-old_sum
    if abs(delta)>1e-9:
        changed_set=set(requested.keys())
        others=[m for m in adjustable if m not in changed_set]
        if others:
            pool=sum(float(adjusted[m]) for m in others)
            if pool>0:
                for m in others:
                    adjusted[m]=max(0.0,float(adjusted[m])-delta*(float(adjusted[m])/pool))
        else:
            # If every adjustable material was edited, do not force a hidden
            # change: let the user see the resulting burden total.
            pass
    # Strict mandates/recycles stay at baseline.
    for m in fixed:
        adjusted[m]=float(base[m])
    return adjusted

# Default UI parameters
for key,val in {
    "coke_cv":opt.DEFAULT_COKE_CV_KCAL_KG,
    "coke_fc":opt.DEFAULT_COKE_FC_PCT,
    "latent_heat":opt.DEFAULT_HEAT_LATENT_MOISTURE,
    "calc_heat":opt.DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
    "melt_heat":opt.DEFAULT_HEAT_MELTING_PER_KG_SINTER,
    "loss_frac":opt.DEFAULT_HEAT_LOSS_FRACTION,
    "firing_ratio":opt.DEFAULT_FIRING_RATIO_MAX,
    "coke_min":opt.DEFAULT_COKE_MIN_KG_T,
    "coke_max":opt.DEFAULT_COKE_MAX_KG_T,
    "feo_min":opt.DEFAULT_FEO_MIN_PCT,
    "feo_target":opt.DEFAULT_FEO_TARGET_PCT,
    "feo_max":opt.DEFAULT_FEO_MAX_PCT,
    "feo_ref_surplus":opt.DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
    "feo_ref_pct":opt.DEFAULT_FEO_REFERENCE_PCT,
    "feo_slope":opt.DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
    "manual_coke_override":False,
    "manual_coke_rate":65.0,
}.items():
    if key not in st.session_state: st.session_state[key]=val

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏭 SINTER BURDEN OPTIMIZER")
    st.markdown('<div class="small">Hospet Steel Plant • v30 backend</div>',unsafe_allow_html=True)
    st.divider()

    nav=st.radio("WORKSPACE",[
        "Dashboard",
        "RM Stock & Inputs",
        "Dry & Wet Burden / Cost",
        "Optimization Results",
        "Manual Burden Control",
        "What-if Analysis",
        "Bottleneck Analysis",
        "Raw Material Chemistry",
        "Coke Parameters",
        "Upload & Settings",
    ],label_visibility="collapsed")

    st.divider()
    st.markdown("**O&M COST**")
    st.session_state.om_cost=st.number_input(
        "O&M Cost (₹/t)",min_value=0.0,value=float(st.session_state.om_cost),
        step=50.0,key="om_sidebar"
    )
    st.caption("O&M is added to the displayed total cost; it is not part of the LP material-cost objective.")

    if st.session_state.result:
        st.markdown("---")
        st.markdown("**LATEST RUN**")
        st.caption(f"Run #{st.session_state.run_count}")
        st.caption(f"Source: {st.session_state.source_name}")
        status,_=quality_status(st.session_state.result["achieved"])
        st.markdown(f"**Quality:** {status}")

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
h1,h2,h3=st.columns([5.7,1.6,1.3])
with h1:
    st.markdown("# SINTER BURDEN OPTIMIZER")
    st.markdown('<div class="small">Cost-optimal sinter mix • quality assurance • wet/dry costing • independent manual what-if</div>',unsafe_allow_html=True)
with h2:
    status,kind=quality_status(st.session_state.result["achieved"] if st.session_state.result else None)
    st.markdown(f'<div class="badge-{kind}" style="text-align:center;margin-top:10px">{status}</div>',unsafe_allow_html=True)
with h3:
    now=datetime.now()
    st.markdown(f'<div class="small" style="text-align:right;margin-top:10px"><b>PLANT: HOSPET</b><br>{now:%d %b %Y}<br>{now:%I:%M %p}</div>',unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------
def render_kpis(result):
    if not result or result["blend"] is None:
        return
    a=result["achieved"] or {}
    rm_dry=material_cost(result["blend"],result["df"],False)
    total=rm_dry+st.session_state.om_cost
    burden=sum(float(v) for v in result["blend"].values())
    ok=all(quality_checks(a).values())
    prev=st.session_state.previous_cost
    delta=(total-(float(prev)+st.session_state.om_cost)) if prev is not None else None
    cols=st.columns(5,gap="medium")
    data=[
        ("OPTIMIZED COST",f"₹{total:,.2f}/t","RM + O&M"),
        ("TOTAL BURDEN",f"{burden:,.1f} kg/t","LP dry basis"),
        ("FE ACHIEVED",f"{a.get('Fe',0):.3f}%","Target 54.0 ± 0.3%"),
        ("QUALITY","PASS" if ok else "REVIEW","All mandatory chemistry"),
        ("RUN",f"#{st.session_state.run_count}",f"{delta:+,.2f} ₹/t vs prior" if delta is not None else "Current run"),
    ]
    for c,(lab,val,note) in zip(cols,data):
        with c:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

def dashboard_page():
    result=st.session_state.result
    c1,c2,c3=st.columns([4.5,1.5,1.5])
    with c1:
        st.markdown(f'<div class="panel"><div class="panel-title">ACTIVE DATA SOURCE</div><b>{st.session_state.source_name}</b><div class="small">{len(st.session_state.df)} materials • BF Returns + IOL Fines included</div></div>',unsafe_allow_html=True)
    with c2:
        st.session_state.production_t=st.number_input("Production (t)",min_value=1.0,value=float(st.session_state.production_t),step=100.0,key="prod_dash")
    with c3:
        if st.button("🚀 RUN OPTIMIZER",type="primary",use_container_width=True):
            with st.spinner("Running PuLP / CBC v30..."):
                try:
                    run_optimizer()
                    st.rerun()
                except Exception as e:
                    st.error(f"Optimizer error: {e}")

    if st.session_state.inputs_changed:
        st.markdown('<div class="notice">✏ Inputs changed. The displayed optimization remains the last run until you press RUN OPTIMIZER.</div>',unsafe_allow_html=True)

    if result and result["blend"] is not None:
        render_kpis(result)
        st.write("")
        left,right=st.columns([1.05,1.25],gap="medium")
        with left:
            st.markdown('<div class="panel"><div class="panel-title">OPTIMIZED BURDEN</div><div class="panel-sub">Same material sequence as the backend output.</div>',unsafe_allow_html=True)
            rows=[]
            total=sum(result["blend"].values())
            for m,q in result["blend"].items():
                if m in result["df"].index and q>0:
                    rows.append({"Material":m,"Group":result["df"].loc[m,"Group"],"kg/t":q,"% Burden":q/total*100 if total else 0})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=min(430,58+len(rows)*36))
            st.markdown('</div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel"><div class="panel-title">CHEMISTRY ACHIEVED</div><div class="panel-sub">Backend-calculated quality values.</div>',unsafe_allow_html=True)
            a=result["achieved"]
            qdf=pd.DataFrame({
                "Parameter":["Fe","SiO₂","Al₂O₃","Al₂O₃/SiO₂","Basicity","MgO","CaO","B4"],
                "Achieved":[a["Fe"],a["SiO2"],a["Al2O3"],a["Al2O3/SiO2"],a["Basicity"],a["MgO"],a["CaO"],a["B4"]],
                "Target":["53.7–54.3","≤5.8","≤4.5","≤0.98","1.9–2.0","2.2–2.4","10.5–11.5","1.8–2.2 info"],
            })
            st.dataframe(qdf,use_container_width=True,hide_index=True,height=350)
            st.markdown('</div>',unsafe_allow_html=True)
        if result["diagnostics"]:
            with st.expander("Solver diagnostics"):
                for d in result["diagnostics"]:
                    st.write(d)
    else:
        st.info("No optimization result yet. Edit inputs if required and click RUN OPTIMIZER.")

def rm_stock_page():
    st.markdown("## RM Stock & Inputs")
    st.markdown('<div class="small">Commercial values are dashboard inputs. Chemistry remains in the separate chemistry workspace.</div>',unsafe_allow_html=True)
    df=st.session_state.df
    rows=[]
    for m in df.index:
        rows.append({
            "Material":m,"Group":str(df.loc[m,"Group"]),
            "Price (₹/t)":float(df.loc[m,"Price_Rs_t"]),
            "RM Stock (t)":float(df.loc[m,"Available_Tonnes"]),
            "Tech Max (kg/t)":float(df.loc[m,"Tech_Max"]),
            "Moisture (%)":float(df.loc[m].get("Moisture_Pct",0)),
            "Available":float(df.loc[m,"Available_Tonnes"])>0,
        })
    ed=st.data_editor(pd.DataFrame(rows),key="commercial_editor",use_container_width=True,hide_index=True,
                      disabled=["Material","Group"],column_config={
                          "Price (₹/t)":st.column_config.NumberColumn(format="%.2f"),
                          "RM Stock (t)":st.column_config.NumberColumn(format="%.2f"),
                          "Tech Max (kg/t)":st.column_config.NumberColumn(format="%.2f"),
                          "Moisture (%)":st.column_config.NumberColumn(format="%.2f"),
                          "Available":st.column_config.CheckboxColumn(),
                      })
    if st.button("APPLY RM INPUT CHANGES",type="primary"):
        apply_commercial_editor(ed)
        st.success("Inputs applied. Run optimizer to update the solution.")
        st.rerun()

def chemistry_page():
    st.markdown("## Raw Material Chemistry")
    st.markdown('<div class="small">Editable chemistry and moisture master. These values feed the v30 backend on the next run.</div>',unsafe_allow_html=True)
    df=st.session_state.df
    rows=[]
    for m in df.index:
        rows.append({
            "Material":m,"Group":str(df.loc[m,"Group"]),
            "Fe":float(df.loc[m,"Fe"]),"SiO2":float(df.loc[m,"SiO2"]),
            "Al2O3":float(df.loc[m,"Al2O3"]),"CaO":float(df.loc[m,"CaO"]),
            "MgO":float(df.loc[m,"MgO"]),"LOI":float(df.loc[m,"LOI"]),
            "Moisture_Pct":float(df.loc[m].get("Moisture_Pct",0)),
            "Tech_Min":float(df.loc[m,"Tech_Min"]),"Tech_Max":float(df.loc[m,"Tech_Max"]),
        })
    ed=st.data_editor(pd.DataFrame(rows),key=f"chem_editor_{st.session_state.chem_editor_version}",
                      use_container_width=True,hide_index=True,
                      disabled=["Material","Group"],
                      column_config={"Moisture_Pct":st.column_config.NumberColumn("Moisture (%)",format="%.2f")})
    if st.button("APPLY CHEMISTRY CHANGES",type="primary"):
        apply_chemistry_editor(ed)
        st.success("Chemistry changes applied.")
        st.rerun()

def dry_wet_page():
    st.markdown("## Dry & Wet Burden / Cost")
    st.markdown('<div class="small">Both views contain burden composition + burden percentage + cost composition + cost percentage. O&M is included in the displayed total cost percentage.</div>',unsafe_allow_html=True)
    if not st.session_state.result or not st.session_state.result["blend"]:
        st.info("Run optimizer first."); return
    result=st.session_state.result
    dry,_,_,_=basis_table(result["blend"],result["df"],wet=False,include_om=True)
    wet,_,_,_=basis_table(result["blend"],result["df"],wet=True,include_om=True)
    a,b=st.columns(2,gap="medium")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">DRY BASIS</div><div class="panel-sub">LP-native burden basis.</div>',unsafe_allow_html=True)
        st.dataframe(dry,use_container_width=True,hide_index=True,height=520)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">WET / AS-RECEIVED BASIS</div><div class="panel-sub">Moisture-adjusted purchase burden and cost.</div>',unsafe_allow_html=True)
        st.dataframe(wet,use_container_width=True,hide_index=True,height=520)
        st.markdown('</div>',unsafe_allow_html=True)

def results_page():
    st.markdown("## Optimization Results")
    r=st.session_state.result
    if not r or not r["blend"]:
        st.info("Run optimizer first."); return
    render_kpis(r)
    st.write("")
    dry,rm,total_b,total= basis_table(r["blend"],r["df"],wet=False,include_om=True)
    st.markdown('<div class="panel"><div class="panel-title">OPTIMIZED BURDEN + COST</div>',unsafe_allow_html=True)
    st.dataframe(dry,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)
    heat=compute_coke_heat_balance_diagnostic(
        r["blend"],r["df"],1000,
        st.session_state.coke_cv,st.session_state.coke_fc,
        st.session_state.latent_heat,st.session_state.calc_heat,
        st.session_state.melt_heat,st.session_state.loss_frac,
        st.session_state.feo_min,st.session_state.feo_target,st.session_state.feo_max,
        st.session_state.feo_ref_surplus,st.session_state.feo_ref_pct,st.session_state.feo_slope
    )
    if heat:
        with st.expander("Coke / FeO diagnostic"):
            st.write({k:v for k,v in heat.items() if k not in ("Controller_Suggestion","note")})
            st.info(heat["Controller_Suggestion"])

def manual_page():
    st.markdown("## Manual Burden Control")
    r=st.session_state.result
    if not r or not r["blend"]:
        st.info("Run optimizer first."); return
    df=r["df"]
    # This baseline is frozen when RUN OPTIMIZER is pressed.
    if st.session_state.manual_base is None:
        reset_manual_baseline(r["blend"])
    base=st.session_state.manual_base.copy()
    adjustable=[m for m in base if m in df.index and df.loc[m,"Group"] in ("Iron_ore","Flux","Fuel")]
    fixed=[m for m in base if m in df.index and df.loc[m,"Group"] not in ("Iron_ore","Flux","Fuel")]
    st.markdown('<div class="notice">This is an independent what-if layer. The latest optimized blend is the baseline. Changing sliders does NOT rerun the optimizer and does NOT change the optimizer result. IOL Fines, BF Returns and recycle materials stay at baseline.</div>',unsafe_allow_html=True)

    cols=st.columns(2,gap="medium")
    req={}
    for i,m in enumerate(adjustable):
        b=float(base[m])
        if df.loc[m,"Group"]=="Iron_ore": rnge=.15
        elif df.loc[m,"Group"]=="Flux": rnge=.10
        else: rnge=.10
        mn=max(0.0,b*(1-rnge))
        mx=max(mn+0.5,b*(1+rnge))
        key=f"manual_slider_{m}"
        current=float(st.session_state.get(key,b))
        current=max(mn,min(current,mx))
        with cols[i%2]:
            req[m]=st.slider(f"{m} — baseline {b:.1f} kg/t",mn,mx,current,0.5,key=key)
    if st.button("↺ RESET TO OPTIMIZED BASELINE"):
        reset_manual_baseline(base)
        st.rerun()

    adjusted=manual_adjusted_blend(base,df,req)
    st.session_state.manual_adjusted=adjusted
    ach=compute_achieved(adjusted,df,1000)
    rm_dry=material_cost(adjusted,df,False)
    total_cost=rm_dry+st.session_state.om_cost
    base_rm=material_cost(base,df,False)
    base_total=base_rm+st.session_state.om_cost
    total_b=sum(adjusted.values())
    checks=quality_checks(ach)
    cc=st.columns(5)
    vals=[
        ("BASE COST",f"₹{base_total:,.2f}/t","Optimized baseline"),
        ("MANUAL COST",f"₹{total_cost:,.2f}/t",f"{total_cost-base_total:+,.2f} ₹/t"),
        ("BURDEN",f"{total_b:,.1f} kg/t",f"Baseline {sum(base.values()):,.1f}"),
        ("Fe",f"{ach['Fe']:.3f}%","Manual"),
        ("QUALITY","PASS" if all(checks.values()) else "REVIEW","Manual only"),
    ]
    for c,(l,v,n) in zip(cc,vals):
        with c: st.markdown(f'<div class="kpi"><div class="kpi-label">{l}</div><div class="kpi-value">{v}</div><div class="kpi-note">{n}</div></div>',unsafe_allow_html=True)
    st.write("")
    a,b=st.columns([1.3,1],gap="medium")
    with a:
        rows=[]
        for m,q in adjusted.items():
            if m in df.index and q>0:
                rows.append({"Material":m,"Group":df.loc[m,"Group"],"Baseline kg/t":base[m],"Manual kg/t":q,"Change kg/t":q-base[m],"Change %":((q-base[m])/base[m]*100 if base[m] else 0)})
        st.markdown('<div class="panel"><div class="panel-title">MANUAL BURDEN</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        qdf=pd.DataFrame({"Parameter":list(ach.keys()),"Manual":list(ach.values()),"Status":[("PASS" if checks.get(k,True) else "REVIEW") for k in ach.keys()]})
        st.markdown('<div class="panel"><div class="panel-title">MANUAL CHEMISTRY IMPACT</div>',unsafe_allow_html=True)
        st.dataframe(qdf,use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="small">The optimizer result remains untouched. This section is specifically for reconciling theoretical LP behavior with practical operator adjustments.</div>',unsafe_allow_html=True)

def whatif_page():
    st.markdown("## What-if Analysis")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run optimizer first."); return
    df=r["df"]
    base_cost=material_cost(r["blend"],df,False)+st.session_state.om_cost
    mats=[m for m in df.index if df.loc[m,"Group"] in ("Iron_ore","Flux","IOL_Fines_Mandate","BF_Returns_Mandate") and df.loc[m,"Available_Tonnes"]>0]
    selected=st.multiselect("Materials to test as unavailable",mats,default=[])
    if st.button("RUN SELECTED SCENARIOS",type="primary"):
        results=[]
        for mat in selected:
            d=df.copy(); d.loc[mat,"Available_Tonnes"]=0
            rr=solve_blend_with_compensation(
                d,1000,TARGETS,
                coke_cv=st.session_state.coke_cv,coke_fc=st.session_state.coke_fc,
                latent_heat=st.session_state.latent_heat,calcination_heat=st.session_state.calc_heat,
                melting_heat=st.session_state.melt_heat,loss_fraction=st.session_state.loss_frac,
                coke_min_rate=st.session_state.coke_min,coke_max_rate=st.session_state.coke_max,
                feo_min=st.session_state.feo_min,feo_target=st.session_state.feo_target,feo_max=st.session_state.feo_max,
                feo_ref_surplus=st.session_state.feo_ref_surplus,feo_ref_pct=st.session_state.feo_ref_pct,
                feo_thermal_slope=st.session_state.feo_slope
            )
            status,blend,cost,ach,diag,fallback=rr
            if blend:
                tc=material_cost(blend,d,False)+st.session_state.om_cost
                results.append({"Missing":mat,"Status":status,"Total Cost ₹/t":tc,"Change ₹/t":tc-base_cost,
                                "Fe %":ach.get("Fe",np.nan),"Fallback":fallback})
            else:
                results.append({"Missing":mat,"Status":status,"Total Cost ₹/t":np.nan,"Change ₹/t":np.nan,"Fe %":np.nan,"Fallback":fallback})
        st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True)

def bottleneck_page():
    st.markdown("## Bottleneck Analysis")
    r=st.session_state.result
    if not r or not r["achieved"]: st.info("Run optimizer first."); return
    a=r["achieved"]
    rows=[
        ("Fe",a["Fe"],FE_LOWER,FE_UPPER),
        ("SiO₂",a["SiO2"],None,TARGETS["SiO2_max"]),
        ("Al₂O₃",a["Al2O3"],None,TARGETS["Al2O3_max"]),
        ("Al₂O₃/SiO₂",a["Al2O3/SiO2"],None,TARGETS["Al2O3_SiO2_max"]),
        ("Basicity",a["Basicity"],TARGETS["Basicity_min"],TARGETS["Basicity_max"]),
        ("MgO",a["MgO"],TARGETS["MgO_min"],TARGETS["MgO_max"]),
        ("CaO",a["CaO"],TARGETS["CaO_min"],TARGETS["CaO_max"]),
    ]
    out=[]
    for p,v,lo,hi in rows:
        if lo is None: margin=hi-v; status="PASS" if margin>=0 else "OUT"
        else: margin=min(v-lo,hi-v); status="PASS" if margin>=0 else "OUT"
        out.append({"Parameter":p,"Achieved":v,"Lower":lo if lo is not None else "—","Upper":hi,"Margin":margin,"Status":status})
    st.dataframe(pd.DataFrame(out),use_container_width=True,hide_index=True)

def coke_page():
    st.markdown("## Coke Parameters")
    st.markdown('<div class="notice">These controls map directly to the v30 backend parameters. The firing-ratio upper limit remains diagnostic-only, exactly as configured in the backend.</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.session_state.coke_cv=st.number_input("Coke CV (kcal/kg)",value=float(st.session_state.coke_cv),step=50.0)
        st.session_state.coke_fc=st.number_input("Fixed Carbon (%)",value=float(st.session_state.coke_fc),step=.5)
        st.session_state.coke_min=st.number_input("Coke Min (kg/t)",value=float(st.session_state.coke_min),step=1.0)
        st.session_state.coke_max=st.number_input("Coke Max (kg/t)",value=float(st.session_state.coke_max),step=1.0)
    with c2:
        st.session_state.latent_heat=st.number_input("Latent Heat (kcal/kg H₂O)",value=float(st.session_state.latent_heat),step=10.0)
        st.session_state.calc_heat=st.number_input("Calcination Heat (kcal/kg LOI)",value=float(st.session_state.calc_heat),step=10.0)
        st.session_state.melt_heat=st.number_input("Melting Heat (kcal/kg sinter)",value=float(st.session_state.melt_heat),step=5.0)
        st.session_state.loss_frac=st.number_input("Heat Loss Fraction",value=float(st.session_state.loss_frac),min_value=0.0,max_value=.99,step=.01)
    with c3:
        st.session_state.feo_min=st.number_input("FeO Min (%)",value=float(st.session_state.feo_min),step=.1)
        st.session_state.feo_target=st.number_input("FeO Target (%)",value=float(st.session_state.feo_target),step=.1)
        st.session_state.feo_max=st.number_input("FeO Max (%)",value=float(st.session_state.feo_max),step=.1)
        st.session_state.feo_ref_surplus=st.number_input("FeO Reference Thermal Surplus",value=float(st.session_state.feo_ref_surplus),step=5000.0)
        st.session_state.feo_ref_pct=st.number_input("FeO Reference (%)",value=float(st.session_state.feo_ref_pct),step=.1)
        st.session_state.feo_slope=st.number_input("FeO Thermal Slope (%/10k kcal)",value=float(st.session_state.feo_slope),step=.05)
    st.markdown("---")
    st.session_state.manual_coke_override=st.checkbox("Manual coke override",value=bool(st.session_state.manual_coke_override))
    st.session_state.manual_coke_rate=st.number_input("Forced Coke Rate (kg/t)",min_value=0.0,max_value=200.0,value=float(st.session_state.manual_coke_rate),step=.5,disabled=not st.session_state.manual_coke_override)
    st.caption("Changing parameters does not retroactively change the last result. Press RUN OPTIMIZER after changes.")

def settings_page():
    st.markdown("## Upload & Settings")
    a,b=st.columns([1.5,1],gap="medium")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">MASTER CHEMISTRY EXCEL</div>',unsafe_allow_html=True)
        up=st.file_uploader("Upload .xlsx",type=["xlsx"],key="master_upload")
        if up:
            try:
                loaded=load_chemistry_from_excel({up.name:up.getvalue()})
                st.success(f"{len(loaded)} materials loaded.")
                if st.button("ACTIVATE UPLOADED MASTER",type="primary",use_container_width=True):
                    st.session_state.df=clean_numeric_df(loaded)
                    st.session_state.source_name=f"Uploaded • {up.name}"
                    st.session_state.result=None
                    st.session_state.inputs_changed=False
                    reset_manual_baseline(None)
                    st.rerun()
            except Exception as e:
                st.error(f"Excel error: {e}")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">SYSTEM</div>',unsafe_allow_html=True)
        if st.button("↺ RESTORE BUILT-IN v30 MASTER",use_container_width=True):
            st.session_state.df=get_default_chemistry().copy()
            st.session_state.source_name="Built-in v30 Chemistry"
            st.session_state.result=None
            st.session_state.inputs_changed=False
            reset_manual_baseline(None)
            st.rerun()
        st.markdown('<div class="notice">BF_Returns and IOL_Fines are part of the v30 master and are kept as strict 17% / 8% mandates by the backend.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------
pages={
    "Dashboard":dashboard_page,
    "RM Stock & Inputs":rm_stock_page,
    "Dry & Wet Burden / Cost":dry_wet_page,
    "Optimization Results":results_page,
    "Manual Burden Control":manual_page,
    "What-if Analysis":whatif_page,
    "Bottleneck Analysis":bottleneck_page,
    "Raw Material Chemistry":chemistry_page,
    "Coke Parameters":coke_page,
    "Upload & Settings":settings_page,
}
pages[nav]()
st.markdown('<div class="small" style="text-align:center;margin-top:20px">Sinter Burden Optimizer • Hospet Steel Plant • v30 backend integrated</div>',unsafe_allow_html=True)
