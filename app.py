import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

import optimizer as opt

st.set_page_config(page_title="Sinter Burden Control", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

# ----------------------------- CONSTANTS ---------------------------------
TARGETS = opt.TARGETS
CHEM_COLS = ["Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Moisture_Pct"]
GROUPS = ["Iron_ore", "Flux", "Recycle", "Fuel"]
GROUP_LABEL = {"Iron_ore":"Iron Ore", "Flux":"Flux", "Recycle":"Recycle", "Fuel":"Fuel"}
ALT_NAME_FALLBACK = {"HAEMA", "BAUXA", "BAUXUA"}

# ----------------------------- STYLE -------------------------------------
st.markdown("""
<style>
:root { --bg:#071016; --panel:#0d1a21; --panel2:#111f27; --line:#28404d; --text:#edf5fa; --muted:#8ea6b4; --accent:#2f82b3; --good:#25c481; --warn:#f2b94b; --bad:#ff5555; }
html, body, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { width:210px !important; min-width:210px !important; background:#09131a; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { width:210px !important; }
.block-container { max-width:1450px; padding:1.2rem 1.4rem 2.5rem; }
.small { color:var(--muted); font-size:.72rem; }
.eyebrow { color:#73b5d7; font-size:.62rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
h1 { font-size:2rem !important; margin:.05rem 0 .15rem !important; letter-spacing:.01em; }
h2,h3 { letter-spacing:.01em; }
.panel { background:linear-gradient(180deg,#102029,#0d1a21); border:1px solid var(--line); border-radius:8px; padding:.65rem .7rem; margin:.55rem 0; }
.panel-title { color:#8fd0ef; font-size:.67rem; font-weight:900; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.35rem; }
.notice { background:#12242e; border:1px solid #31566a; border-radius:7px; padding:.6rem .75rem; color:#cce6f3; font-size:.76rem; }
.notice-w { border-color:#795b18; background:#241e0e; }
.hero { background:#0d1a21; border:1px solid var(--line); border-radius:8px; padding:.7rem .85rem; margin:.5rem 0 .7rem; }
.kpi { background:#101d24; border:1px solid var(--line); border-radius:8px; padding:.65rem .7rem; min-height:82px; }
.kpi-label { font-size:.58rem; color:#7da4b7; letter-spacing:.11em; font-weight:900; text-transform:uppercase; }
.kpi-value { font-size:1.15rem; font-weight:900; margin-top:.22rem; }
.kpi-sub { font-size:.6rem; color:#6f8998; margin-top:.12rem; }
.kpi-g { border-left:3px solid var(--good); } .kpi-r { border-left:3px solid var(--bad); } .kpi-a { border-left:3px solid var(--warn); } .kpi-s { border-left:3px solid #4ca8df; }
.nav-title { color:#7397a9; font-size:.56rem; font-weight:900; letter-spacing:.16em; margin:.75rem 0 .3rem; }
.sidebar-brand { font-weight:900; font-size:.86rem; letter-spacing:.03em; }
div.stButton > button { border-radius:6px; border:1px solid #2b4c5c; background:#101f27; color:#edf5fa; font-size:.72rem; min-height:2rem; }
div.stButton > button:hover { border-color:#4e94bb; color:white; }
button[kind="primary"] { background:#2c78a5 !important; border-color:#3e91c1 !important; }
[data-testid="stDataEditor"] { border:1px solid var(--line); border-radius:7px; overflow:hidden; }
[data-testid="stDataEditor"] [role="gridcell"] { font-size:12px; }
[data-testid="stFileUploader"] { background:#111a21; border-radius:7px; padding:.25rem; }
[data-testid="stMetric"] { background:#101d24; border:1px solid var(--line); border-radius:8px; padding:.45rem; }
.footer { color:#526d7b; font-size:.58rem; text-align:right; margin-top:1rem; }
[data-testid="stDataEditor"] [role="gridcell"], [data-testid="stDataFrame"] [role="gridcell"] {font-size:11px !important;}\n/* Paired dashboard tables: equal width + auto-fit height so every material row is visible. */
.dashboard-paired-table { width:100%; }
.dashboard-paired-table [data-testid="stDataEditor"],
.dashboard-paired-table [data-testid="stDataFrame"] { overflow:hidden !important; }
.dashboard-paired-table [data-testid="stDataEditor"] > div,
.dashboard-paired-table [data-testid="stDataFrame"] > div { overflow:hidden !important; }
.material-chart-note { color:#8ea6b4; font-size:.68rem; margin:.15rem 0 .45rem; }

</style>
""", unsafe_allow_html=True)

# ----------------------------- STATE -------------------------------------
def _classify_type(df):
    """Always return a master dataframe with a real Material_Type column.

    The user's existing master workbook does not need a Type/Material_Type
    column: HAEMA/BAUXA/BAUXUA are recognised as alternatives automatically.
    If a type column exists, it is used.
    """
    df = df.copy()

    # Find a supplied type/role/category column after normalising spaces/case.
    type_col = next(
        (c for c in df.columns
         if str(c).strip().lower().replace(" ", "_")
         in {"type", "material_type", "role", "material_role", "category"}),
        None,
    )

    if type_col is not None:
        vals = df[type_col].astype("string").fillna("").str.strip().str.lower()
        is_alt = vals.str.contains(r"alt|alternative|contingency", regex=True, na=False)
        df["Material_Type"] = np.where(is_alt, "Alternative", "Primary")
        # Do not leave duplicate type columns behind.
        drop_cols = [c for c in {type_col, "Material_Type"} if c in df.columns and c != "Material_Type"]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
    else:
        names = pd.Index(df.index.astype(str)).str.strip().str.upper()
        df["Material_Type"] = np.where(names.isin(ALT_NAME_FALLBACK), "Alternative", "Primary")

    # Defensive guarantee: downstream code may safely access this column.
    if "Material_Type" not in df.columns:
        df.insert(len(df.columns), "Material_Type", "Primary")
    df["Material_Type"] = df["Material_Type"].astype(str).str.strip().str.title()
    df.loc[~df["Material_Type"].isin(["Primary", "Alternative"]), "Material_Type"] = "Primary"
    return df

def normalize_master(raw):
    df = raw.copy()
    df.columns = [str(c).strip().replace(" ","_") for c in df.columns]
    # accept common aliases
    aliases = {
        "SiO₂":"SiO2", "SiO₂_%":"SiO2", "Al₂O₃":"Al2O3", "Al₂O₃_%":"Al2O3",
        "CaO_%":"CaO", "MgO_%":"MgO", "Price_₹/t":"Price_Rs_t", "Price_₹_t":"Price_Rs_t",
        "Price":"Price_Rs_t", "RM_Stock":"Available_Tonnes", "RM_Stock_t":"Available_Tonnes",
        "Stock":"Available_Tonnes", "Tech_Max_t/d":"Tech_Max", "Tech_Max_t_d":"Tech_Max",
        "Tech_Min_t/d":"Tech_Min", "Tech_Min_t_d":"Tech_Min", "Moisture":"Moisture_Pct", "Moisture_%":"Moisture_Pct", "Moisture_Pct":"Moisture_Pct"
    }
    df.rename(columns={c:aliases.get(c,c) for c in df.columns}, inplace=True)
    if "Material" in df.columns:
        df["Material"] = df["Material"].astype(str).str.strip()
        df = df.set_index("Material")
    elif df.index.name is None:
        raise ValueError("Master Excel must contain a Material column.")
    required = ["Group", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Tech_Min", "Tech_Max", "Price_Rs_t", "Available_Tonnes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Master Excel missing: " + ", ".join(missing))
    if "Moisture_Pct" not in df.columns:
        df["Moisture_Pct"] = 0.0
    df = _classify_type(df)
    df["Group"] = df["Group"].astype(str).str.strip()
    for c in CHEM_COLS + ["Tech_Min","Tech_Max","Price_Rs_t","Available_Tonnes"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for m in df[df["Group"]=="Recycle"].index:
        df.loc[m,"Tech_Max"] = df.loc[m,"Tech_Min"]
    if df.index.duplicated().any():
        raise ValueError("Duplicate material names found in the master Excel.")
    return df

def initial_df():
    df = opt.get_default_chemistry().copy()
    if "Moisture_Pct" not in df.columns:
        df["Moisture_Pct"] = 0.0
    if "Material_Type" not in df.columns:
        df["Material_Type"] = "Primary"
    return df

if "master_df" not in st.session_state:
    st.session_state.master_df = initial_df()
    st.session_state.source = "Built-in Master Chemistry"
    st.session_state.production = 1100.0
    st.session_state.available = {m: True for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Type"]=="Primary"}
    st.session_state.include_alt = {m: False for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Type"]=="Alternative"}
    st.session_state.result = None
    st.session_state.manual_base = None
    st.session_state.manual = None
    st.session_state.whatif = None
    st.session_state.runs = 0
    st.session_state.changed = False

# Cost / thermal controls introduced by the v30 backend.
def _backend_default(name, fallback):
    return float(getattr(opt, name, fallback))

if "om_cost" not in st.session_state:
    st.session_state.om_cost = _backend_default("DEFAULT_OM_COST_RS_T", 1500.0)
if "coke_cv" not in st.session_state:
    st.session_state.coke_cv = _backend_default("DEFAULT_COKE_CV_KCAL_KG", 6800.0)
if "coke_fc" not in st.session_state:
    st.session_state.coke_fc = _backend_default("DEFAULT_COKE_FC_PCT", 71.35)
if "latent_heat" not in st.session_state:
    st.session_state.latent_heat = _backend_default("DEFAULT_HEAT_LATENT_MOISTURE", 540.0)
if "calcination_heat" not in st.session_state:
    st.session_state.calcination_heat = _backend_default("DEFAULT_HEAT_CALCINATION_PER_LOI_KG", 420.0)
if "melting_heat" not in st.session_state:
    st.session_state.melting_heat = _backend_default("DEFAULT_HEAT_MELTING_PER_KG_SINTER", 60.0)
if "loss_fraction" not in st.session_state:
    st.session_state.loss_fraction = _backend_default("DEFAULT_HEAT_LOSS_FRACTION", 0.12)
if "firing_ratio_max" not in st.session_state:
    st.session_state.firing_ratio_max = _backend_default("DEFAULT_FIRING_RATIO_MAX", 1.10)
if "coke_min_rate" not in st.session_state:
    st.session_state.coke_min_rate = _backend_default("DEFAULT_COKE_MIN_KG_T", 55.0)
if "coke_max_rate" not in st.session_state:
    st.session_state.coke_max_rate = _backend_default("DEFAULT_COKE_MAX_KG_T", 85.0)
if "feo_min" not in st.session_state:
    st.session_state.feo_min = _backend_default("DEFAULT_FEO_MIN_PCT", 8.5)
if "feo_target" not in st.session_state:
    st.session_state.feo_target = _backend_default("DEFAULT_FEO_TARGET_PCT", 9.2)
if "feo_max" not in st.session_state:
    st.session_state.feo_max = _backend_default("DEFAULT_FEO_MAX_PCT", 10.0)
if "manual_coke_override" not in st.session_state:
    st.session_state.manual_coke_override = False
if "manual_coke_rate" not in st.session_state:
    st.session_state.manual_coke_rate = 65.0
if "nav" not in st.session_state: st.session_state.nav = "Dashboard"


# ------------------------- REDESIGN HELPERS -------------------------
def display_material_sequence(df):
    return [str(m) for m in df.index]

def aligned_result_table(blend, df, include_total=True):
    mats = display_material_sequence(df)
    total = float(sum(float(blend.get(m, 0.0)) for m in mats))
    total_cost = float(sum(float(blend.get(m, 0.0))*float(df.loc[m,"Price_Rs_t"])/1000 for m in mats))
    rows = []
    for m in mats:
        q = float(blend.get(m, 0.0))
        cost = q * float(df.loc[m,"Price_Rs_t"]) / 1000
        rows.append({
            "Material": m,
            "Group": GROUP_LABEL.get(df.loc[m,"Group"], df.loc[m,"Group"]),
            "kg/t": q,
            "% Burden": q/total*100 if total else 0.0,
            "Cost ₹/t": cost,
            "% Cost": cost/total_cost*100 if total_cost else 0.0
        })
    if include_total:
        rows.append({"Material":"TOTAL","Group":"","kg/t":total,"% Burden":100.0 if total else 0.0,
                     "Cost ₹/t":total_cost,"% Cost":100.0 if total_cost else 0.0})
    return pd.DataFrame(rows)

def paired_table_height(n):
    return max(320, 31*(n+1)+40)

# ----------------------------- DATA HELPERS -------------------------------
def primary_names():
    return [m for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Type"]=="Primary"]
def alt_names():
    return [m for m in st.session_state.master_df.index if st.session_state.master_df.loc[m,"Material_Type"]=="Alternative"]

def editable_chemistry():
    df=st.session_state.master_df.copy()
    cols=["Material_Type","Group"]+CHEM_COLS
    view=df[cols].reset_index().rename(columns={"Material":"Material","Material_Type":"Type","Group":"Group","SiO2":"SiO₂","Al2O3":"Al₂O₃", "Moisture_Pct":"Moisture %"})
    return view

def save_chemistry(view):
    v=view.copy().set_index("Material")
    v.rename(columns={"Type":"Material_Type","SiO₂":"SiO2","Al₂O₃":"Al2O3", "Moisture %":"Moisture_Pct"}, inplace=True)
    for m in v.index:
        if m in st.session_state.master_df.index:
            for c in CHEM_COLS:
                st.session_state.master_df.loc[m,c]=float(v.loc[m,c])
    st.session_state.changed=True

def active_df():
    # Alternatives OFF are physically removed from the solver dataframe so they do not
    # count as "missing iron ores" and do not trigger shortage tiers.
    df=st.session_state.master_df.copy()
    keep=[]
    for m in df.index:
        typ=df.loc[m,"Material_Type"]
        if typ=="Alternative":
            if st.session_state.include_alt.get(m,False): keep.append(m)
        else:
            keep.append(m)
    df=df.loc[keep].copy()
    for m in df.index:
        if df.loc[m,"Material_Type"]=="Primary" and not st.session_state.available.get(m,True):
            df.loc[m,"Available_Tonnes"]=0
        if df.loc[m,"Material_Type"]=="Alternative" and not st.session_state.include_alt.get(m,False):
            df.loc[m,"Available_Tonnes"]=0
    return df

def run_optimizer():
    df=active_df()
    kwargs={
        "om_cost": st.session_state.om_cost,
        "coke_cv": st.session_state.coke_cv,
        "coke_fc": st.session_state.coke_fc,
        "latent_heat": st.session_state.latent_heat,
        "calcination_heat": st.session_state.calcination_heat,
        "melting_heat": st.session_state.melting_heat,
        "loss_fraction": st.session_state.loss_fraction,
        "firing_ratio_max": st.session_state.firing_ratio_max,
        "coke_min_rate": st.session_state.coke_min_rate,
        "coke_max_rate": st.session_state.coke_max_rate,
        "feo_min": st.session_state.feo_min,
        "feo_target": st.session_state.feo_target,
        "feo_max": st.session_state.feo_max,
        "manual_override": st.session_state.manual_coke_override,
        "manual_coke_rate": st.session_state.manual_coke_rate,
    }
    # Keep compatibility with older backend revisions if any optional parameter is absent.
    try:
        x=opt.solve_blend_with_compensation(df, float(st.session_state.production), TARGETS, baseline_blend=None, **kwargs)
    except TypeError:
        x=opt.solve_blend_with_compensation(df, float(st.session_state.production), TARGETS, baseline_blend=None)
    st.session_state.result={"status":x[0],"blend":x[1],"cost":x[2],"achieved":x[3],"diagnostics":x[4],"fallback":x[5],"df":df.copy()}
    st.session_state.manual_base=x[1].copy() if x[1] else None
    st.session_state.manual=x[1].copy() if x[1] else None
    st.session_state.runs += 1
    st.session_state.changed=False
    st.session_state.whatif=None

def quality_ok(a):
    if not a: return False
    checks = [
        FE_LOWER := opt.FE_LOWER <= a["Fe"] <= opt.FE_UPPER,
        a["SiO2"] <= TARGETS["SiO2_max"], a["Al2O3"] <= TARGETS["Al2O3_max"],
        a["Al2O3/SiO2"] <= TARGETS["Al2O3_SiO2_max"],
        TARGETS["Basicity_min"] <= a["Basicity"] <= TARGETS["Basicity_max"],
        TARGETS["MgO_min"] <= a["MgO"] <= TARGETS["MgO_max"],
        TARGETS["CaO_min"] <= a["CaO"] <= TARGETS["CaO_max"],
    ]
    return all(checks)

def result_table(blend,df, include_zero=False):
    rows=[]
    if not blend: return pd.DataFrame(columns=["Material","Group","kg/t","% Burden","Cost ₹/t","% Cost"])
    total=sum(float(v) for v in blend.values())
    total_cost=sum(float(q)*float(df.loc[m,"Price_Rs_t"])/1000 for m,q in blend.items())
    sequence=list(df.index)
    for m in sequence:
        q=float(blend.get(m,0))
        if not include_zero and q<=1e-8: continue
        cost=q*float(df.loc[m,"Price_Rs_t"])/1000
        rows.append({"Material":m,"Group":GROUP_LABEL.get(df.loc[m,"Group"],df.loc[m,"Group"]),"kg/t":q,"% Burden":(q/total*100 if total else 0),"Cost ₹/t":cost,"% Cost":(cost/total_cost*100 if total_cost else 0)})
    rows.append({"Material":"TOTAL","Group":"","kg/t":total,"% Burden":100.0,"Cost ₹/t":total_cost,"% Cost":100.0})
    return pd.DataFrame(rows)

def kpi(label,value,sub="",kind="s"):
    return f'<div class="kpi kpi-{kind}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'

def material_contribution_chart(blend, df, kind="burden", height=430):
    """Material-level contribution visual for the optimized/manual recipe.

    Uses horizontal bars instead of group-level donuts so the user can see
    exactly which material is contributing to burden or cost.
    """
    rows = []
    for m, q in blend.items():
        q = float(q)
        if q <= 1e-9 or m not in df.index:
            continue
        price = float(df.loc[m, "Price_Rs_t"])
        value = q if kind == "burden" else q * price / 1000
        rows.append({
            "Material": str(m),
            "Group": str(df.loc[m, "Group"]),
            "Value": value
        })

    if not rows:
        return go.Figure()

    chart_df = pd.DataFrame(rows)
    total = float(chart_df["Value"].sum())
    chart_df["Pct"] = chart_df["Value"] / total * 100 if total else 0.0

    # Keep the largest contributors visually prominent.
    chart_df = chart_df.sort_values("Value", ascending=True)

    # Industrial palette selected to remain readable against the dark UI.
    group_colors = {
        "Iron_ore": "#6072f5",
        "Flux": "#f05a45",
        "Recycle": "#00c99b",
        "Fuel": "#a45bea",
    }

    fig = go.Figure()
    for group in GROUPS:
        part = chart_df[chart_df["Group"] == group]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            x=part["Value"],
            y=part["Material"],
            orientation="h",
            name=GROUP_LABEL.get(group, group),
            marker_color=group_colors.get(group, "#6f8794"),
            text=[f"{p:.1f}%" for p in part["Pct"]],
            textposition="outside",
            textfont=dict(size=11, color="#edf5fa"),
            customdata=np.column_stack([part["Pct"].to_numpy(), part["Group"].to_numpy()]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + ("Burden: %{x:.2f} kg/t" if kind == "burden" else "Cost: ₹%{x:.2f}/t")
                + "<br>Contribution: %{customdata[0]:.2f}%"
                + "<br>Group: %{customdata[1]}<extra></extra>"
            ),
        ))

    title = "MATERIAL CONTRIBUTION • % OF TOTAL BURDEN" if kind == "burden" else "MATERIAL CONTRIBUTION • % OF TOTAL COST"
    unit = "kg/t" if kind == "burden" else "₹/t"

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(height, 120 + 32 * len(chart_df)),
        margin=dict(l=105, r=65, t=38, b=45),
        barmode="stack",
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=12, color="#8fd0ef"),
        ),
        xaxis=dict(
            title=unit,
            gridcolor="#1e333f",
            zeroline=False,
            tickfont=dict(size=10, color="#9db1bc"),
            title_font=dict(size=10, color="#8ea6b4"),
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color="#edf5fa"),
            gridcolor="rgba(0,0,0,0)",
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=1,
            xanchor="right",
            font=dict(size=9, color="#cbd8df"),
        ),
        showlegend=True,
    )
    return fig

def quality_cards(ach):
    items=[("Fe",ach.get("Fe",np.nan),f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}","%"),("SiO₂",ach.get("SiO2",np.nan),f"≤ {TARGETS['SiO2_max']}","%"),("Al₂O₃",ach.get("Al2O3",np.nan),f"≤ {TARGETS['Al2O3_max']}","%"),("Al₂O₃/SiO₂",ach.get("Al2O3/SiO2",np.nan),f"≤ {TARGETS['Al2O3_SiO2_max']}",""),("Basicity",ach.get("Basicity",np.nan),f"{TARGETS['Basicity_min']}–{TARGETS['Basicity_max']}",""),("MgO",ach.get("MgO",np.nan),f"{TARGETS['MgO_min']}–{TARGETS['MgO_max']}","%"),("CaO",ach.get("CaO",np.nan),f"{TARGETS['CaO_min']}–{TARGETS['CaO_max']}","%"),("B4",ach.get("B4",np.nan),"1.8–2.2 info","")]
    cols=st.columns(8)
    for c,(lab,val,tgt,unit) in zip(cols,items):
        c.markdown(kpi(lab,f"{val:.3f}{unit}",tgt,"g" if lab=="B4" or (lab in ["Fe","SiO₂","Al₂O₃","Al₂O₃/SiO₂","Basicity","MgO","CaO"] and quality_ok(ach)) else "a"),unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown('<div class="eyebrow">HOSPET ALLOY STEEL PLANT</div>',unsafe_allow_html=True)
    st.markdown(f"<h1>{title}</h1><div class='small'>{subtitle}</div>",unsafe_allow_html=True)

# ----------------------------- SIDEBAR -----------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">HOSPET STEELS LIMITED</div><div class="small">Kalyani Steels × Mukand • Hospet</div>',unsafe_allow_html=True)
    st.markdown("---")
    nav_groups=[("WORKSPACE",["Dashboard"]),("OPERATIONS",["RM Stock","Optimization Results","Manual Burden Control","Alternative Raw Material"]),("ANALYSIS",["Burden Composition","Cost Composition","Cost Analysis","What-if Analysis","Bottleneck Analysis"]),("REPORTING",["Reports"]),("SYSTEM",["Upload & Settings"])]
    for head,items in nav_groups:
        st.markdown(f'<div class="nav-title">{head}</div>',unsafe_allow_html=True)
        for item in items:
            if st.button(item,key="nav_"+item,use_container_width=True,type="primary" if st.session_state.nav==item else "secondary"):
                st.session_state.nav=item; st.rerun()
    st.markdown("---")
    st.markdown(f'<div class="small"><b>DATA</b><br>{st.session_state.source}<br>{len(primary_names())} primary<br>{len(alt_names())} alternative<br><br><b>MODEL</b><br>v30 • Ready</div>',unsafe_allow_html=True)


# ----------------------------- COST / THERMAL HELPERS -----------------------
def dry_wet_tables(blend, df, om_cost):
    rows_dry=[]; rows_wet=[]
    total_dry=sum(float(blend.get(m,0.0)) for m in blend if m in df.index)
    total_wet=0.0; rm_dry=0.0; rm_wet=0.0
    for m in df.index:
        q=float(blend.get(m,0.0))
        if q < 0: q=0.0
        moisture=float(df.loc[m,"Moisture_Pct"]) if "Moisture_Pct" in df.columns else 0.0
        moisture=max(0.0,min(99.9,moisture))
        wet_q=q/(1.0-moisture/100.0) if q and moisture<100 else q
        price=float(df.loc[m,"Price_Rs_t"])
        dc=q*price/1000.0; wc=wet_q*price/1000.0
        rm_dry += dc; rm_wet += wc; total_wet += wet_q
        rows_dry.append({"Material":m,"Burden kg/t":q,"Burden %":q/total_dry*100 if total_dry else 0.0,"Cost ₹/t":dc})
        rows_wet.append({"Material":m,"Burden kg/t":wet_q,"Burden %":wet_q/total_wet*100 if total_wet else 0.0,"Cost ₹/t":wc})
    dry_total=rm_dry+float(om_cost); wet_total=rm_wet+float(om_cost)
    for rows, rm, total in [(rows_dry,rm_dry,dry_total),(rows_wet,rm_wet,wet_total)]:
        for r in rows: r["Cost %"]=r["Cost ₹/t"]/total*100 if total else 0.0
        rows.append({"Material":"O&M","Burden kg/t":np.nan,"Burden %":np.nan,"Cost ₹/t":float(om_cost),"Cost %":float(om_cost)/total*100 if total else 0.0})
        rows.append({"Material":"TOTAL","Burden kg/t":total_dry if rows is rows_dry else total_wet,"Burden %":100.0,"Cost ₹/t":total,"Cost %":100.0})
    return pd.DataFrame(rows_dry), pd.DataFrame(rows_wet), rm_dry, rm_wet, dry_total, wet_total

def coke_diagnostic(blend, df):
    fn=getattr(opt,"compute_coke_heat_balance_diagnostic",None)
    if not callable(fn) or not blend: return None
    try:
        return fn(blend,df,1000,st.session_state.coke_cv,st.session_state.coke_fc,st.session_state.latent_heat,
                  st.session_state.calcination_heat,st.session_state.melting_heat,st.session_state.loss_fraction,
                  st.session_state.feo_min,st.session_state.feo_target,st.session_state.feo_max,
                  getattr(opt,"DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL",189180.0),
                  getattr(opt,"DEFAULT_FEO_REFERENCE_PCT",8.6),
                  getattr(opt,"DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL",0.35),
                  getattr(opt,"DEFAULT_REFERENCE_COKE_CV_KCAL_KG",6800.0),
                  getattr(opt,"DEFAULT_REFERENCE_COKE_FC_PCT",71.35))
    except Exception:
        return None

def cost_analysis():
    page_header("Cost Analysis","Dry-basis process economics vs wet / as-received procurement economics.")
    r=st.session_state.result
    c1,c2,c3=st.columns(3)
    with c1:
        st.session_state.om_cost=st.number_input("O&M Cost ₹/t",min_value=0.0,value=float(st.session_state.om_cost),step=50.0,key="om_cost_input")
    with c2:
        st.metric("Cost Basis","Dry + Wet")
    with c3:
        st.caption("O&M is added to total sinter cost; it does not change the raw-material optimizer objective.")
    if not r or not r.get("blend"):
        st.info("Run the optimizer first."); return
    dry,wet,rm_dry,rm_wet,total_dry,total_wet=dry_wet_tables(r["blend"],r["df"],st.session_state.om_cost)
    a,b=st.columns(2,gap="small")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">DRY BASIS — BURDEN & COST COMPOSITION</div></div>',unsafe_allow_html=True)
        st.dataframe(dry,hide_index=True,use_container_width=True,height=max(360,34*len(dry)+45),column_config={"Burden kg/t":st.column_config.NumberColumn("Burden kg/t",format="%.2f"),"Burden %":st.column_config.NumberColumn("Burden %",format="%.2f"),"Cost ₹/t":st.column_config.NumberColumn("Cost ₹/t",format="₹ %.2f"),"Cost %":st.column_config.NumberColumn("Cost %",format="%.2f")})
        st.markdown(f'<div class="small">RM cost: ₹{rm_dry:,.2f}/t • O&M: ₹{st.session_state.om_cost:,.2f}/t • <b>Total dry cost: ₹{total_dry:,.2f}/t</b></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">WET / AS-RECEIVED — BURDEN & COST COMPOSITION</div></div>',unsafe_allow_html=True)
        st.dataframe(wet,hide_index=True,use_container_width=True,height=max(360,34*len(wet)+45),column_config={"Burden kg/t":st.column_config.NumberColumn("Burden kg/t",format="%.2f"),"Burden %":st.column_config.NumberColumn("Burden %",format="%.2f"),"Cost ₹/t":st.column_config.NumberColumn("Cost ₹/t",format="₹ %.2f"),"Cost %":st.column_config.NumberColumn("Cost %",format="%.2f")})
        st.markdown(f'<div class="small">RM procurement cost: ₹{rm_wet:,.2f}/t • O&M: ₹{st.session_state.om_cost:,.2f}/t • <b>Total wet cost: ₹{total_wet:,.2f}/t</b></div>',unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">🔥 COKE & THERMAL PARAMETERS</div><div class="small">These controls are passed to the v30 optimizer when available. Re-run the optimizer after changing them.</div></div>',unsafe_allow_html=True)
    q1,q2,q3,q4=st.columns(4)
    with q1: st.session_state.coke_cv=st.number_input("Coke CV (kcal/kg)",min_value=1000.0,value=float(st.session_state.coke_cv),step=50.0)
    with q2: st.session_state.coke_fc=st.number_input("Fixed Carbon (%)",min_value=1.0,max_value=100.0,value=float(st.session_state.coke_fc),step=.1)
    with q3: st.session_state.coke_min_rate=st.number_input("Coke Min (kg/t)",min_value=0.0,value=float(st.session_state.coke_min_rate),step=1.0)
    with q4: st.session_state.coke_max_rate=st.number_input("Coke Max (kg/t)",min_value=0.0,value=float(st.session_state.coke_max_rate),step=1.0)
    q5,q6,q7,q8=st.columns(4)
    with q5: st.session_state.feo_min=st.number_input("FeO Min (%)",min_value=0.0,value=float(st.session_state.feo_min),step=.1)
    with q6: st.session_state.feo_target=st.number_input("FeO Target (%)",min_value=0.0,value=float(st.session_state.feo_target),step=.1)
    with q7: st.session_state.feo_max=st.number_input("FeO Max (%)",min_value=0.0,value=float(st.session_state.feo_max),step=.1)
    with q8: st.session_state.manual_coke_override=st.checkbox("Manual Coke Override",value=bool(st.session_state.manual_coke_override))
    if st.session_state.manual_coke_override:
        st.session_state.manual_coke_rate=st.number_input("Fixed Coke Rate (kg/t)",min_value=float(st.session_state.coke_min_rate),max_value=float(st.session_state.coke_max_rate),value=float(st.session_state.manual_coke_rate),step=.5)
    if r.get("blend"):
        diag=coke_diagnostic(r["blend"],r["df"])
        if diag:
            x,y,z=st.columns(3)
            x.metric("Predicted FeO",f"{diag.get('FeO_Estimate_Pct',np.nan):.2f}%")
            y.metric("Thermal Surplus",f"{diag.get('Thermal_Surplus_kcal',diag.get('Thermal_Surplus',np.nan)):,.0f}")
            z.metric("Firing Ratio",f"{diag.get('Firing_Ratio',np.nan):.3f}")
            st.caption(str(diag.get("Controller_Suggestion", "")))

# ----------------------------- DASHBOARD ---------------------------------
def dashboard():
    page_header("SINTER BURDEN CONTROL","Cost optimization • quality assurance • raw material decision support")
    st.markdown('<div class="hero"><div class="panel-title">DATA CONTROL CENTER</div><div class="small">Upload one Master Chemistry Excel. Primary materials and optional alternative ores are read from the same workbook. Alternative ores remain excluded until the user explicitly enables them.</div></div>',unsafe_allow_html=True)
    up1,up2=st.columns([1.7,1])
    with up1:
        f=st.file_uploader("MASTER EXCEL • XLSX",type=["xlsx"],key="dash_master")
        if f is not None:
            try:
                newdf=normalize_master(pd.read_excel(f))
                if st.button("ACTIVATE MASTER",type="primary",key="activate_dash"):
                    st.session_state.master_df=newdf; st.session_state.source=f.name
                    st.session_state.available={m:True for m in primary_names()}; st.session_state.include_alt={m:False for m in alt_names()}
                    st.session_state.result=None; st.session_state.changed=False; st.session_state.runs=0
                    st.rerun()
                st.markdown(f'<div class="small">Detected {int((newdf["Material_Type"]=="Primary").sum())} primary • {int((newdf["Material_Type"]=="Alternative").sum())} alternative</div>',unsafe_allow_html=True)
            except Exception as e: st.error(str(e))
    with up2:
        st.markdown(f'<div class="notice"><b>ACTIVE MASTER</b><br>{st.session_state.source}<br>{len(primary_names())} primary • {len(alt_names())} alternative</div>',unsafe_allow_html=True)
    st.write("")
    c1,c2,c3=st.columns([1,1.6,1])
    with c1: st.session_state.production=st.number_input("Production (t)",min_value=1.0,value=float(st.session_state.production),step=10.0,key="prod")
    with c2:
        if st.button("🚀 RUN OPTIMIZER",type="primary",use_container_width=True):
            with st.spinner("Optimizing v30…"): run_optimizer()
            st.rerun()
    with c3: st.markdown(f'<div class="notice" style="text-align:center"><b>RUN #{st.session_state.runs}</b><br>{sum(st.session_state.include_alt.values())} alternative enabled</div>',unsafe_allow_html=True)
    if st.session_state.changed: st.markdown('<div class="notice notice-w">Inputs changed — run optimizer to apply.</div>',unsafe_allow_html=True)

    r=st.session_state.result
    if r and r["blend"]:
        bd=aligned_result_table(r["blend"],r["df"]); total=float(bd.iloc[-1]["kg/t"]); cost=float(bd.iloc[-1]["Cost ₹/t"]); ach=r["achieved"]; ok=quality_ok(ach); used_alt=[m for m in alt_names() if r["blend"].get(m,0)>0]
        cols=st.columns(6)
        cards=[("OPTIMIZED COST",f"₹{cost:,.2f}/t","Material cost","s"),("TOTAL BURDEN",f"{total:,.1f} kg/t","Raw burden","g"),("Fe",f"{ach['Fe']:.3f}%",f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}","a"),("QUALITY","PASS" if ok else "REVIEW","Backend targets","g" if ok else "r"),("ALT ORE","USED" if used_alt else "NOT USED","Contingency","a"),("RUN",f"#{st.session_state.runs}","v30","s")]
        for c,(l,v,s,k) in zip(cols,cards): c.markdown(kpi(l,v,s,k),unsafe_allow_html=True)
    else:
        cols=st.columns(5)
        for c,(l,v,s,k) in zip(cols,[("OPTIMIZED COST","—","Run optimizer","s"),("TOTAL BURDEN","—","Run optimizer","g"),("Fe","—","Target band","a"),("QUALITY","READY","Awaiting run","g"),("ALT ORE","NOT USED","OFF by default","a")]): c.markdown(kpi(l,v,s,k),unsafe_allow_html=True)

    # ACHIEVED CHEMISTRY
    st.markdown('<div class="panel"><div class="panel-title">CHEMISTRY ACHIEVED</div></div>',unsafe_allow_html=True)
    if r and r["achieved"]: quality_cards(r["achieved"])
    else: st.info("Chemistry achievement will appear after optimization.")

    # FULL-WIDTH EDITABLE CHEMISTRY
    st.markdown('<div class="panel"><div class="panel-title">RAW MATERIAL CHEMISTRY — EDITABLE</div><div class="small">Loaded from the same master Excel. Chemistry remains editable here and the edited values are used by the optimizer.</div></div>',unsafe_allow_html=True)
    chem=editable_chemistry()
    edited=st.data_editor(chem,key="chem_editor",hide_index=True,use_container_width=True,height=max(230,42*len(chem)+45),disabled=["Material","Type","Group"],column_config={"Material":st.column_config.TextColumn("Material"),"Type":st.column_config.TextColumn("Type"),"Group":st.column_config.TextColumn("Group"),**{c:st.column_config.NumberColumn("Moisture %" if c=="Moisture_Pct" else c,min_value=0,step=.01) for c in CHEM_COLS}})
    if not edited.equals(chem): save_chemistry(edited)

    # SIDE-BY-SIDE INPUT AND OUTPUT — SAME MATERIAL ORDER / SAME HEIGHT
    st.markdown('<div class="dashboard-paired-table">', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="small")

    # Both tables use the same material rows and an exact row-based height.
    # This prevents the default Streamlit grid from creating a vertical
    # scrollbar while keeping the two tables visually aligned.
    material_count = len(st.session_state.master_df.index)
    table_row_px = 34
    table_header_px = 38
    shared_h = max(210, table_header_px + table_row_px * material_count + 8)

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">RAW MATERIAL INPUTS</div>'
            '<div class="small">Same material sequence as the optimized table. '
            'Price, RM stock and Tech Max are editable here.</div></div>',
            unsafe_allow_html=True
        )

        inp = st.session_state.master_df.reset_index()[
            ["Material", "Material_Type", "Group", "Price_Rs_t", "Available_Tonnes", "Tech_Max"]
        ].copy()
        inp.rename(
            columns={
                "Material_Type": "Type",
                "Price_Rs_t": "Price ₹/t",
                "Available_Tonnes": "RM Stock t",
                "Tech_Max": "Tech Max"
            },
            inplace=True
        )

        edited_inp = st.data_editor(
            inp,
            key="input_editor_v24_static",
            hide_index=True,
            use_container_width=True,
            height=shared_h,
            disabled=["Material", "Type", "Group"],
            column_config={
                "Material": st.column_config.TextColumn("Material", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Group": st.column_config.TextColumn("Group", width="small"),
                "Price ₹/t": st.column_config.NumberColumn("Price ₹/t", min_value=0, step=1, format="%.0f", width="small"),
                "RM Stock t": st.column_config.NumberColumn("RM Stock t", min_value=0, step=100, format="%.0f", width="small"),
                "Tech Max": st.column_config.NumberColumn("Tech Max", min_value=0, step=1, format="%.0f", width="small"),
            }
        )

        if not edited_inp.equals(inp):
            for _, row in edited_inp.iterrows():
                m = row.Material
                st.session_state.master_df.loc[m, "Price_Rs_t"] = float(row["Price ₹/t"])
                st.session_state.master_df.loc[m, "Available_Tonnes"] = float(row["RM Stock t"])
                st.session_state.master_df.loc[m, "Tech_Max"] = float(row["Tech Max"])
            st.session_state.changed = True

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">OPTIMIZED BURDEN & COST</div>'
            '<div class="small">Exactly the same material sequence as the input table. '
            'Zero-use materials remain visible for direct row-by-row comparison.</div></div>',
            unsafe_allow_html=True
        )

        if r and r["blend"]:
            # No TOTAL row here: both dashboard tables contain exactly the
            # same material rows, so every row lines up directly.
            out = aligned_result_table(r["blend"], r["df"], include_total=False)
            st.dataframe(
                out,
                hide_index=True,
                use_container_width=True,
                height=shared_h,
                column_config={
                    "Material": st.column_config.TextColumn("Material", width="medium"),
                    "Group": st.column_config.TextColumn("Group", width="small"),
                    "kg/t": st.column_config.NumberColumn("kg/t", format="%.2f", width="small"),
                    "% Burden": st.column_config.NumberColumn("% Burden", format="%.2f", width="small"),
                    "Cost ₹/t": st.column_config.NumberColumn("Cost ₹/t", format="₹ %.2f", width="small"),
                    "% Cost": st.column_config.NumberColumn("% Cost", format="%.2f", width="small")
                }
            )
        else:
            st.info("Run the optimizer to populate the optimized burden and cost table.")

    st.markdown('</div>', unsafe_allow_html=True)

    # MATERIAL-LEVEL CONTRIBUTION VISUAL
    st.markdown(
        '<div class="panel"><div class="panel-title">MATERIAL CONTRIBUTION</div>'
        '<div class="small">Each bar represents an individual material. '
        'The percentage is that material’s contribution to total burden or total cost.</div></div>',
        unsafe_allow_html=True
    )

    if r and r["blend"]:
        df = r["df"]
        blend = r["blend"]
        a, b = st.columns(2)

        with a:
            st.markdown(
                '<div class="panel"><div class="panel-title">BURDEN CONTRIBUTION</div>'
                '<div class="material-chart-note">Material share of total burden • kg/t</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(
                material_contribution_chart(blend, df, "burden", height=max(360, 115 + 31 * len(blend))),
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with b:
            st.markdown(
                '<div class="panel"><div class="panel-title">COST CONTRIBUTION</div>'
                '<div class="material-chart-note">Material share of total material cost • ₹/t</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(
                material_contribution_chart(blend, df, "cost", height=max(360, 115 + 31 * len(blend))),
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run the optimizer to display material contribution.")

    if r and r.get("blend"):
        dry,wet,rm_dry,rm_wet,total_dry,total_wet=dry_wet_tables(r["blend"],r["df"],st.session_state.om_cost)
        s1,s2,s3=st.columns(3)
        s1.metric("Dry Total Cost",f"₹{total_dry:,.2f}/t")
        s2.metric("Wet / Procurement Cost",f"₹{total_wet:,.2f}/t")
        s3.metric("O&M",f"₹{st.session_state.om_cost:,.2f}/t")

# ----------------------------- OTHER PAGES -------------------------------
def rm_stock():
    page_header("RM Stock & Commercial Inputs","Daily material availability, price, stock and technical maximums.")
    st.markdown('<div class="notice">Availability, price, RM stock and Tech Max are editable here. Chemistry is maintained on the Dashboard master table.</div>',unsafe_allow_html=True)
    # reuse same input editor through a simplified full-width view
    inp=st.session_state.master_df.reset_index()[["Material","Material_Type","Group","Price_Rs_t","Available_Tonnes","Tech_Max"]].copy(); inp.rename(columns={"Material_Type":"Type","Price_Rs_t":"Price ₹/t","Available_Tonnes":"RM Stock t","Tech_Max":"Tech Max t/d"},inplace=True)
    inp["Availability / Include"]=[st.session_state.include_alt.get(m,False) if st.session_state.master_df.loc[m,"Material_Type"]=="Alternative" else st.session_state.available.get(m,True) for m in inp.Material]
    ed=st.data_editor(inp,key="rm_editor",hide_index=True,use_container_width=True,height=max(250,42*len(inp)+45),disabled=["Material","Type","Group"])
    if not ed.equals(inp):
        for _,row in ed.iterrows():
            m=row.Material; isalt=st.session_state.master_df.loc[m,"Material_Type"]=="Alternative"; st.session_state.master_df.loc[m,"Price_Rs_t"]=float(row["Price ₹/t"]); st.session_state.master_df.loc[m,"Available_Tonnes"]=float(row["RM Stock t"]); st.session_state.master_df.loc[m,"Tech_Max"]=float(row["Tech Max t/d"])
            (st.session_state.include_alt if isalt else st.session_state.available)[m]=bool(row["Availability / Include"])
        st.session_state.changed=True

def results():
    page_header("Optimized Sinter Recipe","Detailed optimizer result with reconciled burden and cost contributions.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run the optimizer first."); return
    bd=aligned_result_table(r["blend"],r["df"]); ach=r["achieved"]; cost=float(bd.iloc[-1]["Cost ₹/t"]); total=float(bd.iloc[-1]["kg/t"])
    c=st.columns(4)
    for col,(l,v,s,k) in zip(c,[("TOTAL COST",f"₹{cost:,.2f}/t","Optimized","s"),("BURDEN",f"{total:,.2f} kg/t","Optimized","g"),("Fe",f"{ach['Fe']:.3f}%",f"{opt.FE_LOWER:.1f}–{opt.FE_UPPER:.1f}","a"),("QUALITY","PASS" if quality_ok(ach) else "REVIEW","All mandatory targets","g" if quality_ok(ach) else "r")]): col.markdown(kpi(l,v,s,k),unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">RECIPE</div></div>',unsafe_allow_html=True); st.dataframe(bd,hide_index=True,use_container_width=True,height=max(250,42*len(bd)+45))
    st.markdown('<div class="panel"><div class="panel-title">QUALITY</div></div>',unsafe_allow_html=True); quality_cards(ach)

def manual():
    page_header("Manual Burden Control","Practical what-if sandbox — optimized recipe is the frozen theoretical baseline; manual changes never modify it.")
    r=st.session_state.result
    if not r or not r["blend"]:
        st.info("Run the optimizer first."); return

    df=r["df"]
    if st.session_state.get("manual_base") is None:
        st.session_state.manual_base=dict(r["blend"])
    base=dict(st.session_state.manual_base)

    adjustable=[m for m in base if df.loc[m,"Group"] in ("Iron_ore","Flux")]
    fixed=[m for m in base if df.loc[m,"Group"] in ("Recycle","Fuel")]

    # Separate widget state from practical-scenario state.
    if "manual_values" not in st.session_state:
        st.session_state.manual_values=dict(base)

    baseline_signature=tuple(sorted((m,round(float(v),6)) for m,v in base.items()))
    if st.session_state.get("manual_signature") != baseline_signature:
        st.session_state.manual_values=dict(base)
        st.session_state.manual_signature=baseline_signature
        for m in adjustable:
            st.session_state.pop(f"manual_slider_{m}",None)

    st.markdown('<div class="notice"><b>THEORETICAL BASELINE → PRACTICAL SIMULATION</b><br>The optimized recipe is frozen as the baseline. Increase or decrease adjustable materials within their practical range. Remaining adjustable materials are redistributed proportionally to preserve total burden. The optimizer result is never overwritten.</div>',unsafe_allow_html=True)

    if st.button("↩ RESET TO OPTIMIZED BASELINE",key="reset_manual",use_container_width=True):
        st.session_state.manual_values=dict(base)
        for m in adjustable: st.session_state.pop(f"manual_slider_{m}",None)
        st.rerun()

    req={}
    cols=st.columns(2)
    for i,m in enumerate(adjustable):
        b=float(base[m]); rng=float(opt.ADJUSTMENT_RANGES.get(df.loc[m,"Group"],0.0))
        mn=max(0.0,b*(1-rng)); mx=max(mn+0.5,b*(1+rng))
        default=min(mx,max(mn,float(st.session_state.manual_values.get(m,b))))
        with cols[i%2]:
            req[m]=st.slider(f"{m} | baseline {b:.2f} kg/t",min_value=float(mn),max_value=float(mx),value=float(default),step=0.5,key=f"manual_slider_{m}")
    st.session_state.manual_values.update(req)

    adjusted=dict(opt.redistribute_adjustment(base,df,req))
    for m in fixed: adjusted[m]=base[m]
    st.session_state.manual=dict(adjusted)

    ach=opt.compute_achieved(adjusted,df,1000)
    base_cost=sum(float(base[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in base)
    manual_cost=sum(float(adjusted[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in adjusted)
    base_total=sum(float(v) for v in base.values()); manual_total=sum(float(v) for v in adjusted.values())

    c=st.columns(6)
    cards=[("BASELINE COST",f"₹{base_cost:,.2f}/t","Theoretical optimized","s"),
           ("PRACTICAL COST",f"₹{manual_cost:,.2f}/t",f"{manual_cost-base_cost:+,.2f}/t","a"),
           ("BASELINE BURDEN",f"{base_total:,.2f} kg/t","Theoretical","s"),
           ("PRACTICAL BURDEN",f"{manual_total:,.2f} kg/t",f"{manual_total-base_total:+,.2f} kg/t","g"),
           ("Fe",f"{ach['Fe']:.3f}%",f"Baseline {r['achieved']['Fe']:.3f}%","a"),
           ("QUALITY","PASS" if quality_ok(ach) else "REVIEW","Practical scenario","g" if quality_ok(ach) else "r")]
    for col,(l,v,s,k) in zip(c,cards): col.markdown(kpi(l,v,s,k),unsafe_allow_html=True)

    compare=[]
    for m in df.index:
        ov=float(base.get(m,0)); mv=float(adjusted.get(m,0))
        if ov==0 and mv==0: continue
        compare.append({"Material":m,"Optimized kg/t":ov,"Practical kg/t":mv,"Change kg/t":mv-ov,"Change %":((mv-ov)/ov*100 if ov else np.nan)})
    st.markdown('<div class="panel"><div class="panel-title">THEORETICAL vs PRACTICAL BURDEN</div><div class="small">Theoretical = optimizer baseline. Practical = current user scenario.</div></div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(compare),hide_index=True,use_container_width=True,height=paired_table_height(len(compare)),
                 column_config={"Optimized kg/t":st.column_config.NumberColumn("Optimized kg/t",format="%.2f"),
                                "Practical kg/t":st.column_config.NumberColumn("Practical kg/t",format="%.2f"),
                                "Change kg/t":st.column_config.NumberColumn("Change kg/t",format="%+.2f"),
                                "Change %":st.column_config.NumberColumn("Change %",format="%+.2f%%")})

    st.markdown('<div class="panel"><div class="panel-title">PRACTICAL CHEMISTRY IMPACT</div></div>',unsafe_allow_html=True)
    quality_cards(ach)

    st.markdown(
        '<div class="panel"><div class="panel-title">PRACTICAL MATERIAL CONTRIBUTION</div>'
        '<div class="small">Material-level view of the current practical scenario. '
        'This is independent of the optimizer result.</div></div>',
        unsafe_allow_html=True
    )
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="panel-title">PRACTICAL BURDEN CONTRIBUTION</div>',unsafe_allow_html=True)
        st.plotly_chart(
            material_contribution_chart(adjusted, df, "burden", height=max(360, 115 + 31 * len(adjusted))),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">PRACTICAL COST CONTRIBUTION</div>',unsafe_allow_html=True)
        st.plotly_chart(
            material_contribution_chart(adjusted, df, "cost", height=max(360, 115 + 31 * len(adjusted))),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        st.markdown('</div>',unsafe_allow_html=True)

def alternative():
    page_header("Alternative Raw Material","Contingency ores are loaded from the same Master Excel and remain excluded until explicitly enabled.")
    alts=alt_names()
    if not alts:
        st.info("No alternative rows were detected in the active Master Excel. Add a Type/Material_Type value such as Alternative in the same workbook and re-upload it.")
        return
    st.markdown('<div class="notice">OFF = completely excluded. ON = eligible for the optimizer; ON does not force usage. Chemistry, price, stock and Tech Max are editable from the Dashboard master/input tables.</div>',unsafe_allow_html=True)
    rows=[]
    for m in alts:
        r=st.session_state.master_df.loc[m]
        rows.append({"Material":m,"Fe":r.Fe,"SiO₂":r.SiO2,"Al₂O₃":r.Al2O3,"CaO":r.CaO,"MgO":r.MgO,"LOI":r.LOI,"Price ₹/t":r.Price_Rs_t,"RM Stock t":r.Available_Tonnes,"Tech Max t/d":r.Tech_Max,"Include in Mix":st.session_state.include_alt.get(m,False)})
    ed=st.data_editor(pd.DataFrame(rows),key="alt_editor",hide_index=True,use_container_width=True,height=max(220,42*len(rows)+45),disabled=["Material"])
    if not ed.equals(pd.DataFrame(rows)):
        for _,row in ed.iterrows():
            m=row.Material
            for src,dst in [("Fe","Fe"),("SiO₂","SiO2"),("Al₂O₃","Al2O3"),("CaO","CaO"),("MgO","MgO"),("LOI","LOI"),("Price ₹/t","Price_Rs_t"),("RM Stock t","Available_Tonnes"),("Tech Max t/d","Tech_Max")]: st.session_state.master_df.loc[m,dst]=float(row[src])
            st.session_state.include_alt[m]=bool(row["Include in Mix"])
        st.session_state.changed=True

def composition_page(kind):
    page_header(
        "Burden Mix & Material Contribution" if kind=="burden" else "Cost Structure & Cost Drivers",
        "Material-level contribution from the latest optimized recipe."
    )
    r=st.session_state.result
    if not r or not r["blend"]:
        st.info("Run optimizer first.")
        return

    df=r["df"]
    blend=r["blend"]
    total=sum(float(v) for v in blend.values())
    cost=sum(float(blend[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in blend)

    vals = (
        {GROUP_LABEL[g]:sum(float(blend[m]) for m in blend if df.loc[m,"Group"]==g) for g in GROUPS}
        if kind=="burden"
        else {GROUP_LABEL[g]:sum(float(blend[m])*float(df.loc[m,"Price_Rs_t"])/1000 for m in blend if df.loc[m,"Group"]==g) for g in GROUPS}
    )

    center=total if kind=="burden" else cost
    a,b=st.columns([1.7,1])

    with a:
        st.markdown(
            '<div class="panel"><div class="panel-title">MATERIAL CONTRIBUTION</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            material_contribution_chart(
                blend, df, kind,
                height=max(400, 125 + 32 * len(blend))
            ),
            use_container_width=True,
            config={"displayModeBar":False}
        )
        st.markdown('</div>',unsafe_allow_html=True)

    with b:
        rows=[
            {
                "Group":g,
                "Value":v,
                "% of Burden" if kind=="burden" else "% of Cost":(v/center*100 if center else 0)
            }
            for g,v in vals.items()
        ]
        st.markdown('<div class="panel"><div class="panel-title">GROUP SUMMARY</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=220)
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">MATERIAL BREAKDOWN</div></div>',unsafe_allow_html=True)
    st.dataframe(
        result_table(blend,df),
        hide_index=True,
        use_container_width=True,
        height=max(250,42*(len(blend)+1)+45)
    )

def whatif():
    page_header("Scenario & Material Risk","Test one-at-a-time material unavailability against the v24.0 model.")
    r = st.session_state.result
    if not r or not r.get("blend"):
        st.info("Run the optimizer first.")
        return
    if st.button("▶ RUN MATERIAL SHORTAGE SCENARIOS",type="primary"):
        with st.spinner("Evaluating scenarios…"):
            scenarios = opt.what_if_analysis(active_df(), TARGETS)
            base_cost = float(r.get("cost") or 0)
            if "Cost ₹/t" in scenarios.columns:
                scenarios["Cost Impact ₹/t"] = scenarios["Cost ₹/t"].apply(
                    lambda x: round(float(x) - base_cost, 2) if pd.notna(x) else np.nan
                )
            st.session_state.whatif = scenarios
    if st.session_state.whatif is not None:
        st.dataframe(
            st.session_state.whatif,
            hide_index=True,
            use_container_width=True,
            height=max(250,42*len(st.session_state.whatif)+45)
        )
    else:
        st.info("Run the scenario analysis.")

def bottleneck():
    page_header("Quality Constraint Pressure","Identify the constraints closest to their limits.")
    r=st.session_state.result
    if not r or not r["achieved"]: st.info("Run optimizer first."); return
    st.dataframe(opt.quality_table(r["achieved"],TARGETS),hide_index=True,use_container_width=True,height=420)

def reports():
    page_header("Reports & Export","Export the latest optimized recipe with reconciled burden and cost contributions.")
    r=st.session_state.result
    if not r or not r["blend"]: st.info("Run optimizer first."); return
    bd=aligned_result_table(r["blend"],r["df"]); st.dataframe(bd,hide_index=True,use_container_width=True,height=max(250,42*len(bd)+45)); st.download_button("⬇ DOWNLOAD OPTIMIZATION REPORT",bd.to_csv(index=False).encode(),"sinter_optimization_report.csv","text/csv",use_container_width=True)

def settings():
    page_header("Upload & Settings","Single master workbook management.")
    f=st.file_uploader("UPLOAD MASTER CHEMISTRY EXCEL",type=["xlsx"],key="settings_master")
    if f:
        try:
            df=normalize_master(pd.read_excel(f)); st.success(f"Validated: {len(df)} materials • {int((df["Material_Type"]=='Primary').sum())} primary • {int((df["Material_Type"]=='Alternative').sum())} alternative")
            if st.button("ACTIVATE MASTER",type="primary"):
                st.session_state.master_df=df; st.session_state.source=f.name; st.session_state.available={m:True for m in primary_names()}; st.session_state.include_alt={m:False for m in alt_names()}; st.session_state.result=None; st.session_state.runs=0; st.rerun()
        except Exception as e: st.error(str(e))
    if st.button("↺ RESTORE BUILT-IN MASTER",use_container_width=True):
        st.session_state.master_df=initial_df(); st.session_state.source="Built-in Master Chemistry"; st.session_state.available={m:True for m in primary_names()}; st.session_state.include_alt={}; st.session_state.result=None; st.session_state.runs=0; st.rerun()
    st.markdown('<div class="notice">The same workbook can carry both Primary and Alternative rows. Use a Type column with values such as Primary / Alternative. If no Type column is supplied, HAEMA/BAUXA are recognised as alternatives as a fallback.</div>',unsafe_allow_html=True)

# ----------------------------- ROUTING -----------------------------------
pages={"Dashboard":dashboard,"RM Stock":rm_stock,"Optimization Results":results,"Manual Burden Control":manual,"Alternative Raw Material":alternative,"Burden Composition":lambda:composition_page("burden"),"Cost Composition":lambda:composition_page("cost"),"Cost Analysis":cost_analysis,"What-if Analysis":whatif,"Bottleneck Analysis":bottleneck,"Reports":reports,"Upload & Settings":settings}
pages[st.session_state.nav]()
st.markdown('<div class="footer">Sinter Burden Control • Hospet Alloy Steel Plant • Production decision-support interface</div>',unsafe_allow_html=True)