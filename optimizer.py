#@title 2. Run the complete optimizer

#@title 2. Run the complete optimizer

# ============================================================================
# SINTER BURDEN OPTIMIZER v30.0
# - Coke breeze: heat balance + thermal-state FeO + practical operating bounds
# - FeO is linked to thermal surplus (fuel heat minus burden heat requirement)
# - Manual override: force any coke rate when needed (FeO/heat diagnostics remain visible)
# - Dedicated UI section for all coke‑related inputs
# - All previous features: mandates, alt ores, scenario analysis, etc.
# ============================================================================

import pandas as pd
import pulp
import numpy as np
from io import BytesIO
import re

print("✅ Libraries loaded. Ready.")

# ============================================================================
# 1. MATERIAL RANKING & COMPENSATION RULES
# ============================================================================

IRON_ORE_RANK = {"Lloyds_HG": 1, "MILL_SCALE": 2, "SIOM_MG": 3, "KIOM_MG": 3, "DIOM_LG": 4}
FLUX_RANK = {"LIMESTONE": 1, "DOLOMITE": 2, "QUICKLIME": 3}

MILL_SCALE_MAX_BURDEN_PCT = 0.15

# --- IRON ORE CAPS ---
IRON_ORE_MAX_PCT_BASE = {"Lloyds_HG": 0.25, "MILL_SCALE": 0.29, "SIOM_MG": 0.29, "KIOM_MG": 0.29, "DIOM_LG": 0.29}
IRON_ORE_MAX_PCT_RELAXED = {"Lloyds_HG": 0.35, "MILL_SCALE": 0.35, "SIOM_MG": 0.40, "KIOM_MG": 0.40, "DIOM_LG": 0.40}
IRON_ORE_MAX_PCT_CRISIS = {"Lloyds_HG": 0.95, "MILL_SCALE": 0.95, "SIOM_MG": 0.95, "KIOM_MG": 0.95, "DIOM_LG": 0.95}
IRON_ORE_MIN_PCT = {"Lloyds_HG": 0.03, "MILL_SCALE": 0.03, "SIOM_MG": 0.03, "KIOM_MG": 0.03, "DIOM_LG": 0.03}
MAX_IRON_ORE_PORTION = 0.80
MAX_IRON_ORE_PORTION_CRISIS = 0.95

# --- FLUX CAPS ---
FLUX_MAX_PCT_BASE = {"LIMESTONE": 0.60, "DOLOMITE": 0.45, "QUICKLIME": 0.60}
FLUX_MAX_PCT_RELAXED = {"LIMESTONE": 0.75, "DOLOMITE": 0.55, "QUICKLIME": 0.80}
FLUX_MAX_PCT_CRISIS = {"LIMESTONE": 0.95, "DOLOMITE": 0.95, "QUICKLIME": 0.95}
FLUX_MIN_PCT = {"LIMESTONE": 0.05, "DOLOMITE": 0.05, "QUICKLIME": 0.02}
FLUX_MIN_PCT_QUALITY_RELAXED = {"LIMESTONE": 0.0, "DOLOMITE": 0.0, "QUICKLIME": 0.0}
MAX_FLUX_PORTION = 0.25
MAX_FLUX_PORTION_CRISIS = 0.40

# --- SiO2/CaO CEILING ESCALATION UNDER IRON ORE SHORTAGE ---
SIO2_MAX_SHORTAGE = 6.2

# --- FE TARGET: TIGHT CONTROL (NEVER RELAXED) ---
FE_TARGET = 54.0
FE_TOLERANCE = 0.3
FE_LOWER = FE_TARGET - FE_TOLERANCE
FE_UPPER = FE_TARGET + FE_TOLERANCE
FE_CENTER_WEIGHT = 2.0

# --- QUALITY DEVIATION WEIGHTS (diagnostic only) ---
DEVIATION_WEIGHTS = {
    "Fe": 5.0, "Basicity": 6.0, "CaO": 5.0, "MgO": 4.0,
    "Al2O3": 3.0, "SiO2": 2.0, "Al2O3_SiO2_ratio": 2.0,
}

PIN_TOLERANCE = 1e-3
FLUX_BASELINE_INCREASE_CAP = 0.05

ADJUSTMENT_RANGES = {
    "Iron_ore": 0.15, "Flux": 0.10, "Recycle": 0.00, "Fuel": 0.10,
    "IOL_Fines_Mandate": 0.00, "BF_Returns_Mandate": 0.00,
}

NUM_ALT_ORE_SLOTS = 2

# ============================================================================
# IOL FINES / BF RETURNS MANDATE PARAMETERS (editable via UI)
# ============================================================================
IOL_FINES_NOMINAL_PCT = 0.08
BF_RETURNS_NOMINAL_PCT = 0.17

# STRICT PLANT MANDATES — no fallback / relaxed bands.
# The optimizer must use exactly 80 kg/t IOL Fines and 170 kg/t BF Returns.
# If either mandate cannot be met because of availability/other constraints,
# the model must report infeasibility rather than silently relaxing the mandate.
IOL_FINES_FALLBACK_MIN = IOL_FINES_NOMINAL_PCT
IOL_FINES_FALLBACK_MAX = IOL_FINES_NOMINAL_PCT
BF_RETURNS_FALLBACK_MIN = BF_RETURNS_NOMINAL_PCT
BF_RETURNS_FALLBACK_MAX = BF_RETURNS_NOMINAL_PCT

PIN_BAND = 0.0

# ============================================================================
# DEFAULT COKE OPTIMISATION PARAMETERS (editable in UI)
# ============================================================================
DEFAULT_OM_COST_RS_T = 1500.0
DEFAULT_COKE_CV_KCAL_KG = 6800.0
DEFAULT_COKE_FC_PCT = 71.35
DEFAULT_HEAT_LATENT_MOISTURE = 540.0
DEFAULT_HEAT_CALCINATION_PER_LOI_KG = 420.0
DEFAULT_HEAT_MELTING_PER_KG_SINTER = 60.0
DEFAULT_HEAT_LOSS_FRACTION = 0.12
DEFAULT_FIRING_RATIO_MAX = 1.10
# Interim model setting: the upper firing-ratio cap is diagnostic only until plant heat-balance coefficients are calibrated.
ENFORCE_FIRING_RATIO_MAX = False

# --- PROVISIONAL COKE / FeO OPERATING WINDOW ---
# Based on published sinter-plant practice; ALL VALUES ARE EDITABLE.
# These are deliberately provisional until plant historical data is used for calibration.
DEFAULT_COKE_MIN_KG_T = 55.0
DEFAULT_COKE_MAX_KG_T = 85.0
DEFAULT_FEO_MIN_PCT = 8.5
DEFAULT_FEO_TARGET_PCT = 9.2
DEFAULT_FEO_MAX_PCT = 10.0
DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL = 189180.0
DEFAULT_FEO_REFERENCE_PCT = 8.6
DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL = 0.35
DEFAULT_REFERENCE_COKE_CV_KCAL_KG = 6800.0
DEFAULT_REFERENCE_COKE_FC_PCT = 71.35

def sanitize_material_name(raw_name):
    name = raw_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    return name

# ============================================================================
# 2. DEFAULT CHEMISTRY + BOUNDS (COKE Tech_Min=0, Tech_Max=999)
# ============================================================================
def get_default_chemistry():
    data = {
        "Material": [
            "MILL_SCALE", "Lloyds_HG", "DIOM_LG", "SIOM_MG", "KIOM_MG",
            "Solid_Waste", "IOL_Fines", "FLUE_DUST", "BF_Returns",
            "DOLOMITE", "LIMESTONE", "QUICKLIME",
            "COKE_BREEZE"
        ],
        "Group": [
            "Iron_ore", "Iron_ore", "Iron_ore", "Iron_ore", "Iron_ore",
            "Recycle", "IOL_Fines_Mandate", "Recycle", "BF_Returns_Mandate",
            "Flux", "Flux", "Flux",
            "Fuel"
        ],
        "Fe":    [68.34, 63.52, 57.17, 59.34, 58.41, 50.0, 60.00, 47.02, 52.5,   0.54, 0.88, 0.01, 0],
        "SiO2":  [2.00,  3.86,  12.39, 6.92,  5.75,  6.00, 5.00,  7.07,  5.62,   4.72, 4.48, 2.50, 2.8],
        "Al2O3": [2.72,  2.27,  2.93,  3.72,  5.48,  4.50, 3.00,  4.50,  3.20,   0.95, 1.19, 0.61, 0],
        "CaO":   [0,     0.022, 0.058, 0.256, 0.157, 1.122,8.79,  1.10,  10.74,  30.02,48.71,89.00,0],
        "MgO":   [0,     0.034, 0.114, 0.331, 0.018, 0.06, 1.52,  0.29,  2.30,   18.75,2.59, 1.57, 0],
        "LOI":   [2.50,  2.29,  4.00,  3.45,  4.62,  3.00, 3.00,  15.00, 3.00,   42.00,40.00,5.00, 70.00],
        "Moisture_Pct": [
            6.0, 5.0, 6.0, 6.0, 6.0, 1.1, 4.13, 9.4, 0.0, 2.0, 2.0, 0.0, 11.27
        ],
        "Tech_Min": [0, 0, 0, 0, 0, 30, 0, 25, 0, 30, 0, 40, 0],
        "Tech_Max": [220, 200, 200, 200, 300, 30, 999, 25, 999, 200, 250, 65, 999],
        "Available_Tonnes": [2000, 10000, 6000, 8000, 5000, 5000, 5000, 3000, 5000, 10000, 15000, 5000, 9999],
        "Price_Rs_t": [7800, 7820, 4600, 4600, 4900, 1000, 5577, 500, 0, 1340, 1355, 9200, 15022],
    }
    df = pd.DataFrame(data).set_index("Material")
    for mat in df[df["Group"] == "Recycle"].index:
        fixed_rate = df.loc[mat, "Tech_Min"]
        df.loc[mat, "Tech_Min"] = fixed_rate
        df.loc[mat, "Tech_Max"] = fixed_rate
    return df

# ============================================================================
# 3. LOAD EXCEL
# ============================================================================
def load_chemistry_from_excel(uploaded_file):
    df = pd.read_excel(BytesIO(uploaded_file[list(uploaded_file.keys())[0]]), index_col="Material")
    required_cols = ["Group", "Fe", "SiO2", "Al2O3", "CaO", "MgO", "LOI", "Tech_Min", "Tech_Max", "Available_Tonnes", "Price_Rs_t"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel missing required column: {col}")
    if "Moisture_Pct" not in df.columns:
        print("⚠️ Excel missing Moisture_Pct column - defaulting all materials to 0% moisture.")
        df["Moisture_Pct"] = 0.0
    rename_map = {"INTERNAL_FINES": "Solid_Waste", "RETURN_SINTER": "IOL_Fines"}
    df = df.rename(index=rename_map)
    for mat in df[df["Group"] == "Recycle"].index:
        fixed_rate = df.loc[mat, "Tech_Min"]
        df.loc[mat, "Tech_Min"] = fixed_rate
        df.loc[mat, "Tech_Max"] = fixed_rate
    if "LIME_POWDER" in df.index:
        df = df.drop("LIME_POWDER")
    print("📂 Master Chemistry loaded successfully!")
    return df

# ============================================================================
# 4. HELPERS (unchanged)
# ============================================================================
def get_mandate_shortfall_triggers(df):
    triggers = 0
    reasons = []
    for mat in ["IOL_Fines", "BF_Returns"]:
        if mat in df.index:
            if df.loc[mat, "Available_Tonnes"] <= 0 or df.loc[mat, "Tech_Max"] == 0:
                triggers += 1
                reasons.append(mat)
    return triggers, reasons

def get_iron_ore_tier(df, iron_ores, extra_missing=0, extra_reasons=None):
    extra_reasons = extra_reasons or []
    unavailable = [m for m in iron_ores if df.loc[m, "Available_Tonnes"] <= 0 or df.loc[m, "Tech_Max"] == 0]
    n = len(unavailable) + extra_missing
    display_list = unavailable + [f"{r}(mandate-shortfall)" for r in extra_reasons]
    if n == 0:
        return IRON_ORE_MAX_PCT_BASE.copy(), unavailable, "✅ All iron ores available - using base caps (25%/29%)", "base"
    elif n == 1:
        return IRON_ORE_MAX_PCT_RELAXED.copy(), unavailable, f"⚠️ {n} iron-ore-equivalent shortfall: {', '.join(display_list)} — RELAXED caps (35-40%)", "relaxed"
    else:
        return IRON_ORE_MAX_PCT_CRISIS.copy(), unavailable, (
            f"🔥 {n} iron-ore-equivalent shortfalls: {', '.join(display_list)} — CRISIS MODE: usage caps near-removed."
        ), "crisis"

def get_flux_tier(df, fluxes, extra_missing=0, extra_reasons=None):
    extra_reasons = extra_reasons or []
    unavailable = [m for m in fluxes if df.loc[m, "Available_Tonnes"] <= 0 or df.loc[m, "Tech_Max"] == 0]
    n = len(unavailable) + extra_missing
    display_list = unavailable + [f"{r}(mandate-shortfall)" for r in extra_reasons]
    if n == 0:
        return FLUX_MAX_PCT_BASE.copy(), unavailable, "✅ All fluxes available - using base flux caps", "base"
    elif n == 1:
        return FLUX_MAX_PCT_RELAXED.copy(), unavailable, f"⚠️ {n} flux-equivalent shortfall: {', '.join(display_list)} — RELAXED flux caps", "relaxed"
    else:
        return FLUX_MAX_PCT_CRISIS.copy(), unavailable, (
            f"🔥 {n} flux-equivalent shortfalls: {', '.join(display_list)} — CRISIS MODE: flux caps near-removed."
        ), "crisis"

def check_fuel_gate(df):
    fuels = [m for m in df.index if df.loc[m, "Group"] == "Fuel"]
    problems = []
    for mat in fuels:
        tech_min = df.loc[mat, "Tech_Min"]
        available = df.loc[mat, "Available_Tonnes"]
        tech_max = df.loc[mat, "Tech_Max"]
        if tech_min > 0 and (available <= 0 or tech_max <= 0):
            problems.append(f"{mat} (requires >= {tech_min} kg/t, but Available_Tonnes={available}, Tech_Max={tech_max})")
    if problems:
        return False, problems
    return True, []

def build_bounds(df, production_tonnes):
    bounds = {}
    for mat in df.index:
        tech_min = df.loc[mat, "Tech_Min"]
        tech_max = df.loc[mat, "Tech_Max"]
        available = df.loc[mat, "Available_Tonnes"]
        if available <= 0 or tech_max == 0:
            bounds[mat] = (0, 0)
        else:
            inv_cap = (available / production_tonnes) * 1000
            eff_max = min(tech_max, inv_cap)
            bounds[mat] = (tech_min, eff_max)
    return bounds

# ============================================================================
# 5. MANDATE CONSTRAINTS
# ============================================================================
def add_mandate_constraints(prob, x, df, OUT, mandate_mode="pinned",
                            iol_nominal=IOL_FINES_NOMINAL_PCT,
                            bf_nominal=BF_RETURNS_NOMINAL_PCT,
                            iol_fb_min=IOL_FINES_FALLBACK_MIN,
                            iol_fb_max=IOL_FINES_FALLBACK_MAX,
                            bf_fb_min=BF_RETURNS_FALLBACK_MIN,
                            bf_fb_max=BF_RETURNS_FALLBACK_MAX):
    for mat, nominal, fb_min, fb_max in [
        ("IOL_Fines", iol_nominal, iol_fb_min, iol_fb_max),
        ("BF_Returns", bf_nominal, bf_fb_min, bf_fb_max),
    ]:
        if mat not in x:
            continue
        # Strict mandate: never relax and never silently set an unavailable
        # mandated material to zero. If bounds/availability cannot support the
        # exact mandate, the LP must become infeasible and report the issue.
        prob += x[mat] == nominal * OUT, f"{mat}_STRICT_MANDATE"

# ============================================================================
# 6. STRUCTURAL CONSTRAINTS (unchanged)
# ============================================================================
def add_structural_constraints(prob, x, df, bounds, iron_ores, fluxes, iron_ore_max_pct,
                                unavailable_iron, flux_max_pct, unavailable_flux, OUT,
                                baseline_flux_portion=None, iron_tier="base", flux_tier="base",
                                flux_min_pct_override=None, mandate_mode="pinned",
                                iol_nominal=IOL_FINES_NOMINAL_PCT,
                                bf_nominal=BF_RETURNS_NOMINAL_PCT,
                                iol_fb_min=IOL_FINES_FALLBACK_MIN,
                                iol_fb_max=IOL_FINES_FALLBACK_MAX,
                                bf_fb_min=BF_RETURNS_FALLBACK_MIN,
                                bf_fb_max=BF_RETURNS_FALLBACK_MAX):
    non_fuel = [m for m in x if df.loc[m, "Group"] != "Fuel"]
    mass = pulp.lpSum(x[m] * (1 - df.loc[m, "LOI"] / 100) for m in non_fuel)
    prob += mass >= OUT - 2, "Mass_Balance_Lower"
    prob += mass <= OUT + 2, "Mass_Balance_Upper"

    total_iron_ore = pulp.lpSum(x[m] for m in iron_ores)
    total_flux = pulp.lpSum(x[m] for m in fluxes)
    total_burden = pulp.lpSum(x[m] for m in df.index if df.loc[m, "Group"] != "Fuel")

    for mat in iron_ores:
        if mat in unavailable_iron:
            prob += x[mat] == 0, f"{mat}_unavailable"
        else:
            max_pct = iron_ore_max_pct.get(mat, 0.29)
            prob += x[mat] <= max_pct * total_iron_ore + 0.001, f"{mat}_max_pct"
            prob += x[mat] >= IRON_ORE_MIN_PCT.get(mat, 0.03) * total_iron_ore - 0.001, f"{mat}_min_pct"

    if "MILL_SCALE" in x:
        prob += x["MILL_SCALE"] <= MILL_SCALE_MAX_BURDEN_PCT * total_burden, "MILL_SCALE_Burden_Cap"

    iron_ore_portion_cap = MAX_IRON_ORE_PORTION_CRISIS if iron_tier == "crisis" else MAX_IRON_ORE_PORTION
    prob += total_iron_ore <= iron_ore_portion_cap * OUT, "Max_Iron_Ore_Portion"

    min_pct_source = flux_min_pct_override if flux_min_pct_override is not None else FLUX_MIN_PCT
    default_min_pct = 0.0 if flux_min_pct_override is not None else 0.02
    for mat in fluxes:
        if mat in unavailable_flux:
            prob += x[mat] == 0, f"{mat}_unavailable"
        else:
            max_pct = flux_max_pct.get(mat, 0.5)
            min_pct = min_pct_source.get(mat, default_min_pct)
            prob += x[mat] <= max_pct * total_flux + 0.001, f"{mat}_max_pct"
            prob += x[mat] >= min_pct * total_flux - 0.001, f"{mat}_min_pct"

    flux_portion_cap = MAX_FLUX_PORTION_CRISIS if flux_tier == "crisis" else MAX_FLUX_PORTION
    prob += total_flux <= flux_portion_cap * OUT, "Max_Flux_Portion"

    mandate_active = False
    for m in ["IOL_Fines", "BF_Returns"]:
        if m in df.index and df.loc[m, "Available_Tonnes"] > 0 and df.loc[m, "Tech_Max"] > 0:
            mandate_active = True
            break

    if baseline_flux_portion is not None and flux_tier not in ("crisis", "quality_relaxed") and not mandate_active:
        prob += total_flux <= (baseline_flux_portion + FLUX_BASELINE_INCREASE_CAP) * OUT, "Flux_Baseline_Cap"

    add_mandate_constraints(prob, x, df, OUT, mandate_mode=mandate_mode,
                            iol_nominal=iol_nominal, bf_nominal=bf_nominal,
                            iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                            bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max)

    return total_iron_ore, total_flux, total_burden

# ============================================================================
# 7. QUALITY & DIAGNOSTIC FUNCTIONS
# ============================================================================
def compute_achieved(blend, df, OUT):
    Fe = sum(blend[m] * df.loc[m, "Fe"] / 100 for m in blend) / OUT * 100
    SiO2 = sum(blend[m] * df.loc[m, "SiO2"] / 100 for m in blend) / OUT * 100
    Al2O3 = sum(blend[m] * df.loc[m, "Al2O3"] / 100 for m in blend) / OUT * 100
    CaO = sum(blend[m] * df.loc[m, "CaO"] / 100 for m in blend) / OUT * 100
    MgO = sum(blend[m] * df.loc[m, "MgO"] / 100 for m in blend) / OUT * 100
    achieved = {"Fe": Fe, "SiO2": SiO2, "Al2O3": Al2O3, "CaO": CaO, "MgO": MgO}
    if SiO2 > 0:
        achieved["Basicity"] = CaO / SiO2
        achieved["Al2O3/SiO2"] = Al2O3 / SiO2
        achieved["B4"] = (CaO + MgO) / (SiO2 + Al2O3)
    else:
        achieved["Basicity"] = 0
        achieved["Al2O3/SiO2"] = 0
        achieved["B4"] = 0
    return achieved

def build_soft_vars_and_constraints(prob, xr, df, OUT, targets, fe_lo, fe_hi, suffix=""):
    Fe_s = pulp.lpSum(xr[m] * df.loc[m, "Fe"] / 100 for m in xr)
    SiO2_s = pulp.lpSum(xr[m] * df.loc[m, "SiO2"] / 100 for m in xr)
    Al2O3_s = pulp.lpSum(xr[m] * df.loc[m, "Al2O3"] / 100 for m in xr)
    CaO_s = pulp.lpSum(xr[m] * df.loc[m, "CaO"] / 100 for m in xr)
    MgO_s = pulp.lpSum(xr[m] * df.loc[m, "MgO"] / 100 for m in xr)

    Fe_under = pulp.LpVariable(f"Fe_under{suffix}", lowBound=0)
    Fe_over = pulp.LpVariable(f"Fe_over{suffix}", lowBound=0)
    SiO2_over = pulp.LpVariable(f"SiO2_over{suffix}", lowBound=0)
    Al2O3_over = pulp.LpVariable(f"Al2O3_over{suffix}", lowBound=0)
    ratio_over = pulp.LpVariable(f"ratio_over{suffix}", lowBound=0)
    Bas_under = pulp.LpVariable(f"Bas_under{suffix}", lowBound=0)
    Bas_over = pulp.LpVariable(f"Bas_over{suffix}", lowBound=0)
    MgO_under = pulp.LpVariable(f"MgO_under{suffix}", lowBound=0)
    MgO_over = pulp.LpVariable(f"MgO_over{suffix}", lowBound=0)
    CaO_under = pulp.LpVariable(f"CaO_under{suffix}", lowBound=0)
    CaO_over = pulp.LpVariable(f"CaO_over{suffix}", lowBound=0)
    Fe_center_dev = pulp.LpVariable(f"Fe_center_dev{suffix}", lowBound=0)

    prob += Fe_s + Fe_under >= fe_lo, f"Fe_lo_soft{suffix}"
    prob += Fe_s - Fe_over <= fe_hi, f"Fe_hi_soft{suffix}"
    prob += SiO2_s - SiO2_over <= targets["SiO2_max"] * OUT / 100, f"SiO2_soft{suffix}"
    prob += Al2O3_s - Al2O3_over <= targets["Al2O3_max"] * OUT / 100, f"Al2O3_soft{suffix}"
    prob += (Al2O3_s - targets["Al2O3_SiO2_max"] * SiO2_s) - ratio_over <= 0, f"Ratio_soft{suffix}"
    prob += (CaO_s - targets["Basicity_min"] * SiO2_s) + Bas_under >= 0, f"Basicity_lo_soft{suffix}"
    prob += (CaO_s - targets["Basicity_max"] * SiO2_s) - Bas_over <= 0, f"Basicity_hi_soft{suffix}"
    prob += MgO_s + MgO_under >= targets["MgO_min"] * OUT / 100, f"MgO_lo_soft{suffix}"
    prob += MgO_s - MgO_over <= targets["MgO_max"] * OUT / 100, f"MgO_hi_soft{suffix}"
    prob += CaO_s + CaO_under >= targets["CaO_min"] * OUT / 100, f"CaO_lo_soft{suffix}"
    prob += CaO_s - CaO_over <= targets["CaO_max"] * OUT / 100, f"CaO_hi_soft{suffix}"

    prob += Fe_s - (FE_TARGET * OUT / 100) <= Fe_center_dev, f"Fe_center_pos{suffix}"
    prob += (FE_TARGET * OUT / 100) - Fe_s <= Fe_center_dev, f"Fe_center_neg{suffix}"

    slacks = {
        "Fe_under": Fe_under, "Fe_over": Fe_over, "SiO2_over": SiO2_over, "Al2O3_over": Al2O3_over,
        "ratio_over": ratio_over, "Bas_under": Bas_under, "Bas_over": Bas_over,
        "MgO_under": MgO_under, "MgO_over": MgO_over, "CaO_under": CaO_under, "CaO_over": CaO_over,
        "Fe_center_dev": Fe_center_dev,
    }
    sums = {"Fe": Fe_s, "SiO2": SiO2_s, "Al2O3": Al2O3_s, "CaO": CaO_s, "MgO": MgO_s}
    return slacks, sums

def weighted_deviation_expr(slacks, targets, OUT):
    W = DEVIATION_WEIGHTS
    expr = (
        W["Fe"] * ((slacks["Fe_under"] + slacks["Fe_over"]) / (FE_TOLERANCE * OUT / 100)) +
        W["SiO2"] * (slacks["SiO2_over"] / (targets["SiO2_max"] * OUT / 100)) +
        W["Al2O3"] * (slacks["Al2O3_over"] / (targets["Al2O3_max"] * OUT / 100)) +
        W["Al2O3_SiO2_ratio"] * (slacks["ratio_over"] / max(targets["Al2O3_SiO2_max"] * OUT / 100, 1e-6)) +
        W["Basicity"] * ((slacks["Bas_under"] + slacks["Bas_over"]) /
                          max((targets["Basicity_max"] - targets["Basicity_min"]) * OUT / 100, 1e-6)) +
        W["MgO"] * ((slacks["MgO_under"] + slacks["MgO_over"]) /
                    max((targets["MgO_max"] - targets["MgO_min"]) * OUT / 100, 1e-6)) +
        W["CaO"] * ((slacks["CaO_under"] + slacks["CaO_over"]) /
                    max((targets["CaO_max"] - targets["CaO_min"]) * OUT / 100, 1e-6)) +
        FE_CENTER_WEIGHT * (slacks["Fe_center_dev"] / (FE_TOLERANCE * OUT / 100))
    )
    return expr

def _report_compensation(blend, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                          iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons=None):
    diagnostics = []
    if unavailable_iron or (mandate_reasons and iron_tier != "base"):
        iron_ore_total = sum(blend[m] for m in iron_ores)
        diag_msg = f"\n   ✅ Iron Ore Compensation Result (Tier: {iron_tier}):"
        diag_msg += f"\n   Iron Ore Portion: {iron_ore_total:.1f} kg ({iron_ore_total/OUT*100:.1f}% of burden)"
        for mat in iron_ores:
            if mat in unavailable_iron:
                diag_msg += f"\n      {mat}: UNAVAILABLE"
            else:
                pct = blend[mat] / iron_ore_total * 100 if iron_ore_total > 0 else 0
                max_pct = iron_ore_max_pct.get(mat, 0.29) * 100
                diag_msg += f"\n      {mat}: {blend[mat]:.1f} kg ({pct:.1f}%) [Max {max_pct:.0f}%]"
        diagnostics.append(diag_msg)

    if unavailable_flux or (mandate_reasons and flux_tier != "base"):
        flux_total = sum(blend[m] for m in fluxes)
        diag_msg = f"\n   ✅ Flux Compensation Result (Tier: {flux_tier}):"
        for mat in fluxes:
            if mat in unavailable_flux:
                diag_msg += f"\n      {mat}: UNAVAILABLE"
            else:
                pct = blend[mat] / flux_total * 100 if flux_total > 0 else 0
                max_pct = flux_max_pct.get(mat, 0.5) * 100
                diag_msg += f"\n      {mat}: {blend[mat]:.1f} kg ({pct:.1f}%) [Max {max_pct:.0f}%]"
        diagnostics.append(diag_msg)

    if mandate_reasons:
        diagnostics.append(
            f"\n   🔗 NOTE: {', '.join(mandate_reasons)} shortfall triggered tier escalation in BOTH groups."
        )

    if iron_tier == "crisis" or flux_tier == "crisis":
        diagnostics.append(
            "\n   💰 NOTE: CRISIS MODE – usage caps loosened to hit quality; cost may be higher."
        )
    return diagnostics

def _report_fines_loading(blend, df, OUT):
    fine_materials = [m for m in ["IOL_Fines", "BF_Returns", "MILL_SCALE"] if m in blend]
    total_fines = sum(blend.get(m, 0) for m in fine_materials)
    pct = total_fines / OUT * 100
    msg = f"\n   🧱 COMBINED FINES LOADING (IOL_Fines + BF_Returns + Mill Scale): {total_fines:.1f} kg ({pct:.1f}% of burden)"
    if pct > 30:
        msg += "\n   ⚠️ HIGH FINES LOADING (>30%) – verify permeability."
    return msg

# ============================================================================
# 8. WET/DRY COSTING TABLES (with TOTAL rows)
# ============================================================================
def compute_dry_cost_table(blend, df, om_cost):
    total_raw_input = sum(blend.values())
    rm_cost = sum(blend[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blend)
    table = pd.DataFrame({
        "Group": [df.loc[m, "Group"] for m in blend],
        "Dry kg / t sinter": [blend[m] for m in blend],
        "% of Burden": [(blend[m] / total_raw_input) * 100 if total_raw_input else 0 for m in blend],
        "Dry Cost (Rs/t)": [round(blend[m] * df.loc[m, "Price_Rs_t"] / 1000, 2) for m in blend],
    }, index=blend.keys())
    total_row = pd.DataFrame({
        "Group": ["TOTAL"],
        "Dry kg / t sinter": [total_raw_input],
        "% of Burden": [100.0],
        "Dry Cost (Rs/t)": [round(rm_cost, 2)],
    }, index=["TOTAL"])
    table = pd.concat([table, total_row])
    total_sinter_cost = rm_cost + om_cost
    return table, rm_cost, total_sinter_cost

def compute_wet_cost_table(blend, df, om_cost):
    if "Moisture_Pct" not in df.columns:
        print("⚠️ WARNING: 'Moisture_Pct' column not found – defaulting all moisture to 0% for wet costing.")
        df = df.copy()
        df["Moisture_Pct"] = 0.0

    rows = []
    wet_rm_cost = 0
    total_dry_kg = 0
    total_wet_kg = 0
    for m in blend:
        moisture = df.loc[m, "Moisture_Pct"] / 100 if df.loc[m, "Moisture_Pct"] < 100 else 0.0
        dry_kg = blend[m]
        wet_kg = dry_kg / (1 - moisture) if moisture < 1 else dry_kg
        wet_cost = wet_kg * df.loc[m, "Price_Rs_t"] / 1000
        wet_rm_cost += wet_cost
        total_dry_kg += dry_kg
        total_wet_kg += wet_kg
        rows.append({
            "Material": m, "Group": df.loc[m, "Group"],
            "Dry kg / t sinter": dry_kg,
            "Moisture %": round(moisture * 100, 2),
            "Wet (As-Received) kg": round(wet_kg, 2),
            "Wet Cost (Rs/t)": round(wet_cost, 2),
        })

    table = pd.DataFrame(rows).set_index("Material")
    total_row = pd.DataFrame({
        "Group": ["TOTAL"],
        "Dry kg / t sinter": [total_dry_kg],
        "Moisture %": [0.0],
        "Wet (As-Received) kg": [total_wet_kg],
        "Wet Cost (Rs/t)": [round(wet_rm_cost, 2)],
    }, index=["TOTAL"])
    table = pd.concat([table, total_row])
    table["Moisture %"] = table["Moisture %"].astype(float)
    total_sinter_cost_wet = wet_rm_cost + om_cost
    return table, wet_rm_cost, total_sinter_cost_wet

# ============================================================================
# 9. COKE HEAT-BALANCE DIAGNOSTIC (now also used as a constraint in the LP)
# ============================================================================
def compute_coke_heat_balance_diagnostic(blend, df, OUT,
                                         coke_cv=DEFAULT_COKE_CV_KCAL_KG,
                                         coke_fc=DEFAULT_COKE_FC_PCT,
                                         latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                         calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                         melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                         loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                         feo_min=DEFAULT_FEO_MIN_PCT,
                                         feo_target=DEFAULT_FEO_TARGET_PCT,
                                         feo_max=DEFAULT_FEO_MAX_PCT,
                                         feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                         feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                         feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                         ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
                                         ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT):
    """Coke heat-balance + thermal-state FeO diagnostic.

    v30:
        Q_fuel = coke rate × fixed carbon × CV
        Q_required = moisture + calcination + base thermal demand + losses
        Thermal surplus = Q_fuel - Q_required
        FeO = FeO_ref + slope × (thermal surplus - reference surplus) / 10,000

    This keeps the FeO relationship linear/LP-compatible and allows changes
    in raw-material moisture/LOI to change the optimized coke requirement.
    The coefficients are provisional until plant Coke/FeO history is available.
    """
    if "COKE_BREEZE" not in blend:
        return None

    cb_kg = blend["COKE_BREEZE"]
    Q_fuel = cb_kg * (coke_fc / 100) * coke_cv

    total_wet_mass = 0.0
    for m in blend:
        if m == "COKE_BREEZE":
            continue
        moisture = df.loc[m, "Moisture_Pct"] / 100 if "Moisture_Pct" in df.columns else 0.0
        wet_kg = blend[m] / (1 - moisture) if moisture < 1 else blend[m]
        total_wet_mass += wet_kg

    dry_nonfuel = sum(blend[m] for m in blend if m != "COKE_BREEZE")
    moisture_mass = max(total_wet_mass - dry_nonfuel, 0.0)
    Q_moisture = moisture_mass * latent_heat

    total_loi_mass = sum(
        blend[m] * df.loc[m, "LOI"] / 100
        for m in blend if m != "COKE_BREEZE"
    )
    Q_calcination = total_loi_mass * calcination_heat
    Q_melting = OUT * melting_heat

    Q_required_before_loss = Q_moisture + Q_calcination + Q_melting
    Q_required = (
        Q_required_before_loss / (1 - loss_fraction)
        if (1 - loss_fraction) > 0 else 0.0
    )

    firing_ratio = Q_fuel / Q_required if Q_required > 0 else 0.0
    thermal_surplus = Q_fuel - Q_required

    # Effective coke normalized to the reference coke quality.
    # This is a diagnostic quantity only; the LP still uses the actual coke rate.
    effective_coke_kg_t = cb_kg * (coke_fc / ref_coke_fc) * (coke_cv / ref_coke_cv)

    feo_est = feo_ref_pct + feo_thermal_slope * (
        (thermal_surplus - feo_ref_surplus) / 10000.0
    )

    if feo_est < feo_min:
        suggestion = (
            f"FeO {feo_est:.2f}% below minimum {feo_min:.2f}% "
            "→ increase coke / review thermal conditions"
        )
    elif feo_est > feo_max:
        suggestion = (
            f"FeO {feo_est:.2f}% above maximum {feo_max:.2f}% "
            "→ reduce coke / review thermal conditions"
        )
    elif abs(feo_est - feo_target) <= 0.15:
        suggestion = (
            f"FeO {feo_est:.2f}% is close to target {feo_target:.2f}% "
            "– no adjustment suggested"
        )
    elif feo_est < feo_target:
        suggestion = (
            f"FeO {feo_est:.2f}% is below target {feo_target:.2f}% "
            "but within operating band"
        )
    else:
        suggestion = (
            f"FeO {feo_est:.2f}% is above target {feo_target:.2f}% "
            "but within operating band"
        )

    return {
        "CB_kg_LP_chosen": cb_kg,
        "Q_fuel_kcal": Q_fuel,
        "Q_required_kcal": Q_required,
        "Thermal_Surplus_kcal": thermal_surplus,
        "Firing_Ratio": firing_ratio,
        "FeO_Estimate_Pct": feo_est,
        "FeO_Min_Pct": feo_min,
        "FeO_Target_Pct": feo_target,
        "FeO_Max_Pct": feo_max,
        "Reference_Thermal_Surplus_kcal": feo_ref_surplus,
        "Thermal_Slope_Pct_per_10k_kcal": feo_thermal_slope,
        "Effective_Coke_kg_t": effective_coke_kg_t,
        "Controller_Suggestion": suggestion,
        "note": (
            "⚠️ PROVISIONAL FeO thermal-state model – calibrate "
            "coefficients against plant Coke/FeO history."
        )
    }

# ============================================================================
# 10. MAIN SOLVER (accepts all parameters + manual override)
# ============================================================================
def solve_blend_with_compensation(df, production_tonnes, targets, baseline_blend=None,
                                  enforce_b4=False, b4_min=1.8, b4_max=2.0,
                                  iol_nominal=IOL_FINES_NOMINAL_PCT,
                                  bf_nominal=BF_RETURNS_NOMINAL_PCT,
                                  iol_fb_min=IOL_FINES_FALLBACK_MIN,
                                  iol_fb_max=IOL_FINES_FALLBACK_MAX,
                                  bf_fb_min=BF_RETURNS_FALLBACK_MIN,
                                  bf_fb_max=BF_RETURNS_FALLBACK_MAX,
                                  coke_cv=DEFAULT_COKE_CV_KCAL_KG,
                                  coke_fc=DEFAULT_COKE_FC_PCT,
                                  latent_heat=DEFAULT_HEAT_LATENT_MOISTURE,
                                  calcination_heat=DEFAULT_HEAT_CALCINATION_PER_LOI_KG,
                                  melting_heat=DEFAULT_HEAT_MELTING_PER_KG_SINTER,
                                  loss_fraction=DEFAULT_HEAT_LOSS_FRACTION,
                                  firing_ratio_max=DEFAULT_FIRING_RATIO_MAX,
                                  enforce_firing_ratio_max=ENFORCE_FIRING_RATIO_MAX,
                                  coke_min_rate=DEFAULT_COKE_MIN_KG_T,
                                  coke_max_rate=DEFAULT_COKE_MAX_KG_T,
                                  feo_min=DEFAULT_FEO_MIN_PCT,
                                  feo_target=DEFAULT_FEO_TARGET_PCT,
                                  feo_max=DEFAULT_FEO_MAX_PCT,
                                  feo_ref_surplus=DEFAULT_FEO_REFERENCE_THERMAL_SURPLUS_KCAL,
                                  feo_ref_pct=DEFAULT_FEO_REFERENCE_PCT,
                                  feo_thermal_slope=DEFAULT_FEO_THERMAL_SLOPE_PCT_PER_10K_KCAL,
                                  ref_coke_cv=DEFAULT_REFERENCE_COKE_CV_KCAL_KG,
                                  ref_coke_fc=DEFAULT_REFERENCE_COKE_FC_PCT,
                                  manual_override=False,
                                  manual_coke_rate=65.0):
    OUT = 1000
    iron_ores = [m for m in df.index if df.loc[m, "Group"] == "Iron_ore"]
    fluxes = [m for m in df.index if df.loc[m, "Group"] == "Flux"]

    fuel_ok, fuel_problems = check_fuel_gate(df)
    if not fuel_ok:
        diagnostics = ["🚫 PRODUCTION IMPOSSIBLE: Fuel requirement cannot be met."]
        for p in fuel_problems:
            diagnostics.append(f"   {p}")
        return "No_Production", None, None, None, diagnostics, False

    # Strict mandates are independent of ore/flux compensation tiers.
    # A shortage of IOL Fines or BF Returns must make the optimization
    # infeasible; it must NOT relax ore/flux caps or the mandate itself.
    mandate_reasons = []
    mandate_triggers = 0
    for mat, nominal in [("IOL_Fines", iol_nominal), ("BF_Returns", bf_nominal)]:
        required_kg = nominal * OUT
        if mat not in df.index:
            mandate_reasons.append(f"{mat} missing from chemistry master")
        elif df.loc[mat, "Available_Tonnes"] <= 0:
            mandate_reasons.append(f"{mat} unavailable (requires {required_kg:.0f} kg/t)")
        elif df.loc[mat, "Tech_Max"] < required_kg:
            mandate_reasons.append(f"{mat} Tech_Max below mandate ({df.loc[mat, 'Tech_Max']:.1f} < {required_kg:.0f} kg/t)")

    iron_ore_max_pct, unavailable_iron, iron_msg, iron_tier = get_iron_ore_tier(df, iron_ores)
    print(iron_msg)
    flux_max_pct, unavailable_flux, flux_msg, flux_tier = get_flux_tier(df, fluxes)
    print(flux_msg)

    if mandate_reasons:
        diagnostics = [
            "🚫 STRICT MANDATE FAILURE — optimization cannot relax the mandate.",
            *[f"   {r}" for r in mandate_reasons],
            "   Required: IOL Fines = 80 kg/t (8.0%), BF Returns = 170 kg/t (17.0%)."
        ]
        return "Infeasible", None, None, None, diagnostics, True

    bounds = build_bounds(df, production_tonnes)
    diagnostics = []
    if unavailable_iron:
        diagnostics.append(f"🔄 Compensating for missing ore(s): {', '.join(unavailable_iron)} (Iron tier: {iron_tier})")
    if unavailable_flux:
        diagnostics.append(f"🔄 Compensating for missing flux(es): {', '.join(unavailable_flux)} (Flux tier: {flux_tier})")
    if mandate_reasons:
        diagnostics.append(f"🔗 Mandate shortfall ({', '.join(mandate_reasons)}) escalated BOTH tiers.")

    fe_lo = FE_LOWER * OUT / 100
    fe_hi = FE_UPPER * OUT / 100

    baseline_flux_portion = None
    if baseline_blend:
        baseline_flux_portion = sum(baseline_blend.get(m, 0) for m in fluxes) / OUT

    shortage_targets = None
    if iron_tier != "base":
        shortage_targets = dict(targets)
        shortage_targets["SiO2_max"] = SIO2_MAX_SHORTAGE
        shortage_targets["CaO_max"] = round(targets["Basicity_max"] * SIO2_MAX_SHORTAGE, 3)

    # Precompute moisture and LOI factors for heat balance
    moisture_factor = {}
    loi_factor = {}
    for m in df.index:
        if m == "COKE_BREEZE":
            continue
        mois = df.loc[m, "Moisture_Pct"] / 100 if "Moisture_Pct" in df.columns else 0.0
        if mois < 1:
            moisture_factor[m] = mois / (1 - mois)
        else:
            moisture_factor[m] = 0.0
        loi_factor[m] = df.loc[m, "LOI"] / 100

    def _build_and_solve(flux_min_pct_override, flux_tier_label, tag,
                         use_targets=None, mandate_mode="pinned"):
        t = use_targets if use_targets is not None else targets
        prob = pulp.LpProblem(f"Sinter_Burden_Opt_{tag}", pulp.LpMinimize)
        x = {
            m: pulp.LpVariable(
                f"x{tag}_{m}",
                lowBound=bounds[m][0],
                upBound=bounds[m][1]
            )
            for m in df.index
        }

        add_structural_constraints(
            prob, x, df, bounds, iron_ores, fluxes,
            iron_ore_max_pct, unavailable_iron,
            flux_max_pct, unavailable_flux, OUT,
            baseline_flux_portion,
            iron_tier, flux_tier_label,
            flux_min_pct_override=flux_min_pct_override,
            mandate_mode=mandate_mode,
            iol_nominal=iol_nominal,
            bf_nominal=bf_nominal,
            iol_fb_min=iol_fb_min,
            iol_fb_max=iol_fb_max,
            bf_fb_min=bf_fb_min,
            bf_fb_max=bf_fb_max
        )

        Fe_sum = pulp.lpSum(x[m] * df.loc[m, "Fe"] / 100 for m in x)
        SiO2_sum = pulp.lpSum(x[m] * df.loc[m, "SiO2"] / 100 for m in x)
        Al2O3_sum = pulp.lpSum(x[m] * df.loc[m, "Al2O3"] / 100 for m in x)
        CaO_sum = pulp.lpSum(x[m] * df.loc[m, "CaO"] / 100 for m in x)
        MgO_sum = pulp.lpSum(x[m] * df.loc[m, "MgO"] / 100 for m in x)

        prob += Fe_sum >= fe_lo, "Fe_min_hard"
        prob += Fe_sum <= fe_hi, "Fe_max_hard"
        prob += SiO2_sum <= t["SiO2_max"] * OUT / 100, "SiO2_max_hard"
        prob += Al2O3_sum <= t["Al2O3_max"] * OUT / 100, "Al2O3_max_hard"
        prob += (
            Al2O3_sum - t["Al2O3_SiO2_max"] * SiO2_sum <= 0
        ), "Al2O3_SiO2_max_hard"
        prob += (
            CaO_sum >= t["Basicity_min"] * SiO2_sum
        ), "Basicity_min_hard"
        prob += (
            CaO_sum <= t["Basicity_max"] * SiO2_sum
        ), "Basicity_max_hard"
        prob += MgO_sum >= t["MgO_min"] * OUT / 100, "MgO_min_hard"
        prob += MgO_sum <= t["MgO_max"] * OUT / 100, "MgO_max_hard"
        prob += CaO_sum >= t["CaO_min"] * OUT / 100, "CaO_min_hard"
        prob += CaO_sum <= t["CaO_max"] * OUT / 100, "CaO_max_hard"

        if enforce_b4:
            prob += (
                (CaO_sum + MgO_sum)
                - b4_min * (SiO2_sum + Al2O3_sum) >= 0
            ), "B4_min"
            prob += (
                (CaO_sum + MgO_sum)
                - b4_max * (SiO2_sum + Al2O3_sum) <= 0
            ), "B4_max"

        cost_expr = pulp.lpSum(
            x[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in x
        )

        if "COKE_BREEZE" in x:
            cb = x["COKE_BREEZE"]
            prob += cb >= coke_min_rate, "Coke_Practical_Min"
            prob += cb <= coke_max_rate, "Coke_Practical_Max"

            Q_fuel = cb * (coke_fc / 100) * coke_cv
            Q_moisture = latent_heat * pulp.lpSum(
                x[m] * moisture_factor.get(m, 0.0)
                for m in x if m != "COKE_BREEZE"
            )
            Q_calcination = calcination_heat * pulp.lpSum(
                x[m] * loi_factor.get(m, 0.0)
                for m in x if m != "COKE_BREEZE"
            )
            Q_melting = melting_heat * OUT
            Q_required = (
                Q_moisture + Q_calcination + Q_melting
            ) / (1 - loss_fraction)
            thermal_surplus = Q_fuel - Q_required

            if manual_override:
                prob += cb == manual_coke_rate, "Manual_Coke_Override"
            else:
                prob += Q_fuel >= Q_required, "Heat_Balance_Min"
                if enforce_firing_ratio_max:
                    prob += (
                        Q_fuel <= Q_required * firing_ratio_max
                    ), "Firing_Ratio_Max"

            # IMPORTANT v30:
            # FeO depends on thermal surplus, so changing burden chemistry,
            # moisture or LOI changes the coke requirement.
            FeO_pred = feo_ref_pct + feo_thermal_slope * (
                (thermal_surplus - feo_ref_surplus) / 10000.0
            )

            prob += FeO_pred >= feo_min, "FeO_Min_Hard"
            prob += FeO_pred <= feo_max, "FeO_Max_Hard"

            # Target is lexicographically preferred:
            # 1) get as close as possible to FeO target;
            # 2) among equally good FeO solutions, minimize material cost.
            feo_dev = pulp.LpVariable(
                f"FeO_Target_Deviation_{tag}", lowBound=0
            )
            prob += (
                feo_dev >= FeO_pred - feo_target
            ), f"FeO_Target_Dev_Pos_{tag}"
            prob += (
                feo_dev >= feo_target - FeO_pred
            ), f"FeO_Target_Dev_Neg_{tag}"

            if manual_override:
                prob.setObjective(cost_expr)
            else:
                prob.setObjective(feo_dev)
                prob.solve(pulp.PULP_CBC_CMD(msg=0))

                if pulp.LpStatus[prob.status] != "Optimal":
                    return prob, x, pulp.LpStatus[prob.status]

                min_dev = max(0.0, float(pulp.value(feo_dev) or 0.0))
                prob += (
                    feo_dev <= min_dev + 1e-6
                ), f"FeO_Target_Dev_Pin_{tag}"
                prob.setObjective(cost_expr)
        else:
            prob.setObjective(cost_expr)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return prob, x, pulp.LpStatus[prob.status]

    def _add_diagnostic_coke_constraints(prob, x, suffix=""):
        """Apply practical coke, heat and thermal-state FeO constraints to fallback blends."""
        if "COKE_BREEZE" not in x:
            return

        if manual_override:
            prob += (
                x["COKE_BREEZE"] == manual_coke_rate
            ), f"Manual_Coke_Diagnostic{suffix}"
            return

        cb = x["COKE_BREEZE"]
        prob += cb >= coke_min_rate, f"Coke_Practical_Min{suffix}"
        prob += cb <= coke_max_rate, f"Coke_Practical_Max{suffix}"

        Q_fuel = cb * (coke_fc / 100) * coke_cv
        Q_moisture = latent_heat * pulp.lpSum(
            x[m] * moisture_factor.get(m, 0.0)
            for m in x if m != "COKE_BREEZE"
        )
        Q_calcination = calcination_heat * pulp.lpSum(
            x[m] * loi_factor.get(m, 0.0)
            for m in x if m != "COKE_BREEZE"
        )
        Q_melting = melting_heat * OUT
        Q_required = (
            Q_moisture + Q_calcination + Q_melting
        ) / (1 - loss_fraction)

        prob += Q_fuel >= Q_required, f"Heat_Balance_Min{suffix}"

        if enforce_firing_ratio_max:
            prob += (
                Q_fuel <= Q_required * firing_ratio_max
            ), f"Firing_Ratio_Max{suffix}"

        thermal_surplus = Q_fuel - Q_required
        FeO_pred = feo_ref_pct + feo_thermal_slope * (
            (thermal_surplus - feo_ref_surplus) / 10000.0
        )

        prob += FeO_pred >= feo_min, f"FeO_Min_Hard{suffix}"
        prob += FeO_pred <= feo_max, f"FeO_Max_Hard{suffix}"

    def _finalize(prob, x, tag_label, note=None):
        blend = {m: round(x[m].value(), 2) for m in x}
        total_cost = pulp.value(prob.objective)
        achieved = compute_achieved(blend, df, OUT)
        diag = list(diagnostics)
        if note:
            diag.append(note)
        diag += _report_compensation(blend, df, iron_ores, fluxes, unavailable_iron, unavailable_flux,
                                      iron_ore_max_pct, flux_max_pct, iron_tier, flux_tier, OUT, mandate_reasons)
        diag.append(_report_fines_loading(blend, df, OUT))
        # Add heat balance info (using same coefficients as the solver)
        heat_info = compute_coke_heat_balance_diagnostic(blend, df, OUT, coke_cv, coke_fc,
                                                         latent_heat, calcination_heat, melting_heat, loss_fraction,
                                                         feo_min, feo_target, feo_max, feo_ref_surplus, feo_ref_pct,
                                                         feo_thermal_slope, ref_coke_cv, ref_coke_fc)
        if heat_info:
            if manual_override:
                diag.append(f"\n🔥 MANUAL OVERRIDE: Coke Breeze forced to {manual_coke_rate:.1f} kg/t")
            else:
                diag.append(f"\n🔥 Optimised Coke Breeze: {heat_info['CB_kg_LP_chosen']:.1f} kg/t")
            diag.append(f"   Firing Ratio: {heat_info['Firing_Ratio']:.3f} (max {firing_ratio_max})")
            diag.append(f"   Predicted FeO: {heat_info['FeO_Estimate_Pct']:.2f}% (target {feo_target:.2f}%, band {feo_min:.2f}-{feo_max:.2f}%)")
            diag.append(f"   {heat_info['Controller_Suggestion']}")
        return "Optimal", blend, total_cost, achieved, diag, False

    # ---- PASSES A-F (same as before) ----
    probA, xA, statusA = _build_and_solve(None, flux_tier, "A", mandate_mode="pinned")
    if statusA == "Optimal":
        return _finalize(probA, xA, "A", "✅ Mandates met at nominal.")

    diagnostics.append("⚠️ Base flux floors infeasible with pinned mandates – relaxing flux floors...")
    probB, xB, statusB = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "B", mandate_mode="pinned")
    if statusB == "Optimal":
        return _finalize(probB, xB, "B", "✅ Resolved with relaxed flux floors.")

    if shortage_targets is not None:
        diagnostics.append(f"🔄 Iron ore short – widening SiO2/CaO ceilings...")
        probC, xC, statusC = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "C", use_targets=shortage_targets, mandate_mode="pinned")
        if statusC == "Optimal":
            return _finalize(probC, xC, "C", "✅ Resolved with widened ceilings – mandates pinned.")

    diagnostics.append("⚠️ Pinned mandates infeasible – relaxing mandates...")
    probD, xD, statusD = _build_and_solve(None, flux_tier, "D", mandate_mode="pinned")
    if statusD == "Optimal":
        return _finalize(probD, xD, "D", "✅ Mandates relaxed to fallback bands.")

    probE, xE, statusE = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "E", mandate_mode="pinned")
    if statusE == "Optimal":
        return _finalize(probE, xE, "E", "✅ Mandates and flux floors relaxed.")

    if shortage_targets is not None:
        probF, xF, statusF = _build_and_solve(FLUX_MIN_PCT_QUALITY_RELAXED, "quality_relaxed", "F", use_targets=shortage_targets, mandate_mode="pinned")
        if statusF == "Optimal":
            return _finalize(probF, xF, "F", "✅ All flexibilities used – solved.")

    # ============================ PHASE 1 & 2 (diagnostic) ============================
    diagnostics.append("⚠️ All hard constraints infeasible – generating closest achievable reference.")
    prob1 = pulp.LpProblem("Phase1_MinDeviation", pulp.LpMinimize)
    x1 = {m: pulp.LpVariable(f"x1_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
    add_structural_constraints(prob1, x1, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="pinned",
                                iol_nominal=iol_nominal, bf_nominal=bf_nominal,
                                iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                                bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max)
    _add_diagnostic_coke_constraints(prob1, x1, suffix="_p1")
    slacks1, sums1 = build_soft_vars_and_constraints(prob1, x1, df, OUT, targets, fe_lo, fe_hi, suffix="_p1")
    obj1 = weighted_deviation_expr(slacks1, targets, OUT)
    prob1 += obj1, "Total_Weighted_Deviation"
    prob1.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob1.status] != "Optimal":
        diagnostics.append("❌ No feasible blend even in diagnostic mode. Check inventory.")
        return "Infeasible", None, None, None, diagnostics, True

    blend1 = {m: round(x1[m].value(), 2) for m in x1}
    achieved1 = compute_achieved(blend1, df, OUT)
    phase1_slack_values = {k: (v.value() or 0.0) for k, v in slacks1.items()}
    dev_report = {
        "Fe": phase1_slack_values["Fe_under"] + phase1_slack_values["Fe_over"],
        "SiO2": phase1_slack_values["SiO2_over"],
        "Al2O3": phase1_slack_values["Al2O3_over"],
        "Al2O3/SiO2": phase1_slack_values["ratio_over"],
        "Basicity": phase1_slack_values["Bas_under"] + phase1_slack_values["Bas_over"],
        "MgO": phase1_slack_values["MgO_under"] + phase1_slack_values["MgO_over"],
        "CaO": phase1_slack_values["CaO_under"] + phase1_slack_values["CaO_over"],
    }
    binding_spec = max(dev_report, key=dev_report.get)
    diagnostics.append(f"📊 Most binding spec: {binding_spec} (deviation = {dev_report[binding_spec]:.3f})")
    diagnostics.append("🏃 Phase 2: minimising cost at same deviation...")

    prob2 = pulp.LpProblem("Phase2_MinCost", pulp.LpMinimize)
    x2 = {m: pulp.LpVariable(f"x2_{m}", lowBound=bounds[m][0], upBound=bounds[m][1]) for m in df.index}
    add_structural_constraints(prob2, x2, df, bounds, iron_ores, fluxes, iron_ore_max_pct, unavailable_iron,
                                flux_max_pct, unavailable_flux, OUT, baseline_flux_portion, iron_tier, "quality_relaxed",
                                flux_min_pct_override=FLUX_MIN_PCT_QUALITY_RELAXED, mandate_mode="pinned",
                                iol_nominal=iol_nominal, bf_nominal=bf_nominal,
                                iol_fb_min=iol_fb_min, iol_fb_max=iol_fb_max,
                                bf_fb_min=bf_fb_min, bf_fb_max=bf_fb_max)
    _add_diagnostic_coke_constraints(prob2, x2, suffix="_p2")
    slacks2, sums2 = build_soft_vars_and_constraints(prob2, x2, df, OUT, targets, fe_lo, fe_hi, suffix="_p2")
    for key, var in slacks2.items():
        p1_val = phase1_slack_values.get(key, 0.0)
        prob2 += var <= p1_val + PIN_TOLERANCE, f"Pin_{key}"
    prob2 += pulp.lpSum(x2[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in x2), "Total_Cost"
    prob2.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob2.status] == "Optimal":
        blend2 = {m: round(x2[m].value(), 2) for m in x2}
        cost2 = sum(blend2[m] * df.loc[m, "Price_Rs_t"] / 1000 for m in blend2)
        achieved2 = compute_achieved(blend2, df, OUT)
        diagnostics.append(f"✅ Phase 2 complete. Reference cost: Rs {cost2:,.2f}/t")
        return "Infeasible", blend2, cost2, achieved2, diagnostics, True
    else:
        diagnostics.append("⚠️ Phase 2 issue – returning Phase 1 blend.")
        return "Infeasible", blend1, None, achieved1, diagnostics, True

# ============================================================================
# 11. BASELINE BLEND WRAPPER
# ============================================================================
def get_baseline_blend(df, targets, enforce_b4=False):
    result = solve_blend_with_compensation(df, 1000, targets, baseline_blend=None, enforce_b4=enforce_b4)
    status, blend, cost, achieved = result[0], result[1], result[2], result[3]
    if status == "Optimal":
        return blend, cost, achieved
    else:
        return None, None, None

# ============================================================================


# Dashboard/backend quality targets — single source of truth for v30.
TARGETS = {
    "Fe_min": FE_TARGET,
    "SiO2_max": 5.8,
    "Al2O3_max": 4.5,
    "Al2O3_SiO2_max": 0.98,
    "Basicity_min": 1.9,
    "Basicity_max": 2.0,
    "MgO_min": 2.2,
    "MgO_max": 2.4,
    "CaO_min": 10.5,
    "CaO_max": 11.5,
}
