# CPS Enrollment Forecasting — Pipeline Flow (Plain-English Guide)

This explains **`CPS_Enrollment_Combined_Pipeline.ipynb`**. The notebook now runs
top-to-bottom in **six clear stages**, each with a `# STAGE N — …` header.

## The one-line idea

> **Two models, no overlap.**
> - **GBT** (Spark gradient-boosted trees) → grades **1–8, 10–12**
> - **XGBoost** → the entry grades **K & 9** (no normal "grade below", so they get
>   their own model that keeps the real Pre‑K→K / Grade 8→9 feeder — nulls kept, not filled)
>
> Each model is trained **and** tested only on its own grades. Their predictions
> are merged, and we measure that combined system.

---

## The flow, in order

```
STAGE 1  FEATURE ENGINEERING   raw tables ─► one leak-free row per (school × grade × year)
STAGE 2  TRAIN BOTH MODELS     GBT ◄ grades 1–8,10–12   |   XGBoost ◄ K & 9
STAGE 3  VALIDATION            score the held-out year with both ─► combined MAE/RMSE/MAPE/MedAPE/R²
STAGE 4  YEAR-WISE BACKTEST    walk-forward, both models per year ─► out-of-sample predictions
STAGE 5  VISUALIZE             year-wise · grade-wise · network-wise charts
STAGE 6  SAVE                  save BOTH models (GBT + XGBoost) + write the forecast
```

---

## Stage 1 — Feature engineering

Turns raw tables into the **feature table** both models read.

- Load raw enrollment + `dim_school` (school name, network, type).
- Build, per school × grade × year: same-grade history, the **feeder grade**
  (Pre‑K for K, Grade 8 for 9), survival rates, school/district totals, school-type flags.
- **No leakage:** every feature uses *earlier* years only.
- **Output:** `ml_enrollment_feature_table_v4` + the CSV the Streamlit app reads.
- A short **K/9 audit** confirms the entry grades are present.

## Stage 2 — Train both models (split by grade)

Same 18-column feature list for both (`FEATURE_COLS` / `XGB_FEATURES`).

- **GBT (grades 1–8, 10–12)** — trained on years **2020–2025**, surge years
  down-weighted, missing values filled with **train-only medians**.
- **XGBoost (K & 9 only)** — trained on **K/9 rows only**, same year window, with
  missing values **kept as NaN** so the real Pre‑K→K / 8→9 feeder is preserved.

Both models exist by the end of this stage: `model` (GBT) and `xgb_k9_model` (XGBoost).

## Stage 3 — Validation → combined metrics

Scores the **held-out validation year** with each model on its own grades, merges
them, and prints the **combined system vs CSR**:

| Metric | Meaning |
|--------|---------|
| **MAE** | average students off, per school-grade |
| **RMSE** | like MAE but punishes big misses |
| **MAPE / MedAPE** | average / typical **% error** |
| **R²** | share of the ups-and-downs explained |

(`val_summary` holds the table.) This stage also includes a **K & 9 head-to-head**
— the all-grades **GBT** vs the dedicated **XGBoost**, scored on the same held-out
K/9 rows (MAE/MAPE/MedAPE table + chart) — which is *why* the forecast overwrites
K/9 with XGBoost. Feature importance and the CSR baseline also live here.

## Stage 4 — Year-wise backtest (walk-forward)

For each past year `T`: refit **GBT on continuing grades** and **XGBoost on K & 9**
using only years **before** `T`, predict `T`, and **merge**. The result, `preds_pdf`,
has `y_hat_ML` = the **hybrid** prediction — so everything downstream is the
combined system vs CSR. A check asserts no grade is served by both models.

- **Output:** `preds_pdf` + the saved table `combined_system_predictions`, then
  `cmp_year` / `cmp_network` / `cmp_grade` (per-year, per-network, per-grade numbers).

## Stage 5 — Visualize

- **Year-wise & network-wise** — trend lines + grouped bars.
- **Grade-wise** — MAE on a **log scale** (so Grade 9's spike doesn't hide the
  other grades) + MedAPE.

All driven by `preds_pdf`, so the charts show the **combined system**.

## Stage 6 — Save both models + forecast

- **GBT** → zipped to `Files/models/gbt_enrollment_model.zip`.
- **XGBoost (K & 9)** → `Files/models/xgb_model_k9.pkl`.
- Next-year **forecast** written to a table — **hybrid**: GBT scores grades
  1–8/10–12, then K & 9 are overwritten with `xgb_k9_model` predictions (raw
  features, nulls kept), so the forecast uses the same routing as the evaluation.

---

## Quick "where do I look?" cheat-sheet

| I want to see… | Stage |
|----------------|-------|
| The features and how they're built | **1** |
| GBT training | **2** |
| XGBoost (K/9) training | **2** |
| **Combined validation MAE / MAPE / etc.** | **3** |
| Year-by-year backtest numbers | **4** |
| Year / network / grade charts | **5** |
| Both models being saved | **6** |

> **Note:** This notebook was reorganized into this clean order. A backup of the
> previous version is saved as `CPS_Enrollment_Combined_Pipeline.ipynb.bak` — once
> you've run the new one top-to-bottom in Fabric and it looks right, you can delete it.
