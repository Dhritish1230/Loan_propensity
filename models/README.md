# Model Artifacts

This folder contains the final portable model files required by the raw scoring dashboard.

## Included Files

- `loan_model_t0_call_targeting_mixed_hybrid.pkl`: T0 pre-call targeting model.
- `feature_columns_t0_call_targeting_mixed_hybrid.pkl`: ordered feature list used by the T0 model.
- `loan_model_t1_sequence_mixed_hybrid.pkl`: T1 post-call loan conversion model.
- `feature_columns_t1_sequence_mixed_hybrid.pkl`: ordered feature list used by the T1 model.

## How The Dashboard Uses Them

The scoring pipeline loads these files from `models/` by default:

```powershell
python -m uvicorn src.multi_month_training.raw_prediction_dashboard:app --host 127.0.0.1 --port 8057
```

If the artifacts are stored somewhere else, set:

```powershell
$env:LOAN_PROPENSITY_MODEL_DIR = "C:\path\to\models"
```

If loading the pickle files manually outside the dashboard, make sure the repository `src/` folder is on `PYTHONPATH` because the model uses a custom hybrid model class:

```powershell
$env:PYTHONPATH = "$(Resolve-Path .\src)"
python -c "import joblib; print(type(joblib.load('models/loan_model_t1_sequence_mixed_hybrid.pkl')).__name__)"
```

## Safety Note

These are Python pickle/joblib files. Only load them in a trusted environment.

Raw customer data is not stored in this repository.
