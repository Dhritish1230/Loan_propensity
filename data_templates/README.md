# Data Templates

These are fake, safe templates for dashboard uploads. Replace the example rows with real monthly data when running locally.

## Files

- `user_data_template.csv`: required for T0 and T1 scoring.
- `call_data_template.csv`: optional for T0, required for useful T1 scoring.
- `label_template_optional.csv`: optional validation input. Do not upload labels until after prediction if you are doing a clean holdout test.

The dashboard joins user and call data through `user_id`. Validation labels join through `uid`.
