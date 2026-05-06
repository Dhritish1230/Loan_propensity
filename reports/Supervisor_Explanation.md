# April Prediction vs Label Validation

## Core Message for Supervisor
April was scored first without using DIY/SFDC conversion labels. After scoring, labels were joined only for validation.

## T1 Top 100 Check
- Same users before vs after labels: True
- Same order before vs after labels: True
- Same scores before vs after labels: True
- Converted users in top 100 after labels: 53
- DIY conversions in top 100: 7
- SFDC conversions in top 100: 46

## Interpretation
The ranking did not change after labels were added. This means the prediction output and the validated outcome list are aligned correctly. Labels only added the actual outcome columns, proving that validation was done after prediction.
