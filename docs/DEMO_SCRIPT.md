# Five-Minute Demo Script

## 1. Start With The Business Problem
"The business has too many possible customers to call, so the question is not just who might convert, but who should be prioritized first."

## 2. Explain The Two-Stage Model
"I separated the workflow into T0 and T1."

- "T0 is before calls. It ranks whom to call first."
- "T1 is after calls. It ranks who is most likely to take the loan using call behavior."

## 3. Show The Clean Validation Story
"For April, I scored the users first without using conversion labels. Only after predictions were saved did I join DIY/SFDC labels."

Key proof:

- Same top-100 users before and after labels.
- Same top-100 order before and after labels.
- Same scores before and after labels.
- 53 of the top 100 actually converted.

## 4. Show The Main Metrics
"The April holdout result was strong."

- AUC: 0.849
- Precision@100: 53%
- Top-decile lift: 6.65x
- Baseline conversion: 0.83%

Simple interpretation:

"Randomly picking 100 users would likely find fewer than one conversion. The model top 100 found 53."

## 5. Show Per-Call Behavior
"I also analyzed how behavior changes per call."

Key points:

- Conversion improves sharply when calls become meaningful.
- 30-59 second average duration had the strongest observed conversion bucket.
- Too many low-quality calls show diminishing returns.

Business interpretation:

"The answer is not call more blindly. The answer is call enough to create engagement, then use T1 to prioritize follow-up."

## 6. Close With The Value
"This is not only a model; it is a business workflow. It produces call queues, follow-up lists, validation reports, per-call insights, and Power BI-ready reporting."

## 7. Suggested Closing Line
"The project is ready for a controlled business pilot, and the next step would be monthly monitoring plus a controlled call-depth experiment."
