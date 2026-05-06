# Original Contributor Note

This repository is a portfolio-safe handover version of the loan propensity project.

The project combines:

- Multi-month training logic.
- T0/T1 model separation.
- Raw scoring pipeline.
- Label-after-score validation.
- Per-call behavior analysis.
- Dashboard and Power BI reporting.

## Why Original Context Matters
Some parts of the project depend on business and data context that is not obvious from code alone:

- Why T0 is not judged the same way as T1.
- Why call duration is valid for T1 but not for T0.
- Why labels must be joined after scoring.
- How raw call data joins were validated.
- Why excessive call counts may indicate fatigue rather than opportunity.

If this project is extended or productionized, preserve the T0/T1 separation and leakage-safe validation workflow.

## Contact Placeholder
Original project contributor: `ADD_NAME_HERE`

Preferred contact: `ADD_EMAIL_OR_LINKEDIN_HERE`

Replace the placeholders before sharing this repository outside the organization.
