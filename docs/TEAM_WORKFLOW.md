# Team Workflow

## Individual teammate

1. Pull or open the repository.
2. Double-click `START_PROJECT.bat`.
3. Select your name.
4. Select the model you want to run.
5. Choose Standard or Tuned mode.
6. Click **Run Analysis**.
7. Review the Individual Results page.
8. Download the run summary if useful for the individual report section.
9. Click **Save Team Results** if using the same shared checkout, or **Export Result Package** if working independently.
10. Send the exported ZIP to the teammate assembling the final comparison.

## Team assembler

1. Pull the latest repository.
2. Launch the application.
3. Open **Team Comparison**.
4. Import each teammate result package.
5. Confirm every run shows the same experiment fingerprint.
6. Confirm the intended algorithm ownership; duplicate model selections are allowed but generate a warning.
7. Review the standardized comparison table and charts.
8. Download **Team Comparison CSV** for the final report.
9. Use the same metrics and model names in the written conclusion.

## Fair-comparison rule

Do not change the dataset, target definition, holdout split, random state, cross-validation contract, or positive-class definition for one teammate only. Those settings define the experiment fingerprint and are intentionally locked for comparability.
