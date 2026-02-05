# public-debt-econometrics-r-python
Panel econometrics on OECD public debt (1990–2023) using R. Fixed/Random effects, Hausman, VIF, BP, DW tests.

## How to run (R)
1. Put the dataset in `data/FinalDataset.xlsx`
2. Open `scripts/main_analysis.R`
3. Install required packages (plm, lmtest, sandwich, car, readxl, dplyr)
4. Run the script end-to-end


# OECD Public Debt — Panel Econometrics (R)

This project is part of an academic research work analyzing the determinants of public debt across OECD countries using panel econometrics and macroeconomic indicators.

## Objective
Identify the macroeconomic and demographic factors associated with high levels of public debt using panel data models.

## Dataset
- Source: World Bank – World Development Indicators (WDI)
- Countries: 36 OECD members
- Period: 1991–2024
- Observations: 1,224 country-year points
- See `data/DATA_DICTIONARY.md`

## Methodology
Panel econometrics using R:

- Pooled OLS
- Fixed Effects / Random Effects models
- Hausman test
- Breusch–Pagan test (heteroskedasticity)
- Durbin–Watson test (autocorrelation)
- VIF (multicollinearity diagnostics)
- Robust standard errors (White)

## Skills Demonstrated
- Panel data econometrics
- Data cleaning and restructuring (wide → long)
- Statistical diagnostics
- Reproducible research structure
- Data storytelling through academic reporting.
