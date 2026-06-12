# DerivaPro Lite Demo

DerivaPro Lite is a static, browser-only demo for the DerivaPro platform.

It is intended for quick product demonstrations where users should not need to install Python, Flask, QuantLib, or any other package.

## Public demo link

```text
https://derivapro.github.io/derivapro-v1/demo/
```

## Requirements

Only a modern web browser is required:

- Google Chrome
- Microsoft Edge
- Safari
- Firefox

No Python environment, package installation, API key, database, or Flask server is required.

## Data and calculation scope

The Lite demo uses:

- browser-side JavaScript formulas
- user-entered inputs
- static sample market data from `demo/data/sample_market_data.json`

The sample data includes:

- a sample spot price
- a sample volatility surface
- a sample yield curve
- illustrative portfolio sensitivities

It does not use live market data, yfinance, FRED, QuantLib, Azure/OpenAI, or the Flask backend.

## Included demo modules

- European option pricing and Greeks
- Static volatility-surface lookup and interpolation
- Fixed-rate bond pricing, duration, convexity, and DV01
- Static yield-curve lookup and interpolation
- Forward contract fair value and payoff chart
- Plain vanilla swap quick valuation
- Portfolio stress illustration
- Sample market data charts

## How to run locally

Open this file in a browser:

```text
demo/index.html
```

If the demo is downloaded as part of the repository, keep the folder structure intact so the logo path and data file continue to resolve:

```text
derivapro-v1/
|-- demo/
|   |-- index.html
|   |-- styles.css
|   |-- app.js
|   |-- README.md
|   `-- data/
|       `-- sample_market_data.json
`-- derivapro/
    `-- static/
        `-- logo.jpg
```

When opened directly through `file://`, the demo falls back to embedded sample market data if the browser blocks local JSON loading.

## Important note

The calculations in this folder are simplified approximations. The full Flask application remains the source for production analytics, market-data workflows, model governance, report generation, and future portfolio-level risk management.
