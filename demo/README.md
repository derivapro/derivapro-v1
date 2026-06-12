# DerivaPro Lite Demo

DerivaPro Lite is a static, browser-only demo for the DerivaPro platform.

It is intended for quick product demonstrations where users should not need to install Python, Flask, QuantLib, or any other package.

## Requirements

Only a modern web browser is required:

- Google Chrome
- Microsoft Edge
- Safari
- Firefox

No Python environment, package installation, API key, database, or Flask server is required.

## How to run locally

Open this file in a browser:

```text
demo/index.html
```

If the demo is downloaded as part of the repository, keep the folder structure intact so the logo path continues to resolve:

```text
derivapro-v1/
|-- demo/
|   |-- index.html
|   |-- styles.css
|   |-- app.js
|   `-- README.md
`-- derivapro/
    `-- static/
        `-- logo.jpg
```

## How to share with other users

Recommended options:

1. Host the repository or the `demo/` folder with GitHub Pages.
2. Zip the repository and ask users to open `demo/index.html` locally.
3. Deploy the `demo/` folder to any static hosting service, such as Netlify, Vercel, GitHub Pages, or an internal web server.

For public demos, GitHub Pages is usually the simplest path.

## Included demo modules

- European option pricing and Greeks
- Fixed-rate bond pricing, duration, convexity, and DV01
- Forward contract fair value and payoff chart
- Plain vanilla swap quick valuation
- Portfolio stress illustration

## Important note

The calculations in this folder are simplified browser-side approximations. The production analytics and model-governance workflows live in the Flask application.
