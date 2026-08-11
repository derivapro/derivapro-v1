# DerivaPro Portfolio Workflow

## Purpose

The Portfolio workspace is the local trading-book layer of DerivaPro. It is designed to let users define a portfolio first, add positions across asset classes, and then use product pricing workflows to value or revalue positions over time.

This is different from a single-product pricing page:

- Product pages price one instrument or one product family.
- Portfolio pages organize many positions into a book.
- Portfolio valuation and risk analytics aggregate saved or future pricing results across positions.

## Recommended User Workflow

1. Create a named portfolio from the Portfolio workspace.
2. Open the portfolio.
3. Add manual positions for the book composition.
4. Use product-specific pricing pages to value selected instruments when needed.
5. Add saved pricing results to the portfolio or update positions with pricing references.
6. Export the portfolio JSON as a private local copy.
7. Import the local JSON later to restore or clone the book.

## Position Types

The current portfolio workflow supports two practical position categories.

### Manual / Unpriced Position

Use this for portfolio construction before valuation. A manual position can represent:

- Cash equity or ETF holding.
- Listed equity option.
- Barrier or Asian option.
- Structured note.
- Bond.
- Swap, swaption, forward, future, CDS, FX, or commodity position.
- Placeholder trade that will be priced later.

Manual positions store trade metadata and flexible terms JSON. They do not require a pricing result.

### Saved Pricing Result Position

Use this when a user has already run a pricing page and wants to add that saved valuation to a portfolio.

Saved-result positions can contribute:

- Price / market value.
- Greeks, where available.
- Stored model parameters.
- Stored result JSON.

## Position Fields

The portfolio position schema includes:

| Field | Description |
|---|---|
| `position_label` | User-facing label for the holding or trade. |
| `trade_id` | Optional desk, booking, or internal trade identifier. |
| `side` | `long` or `short`. |
| `quantity` | Unit count, contract count, or holding amount. |
| `notional` | Optional notional amount. If populated, aggregation uses notional as the exposure multiplier. |
| `currency` | Position currency. |
| `asset_class` | Broad category such as Equity Derivatives, Structured Products, Fixed Income, Rates, Credit, FX, or Commodities. |
| `product_category` | More specific product type such as European Option, Barrier Reverse Convertible, Bond, Swap, or CDS. |
| `underlying` | Underlying asset, reference entity, rate index, basket, or ticker. |
| `valuation_status` | `unpriced`, `ready_for_pricing`, `priced`, or `external`. |
| `terms_json` | Flexible product-specific terms stored on the linked instrument record. |
| `notes` | User notes, assumptions, booking comments, or data-source notes. |

## Local JSON Copies

DerivaPro writes user portfolio JSON copies under:

```text
local_data/portfolios/
```

This folder is ignored by Git. User-created portfolio files can contain confidential holdings, trade IDs, and pricing assumptions, so they must not be committed to GitHub.

The public repository includes only a sample JSON file:

```text
derivapro/static/sample_portfolios/equity_derivatives_portfolio.json
```

That file is illustrative only and should not be treated as market data, valuation guidance, or production portfolio content.

## Current Limitations

- Manual positions are book-construction records; most are not automatically repriced yet.
- Portfolio-level market data refresh is not yet implemented.
- Portfolio-level scenario analysis, stress testing, VaR/CVaR, and risk attribution are planned future layers.
- Cross-currency aggregation currently stores currency but does not yet apply FX conversion.
- Product-specific term validation is intentionally light in the manual-position form; detailed validation belongs in the product pricing workflow.

## Planned Enhancements

- Link manual positions to product pricing pages.
- Price selected positions or the full portfolio from the Portfolio workspace.
- Add portfolio-level scenario templates by asset class.
- Add risk contribution by underlying, asset class, product type, and factor.
- Add CSV/XLSX portfolio import for desk-friendly bulk onboarding.
- Add portfolio snapshots and valuation-date history.
