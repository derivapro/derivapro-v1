**Description:**
An interest rate swap is a derivative where counterparties exchange cash flows, typically fixed versus floating, on a notional amount. The mark-to-market value is the present value difference between receive and pay legs under the chosen curve and conventions.

At a high level:

* $Swap\ Value = PV(Receive\ Leg) - PV(Pay\ Leg)$

This workflow supports custom leg-level assumptions (dates, rates, conventions, tenors, shocks) so users can value and stress test bilateral swap structures.

**Instructions:**
Use this page to configure and value pay/receive swap legs.

1. Enter global assumptions (valuation date, shocks, interpolation, compounding, currency, and calendar).
2. Configure pay leg cash-flow inputs (fixed/floating settings, spot/index dates and rates, day count, tenor).
3. Configure receive leg cash-flow inputs with corresponding conventions.
4. Run valuation to obtain PV by leg and total swap value.
5. Review shocked outputs to assess sensitivity and P&L behavior.