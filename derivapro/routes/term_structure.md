**Description:**

The term structure describes the relationships between interest rates and the maturity of debt instruments. It's often comprised of the zero rate curve, discount factor curve and forward curve. Shocks are applied to the term structure to visualize the impact of various interest rate environments on downstream calculcations. The two shocks are parallel and non-parallel.

For parallel shocks, changes are applied equally across all points of the yield curve where interest rates at every maturity increase or decrease by the same number of basis points.

For non-parallel shocks, changes to the interest rate curve where different parts of the curve move by different amounts. Steepener shocks are where short-term interest rates fall and long-term interest rates rise. On the other hand, flattener shocks are where short-term interest rates rise and long-term interest rates fall.

The non-parallel shock distributions are shared below, adhering to the industry standard (Basel Committee's Interest Rate Risk in the Banking Book (IRRBB))

* Steepener:
$$ \(-0.65 \cdot Absolute Shock \cdot e^{-Tenor(years)/4} + 0.9 \cdot Absolute Shock \cdot (1 - e^{-Tenor(years)/4})) $$

* Flattener:
$$ \(0.8 \cdot Absolute Shock \cdot e^{-Tenor(years)/4} - 0.6 \cdot Absolute Shock \cdot (1 - e^{-Tenor(years)/4})) $$

**Instructions:**
Placeholder