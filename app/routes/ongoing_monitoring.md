# Ongoing Model Monitoring 

## Overview
This test is an assessment of the adequacy of ongoing model governance, including:
- Ongoing monitoring plan
- Assumptions management plan
- Model approval process
- Change management

## Testing Performed
The ongoing governance arrangements, as described in the model document, were assessed to determine:
- Completeness of coverage of performance monitoring tests, metrics, testing frequency, and escalation plans
- Completeness of coverage across the model’s key assumptions that require ongoing consideration and calibration
- Whether model approval and change management processes are aligned to the model risk management framework

Model Risk Management (MRM) noted that model documentation provided the below tables to explain the ongoing monitoring tests, acceptance criteria, and frequency.

<!-- HTML Table Starts Here -->

<h3>Table: Ongoing Monitoring Plan for the Autocallable Model</h3>

<table>
  <thead>
    <tr>
      <th>S. No</th>
      <th>Test Name</th>
      <th>Test Description</th>
      <th>Acceptance Criteria</th>
      <th>Test Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Risk Based P&L Predict Analysis</td>
      <td>For a set of benchmark trades across different currencies, the daily P&L is compared with the P&L predicted using the sensitivities reported by the model.</td>
      <td>
        It is expected that the residual/unexplained P&L, defined as Actual P&L - Predicted P&L, is small. 
        <br><br>
        Here, Predicted P&L = (Risk-based P&L from market moves + P&L attributed to theta + P&L attributed to events). 
        <br><br>
        Cases with material unexplained P&L are marked for further analysis. The materiality is judged based on:
        <ul>
          <li>Absolute value of the unexplained P&L</li>
          <li>Percentage of the unexplained P&L relative to the realized P&L</li>
        </ul>
        No quantitative thresholds are set; they will be set after onset of trading.
      </td>
      <td>
        Daily, using automatic IT workflows, the results of this test are made available to relevant stakeholders of the Model Risk management committee.
        <br><br>
        Additional ad-hoc analysis may be performed at the request of the trading desk, especially for analysis on potential hedge trades.
      </td>
    </tr>
    <tr>
      <td>2</td>
      <td>Sensitivity Analyses with respect to key variables</td>
      <td>
        For a set of benchmark trades, sensitivity analyses of PV and Greeks with respect to key model inputs are carried out. These include:
        <ul>
          <li>Spot/Volatility scenarios</li>
          <li>Correlation scenarios</li>
          <li>FX scenarios</li>
          <li>IR and dividend scenarios</li>
        </ul>
      </td>
      <td>
        This test is mainly for diagnostic purposes, so no quantitative thresholds are set.
      </td>
      <td>
        Weekly, using automatic IT workflows, the results of this test are made available to relevant stakeholders of the Model Risk management committee.
      </td>
    </tr>
  </tbody>
</table>
<!-- HTML Table Ends Here -->

The assumptions management plan defined in the model documentation states that the underlying pricing models are, in principle, sensitive to the forward values calculation and relevant discount factors. In addition, discount factors are dependent on the yield curve methodology. The quality of curve construction is monitored as part of the ongoing monitoring plan for the Yield Curve model, as outlined in [12]. Therefore, risk-based P&L explain serves as a useful test to ensure that the pricing models and sensitivities are able to explain day-over-day P&L for the positions. However, no threshold is suggested for unexplained P&L analysis, which has been left subjective. As such, MVT has raised a low-risk finding in the next section.

The assumptions management plan defines the frequency and processes relating to ongoing assumptions management. However, the assumptions management plan does not capture interpolation assumptions used for implied volatilities.

The model’s approval and change management process is subject to Model Risk Management (MRM) procedures and standards, which include model approval when the model passes validation and a change log to track changes to the model.
