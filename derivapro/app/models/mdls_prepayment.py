import QuantLib as ql
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np

class Prepayment:
    def __init__(self, orig_rate, market_rate, orig_fico, loan_age, orig_ltv, intercept, beta_spread, beta_fico, beta_loan_age, beta_ltv):
        self.orig_rate = orig_rate
        self.market_rate = market_rate
        self.orig_fico = orig_fico
        self.loan_age = loan_age
        self.orig_ltv = orig_ltv
        self.intercept = intercept
        self.beta_spread = beta_spread
        self.beta_fico = beta_fico
        self.beta_loan_age = beta_loan_age
        self.beta_ltv = beta_ltv
    

    def prepayment_probability(self):
        refi_incentive = self.orig_rate - self.market_rate

        log_odds = (
            self.intercept +
            self.beta_spread * refi_incentive +
            self.beta_fico * (self.orig_fico - 700) +  # center FICO
            self.beta_loan_age * self.loan_age +
            self.beta_ltv * self.orig_ltv
            )
        probability = round(1 / (1 + np.exp(-log_odds)) * 100, 2)
        return probability

# odds = Prepayment(
#     orig_rate = 5.875,
#     market_rate = 4,
#     orig_fico = 666,
#     loan_age = 50,
#     orig_ltv = 80,
#     intercept = -6.0,
#     beta_spread = 6.0,        # Strong incentive to prepay if rates drop
#     beta_fico = 0.005,        # Higher credit → more likely to refinance
#     beta_loan_age = 0.01,     # Older loans → higher prepayment likelihood
#     beta_ltv = -0.03,         # Higher LTV → harder to refinance (inverse relationship)
# )

# print("Prepayment Probability: ", odds.prepayment_probability(), "%")



