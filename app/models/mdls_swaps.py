import QuantLib as ql
import datetime

class Swaps:
    def __init__(self, value_date, calendar, interpolation, compounding, currency, shocks,
                 pay_spot_dates, pay_spot_rates, pay_day_count,
                 pay_index_dates, pay_index_rates, pay_compounding_frequency,
                 rec_spot_dates, rec_spot_rates, rec_day_count,
                 rec_index_dates, rec_index_rates, rec_compounding_frequency, epsilon=0.001):
        # Applying to both legs
        self.value_date = self._convert_to_quantlib_date(value_date)
        ql.Settings.instance().evaluationDate = self.value_date
        self.calendar = calendar
        self.interpolation = interpolation
        self.compounding = compounding
        self.currency = currency
        self.pay_spot_dates = [self._convert_to_quantlib_date(date.strip()) for date in pay_spot_dates.split(',')]
        self.pay_spot_rates = [float(rate.strip()) for rate in pay_spot_rates.split(',')]
        self.pay_index_dates = [self._convert_to_quantlib_date(date.strip()) for date in pay_index_dates.split(',')]
        self.pay_index_rates = [float(rate.strip()) for rate in pay_index_rates.split(',')]
        self.shocks = [float(shock.strip()) for shock in shocks.split(',')]
        self.pay_day_count = pay_day_count
        self.pay_compounding_frequency = pay_compounding_frequency
        self.rec_spot_dates = [self._convert_to_quantlib_date(date.strip()) for date in rec_spot_dates.split(',')]
        self.rec_spot_rates = [float(rate.strip()) for rate in rec_spot_rates.split(',')]
        self.rec_index_dates = [self._convert_to_quantlib_date(date.strip()) for date in rec_index_dates.split(',')]
        self.rec_index_rates = [float(rate.strip()) for rate in rec_index_rates.split(',')]
        self.rec_day_count = rec_day_count
        self.rec_compounding_frequency = rec_compounding_frequency
        self.epsilon = epsilon
    
    def _convert_to_quantlib_date(self, date_input):
        """Helper function to convert date string, datetime.date, or QuantLib Date to QuantLib Date format."""
        if isinstance(date_input, str):
            return ql.Date(*[int(i) for i in date_input.split('-')[::-1]])  # Assumes date format is 'YYYY-MM-DD' 
        elif isinstance(date_input, datetime.date):
            return ql.Date(date_input.day, date_input.month, date_input.year)
        elif isinstance(date_input, ql.Date):
            return date_input       
        else:
            raise ValueError("Date format not recognized. Use 'YYYY-MM-DD' string, datetime.date, or QuantLib Date.")

    def _create_yield_curve(self, shock):
        # Apply shocks to rates
        shocked_rates = [rate + shock for rate in self.spot_rates]
        # Create a yield curve with shocked rates
        yield_curve = ql.ZeroCurve(self.spot_dates, shocked_rates, self.day_count, 
                                   self.calendar, self.interpolation, self.compounding, self.compounding_frequency)
        return ql.YieldTermStructureHandle(yield_curve)

    def fixed_rate(self, issue_date, maturity_date, tenor, coupon_rate, face_value):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issue_date = self._convert_to_quantlib_date(issue_date)
        maturity_date = self._convert_to_quantlib_date(maturity_date)
        
        results = {'Values': {'NPV': 'NPV', 'Accrued Interest': 'Accrued Interest',
                              'Price': 'Price', 'YTM': 'YTM', 'Duration': 'Duration',
                              'Dollar Duration': 'Dollar Duration', 'Convexity': 'Convexity',
                              'Dollar Convexity': 'Dollar Convexity'}}
        
        for shock in self.shocks:
            curve_handle = self._create_yield_curve(shock)
            schedule = ql.Schedule(
                issue_date, maturity_date, tenor, self.calendar, 
                ql.Following, ql.Following, ql.DateGeneration.Backward, False
            )
            bond = ql.FixedRateBond(0, float(face_value), schedule, [float(coupon_rate)], self.day_count)
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_handle))
            
            accrued_interest = bond.accruedAmount()
            
            ytm = bond.bondYield(self.day_count, self.compounding, self.compounding_frequency)
            interest_rate = ql.InterestRate(ytm, self.day_count, self.compounding, self.compounding_frequency)
            
            results[round(shock * 10000)] = {
                'NPV': bond.NPV(),
                'Accrued Interest': accrued_interest,
                'Price': bond.cleanPrice(),
                'YTM': ytm,
                'Duration': ql.BondFunctions.duration(bond, interest_rate),
                'Dollar Duration': (bond.NPV() / 100) * ql.BondFunctions.duration(bond, interest_rate),
                'Convexity': ql.BondFunctions.convexity(bond, interest_rate) / 100,
                'Dollar Convexity': (bond.NPV() / 100) * (ql.BondFunctions.convexity(bond, interest_rate) / 100),
            }
        
        return results

    def fixed_rate_amortizing(self, issue_date, maturity_date, tenor, coupon_rate, notionals):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issue_date = self._convert_to_quantlib_date(issue_date)
        maturity_date = self._convert_to_quantlib_date(maturity_date)
        
        results = {'Values': {'NPV': 'NPV', 'Accrued Interest': 'Accrued Interest',
                              'Price': 'Price', 'YTM': 'YTM', 'Duration': 'Duration',
                              'Dollar Duration': 'Dollar Duration', 'Convexity': 'Convexity',
                              'Dollar Convexity': 'Dollar Convexity'}}
        
        for shock in self.shocks:
            curve_handle = self._create_yield_curve(shock)
            schedule = ql.Schedule(
                issue_date, maturity_date, tenor, self.calendar, 
                ql.Following, ql.Following, ql.DateGeneration.Backward, False
            )
            bond = ql.AmortizingFixedRateBond(
                0, list(map(float, notionals.split(','))), schedule, [float(coupon_rate)], self.day_count, ql.Following, issue_date
            )
            bond.setPricingEngine(ql.DiscountingBondEngine(curve_handle))
            
            accrued_interest = bond.accruedAmount()
            
            ytm = bond.bondYield(self.day_count, self.compounding, self.compounding_frequency)
            interest_rate = ql.InterestRate(ytm, self.day_count, self.compounding, self.compounding_frequency)
            
            results[round(shock * 10000)] = {
                'NPV': bond.NPV(),
                'Accrued Interest': accrued_interest,
                'Price': bond.cleanPrice(),
                'YTM': ytm,
                'Duration': ql.BondFunctions.duration(bond, interest_rate),
                'Dollar Duration': (bond.NPV() / 100) * ql.BondFunctions.duration(bond, interest_rate),
                'Convexity': ql.BondFunctions.convexity(bond, interest_rate) / 100,
                'Dollar Convexity': (bond.NPV() / 100) * (ql.BondFunctions.convexity(bond, interest_rate) / 100),
            }
        
        return results

    def _build_zero_curve(self, rates, dayCount):
        return ql.ZeroCurve(self.spotDates, rates, dayCount, self.calendar, self.interpolation,
                            self.compounding, self.compoundingFrequency)

    def _build_yield_term_structure_handle(self, rates, dayCount):
        curve = self._build_zero_curve(rates, dayCount)
        return ql.YieldTermStructureHandle(curve)

    def _calculate_bond_metrics(self, bond, up_bond, dn_bond, npv, holding):
        price = bond.cleanPrice()
        up_price = up_bond.cleanPrice()
        dn_price = dn_bond.cleanPrice()
        
        up_npv = up_price * float(holding) / 100
        dn_npv = dn_price * float(holding) / 100
        
        duration = -1000 * ((up_npv - dn_npv) / (2 * npv))
        convexity = (dn_price + up_price - 2 * price) / ((price * self.epsilon) ** 2)
        
        return price, duration, convexity

    def price_floating(self, issueDate, maturityDate, tenor, spread, holding, dayCount, faceValue=100):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issueDate = self._convert_to_quantlib_date(issueDate)
        maturityDate = self._convert_to_quantlib_date(maturityDate)
        
        scenario_results = {'Values': {'NPV': 'NPV', 'Price': 'Price', 'YTM': 'YTM', 'Duration': 'Duration',
                                       'Dollar Duration': 'Dollar Duration', 'Convexity': 'Convexity',
                                       'Dollar Convexity': 'Dollar Convexity'}}
        
        shocks = [float(shock.strip()) for shock in self.shocks.split(',')]
        
        for shock in shocks:
            shockedRates = [rate + shock for rate in self.spotRates]
            up_shockedRates = [rate + shock + self.epsilon for rate in self.spotRates]
            dn_shockedRates = [rate + shock - self.epsilon for rate in self.spotRates]

            shockedCurveHandle = self._build_yield_term_structure_handle(shockedRates, dayCount)
            upCurveHandle = self._build_yield_term_structure_handle(up_shockedRates, dayCount)
            dnCurveHandle = self._build_yield_term_structure_handle(dn_shockedRates, dayCount)

            shockedIndexRates = [rate + shock for rate in self.indexRates]
            up_shockedIndexRates = [rate + shock + self.epsilon for rate in self.indexRates]
            dn_shockedIndexRates = [rate + shock - self.epsilon for rate in self.indexRates]

            shockedIndexHandle = self._build_yield_term_structure_handle(shockedIndexRates, dayCount)
            upIndexHandle = self._build_yield_term_structure_handle(up_shockedIndexRates, dayCount)
            dnIndexHandle = self._build_yield_term_structure_handle(dn_shockedIndexRates, dayCount)

            schedule = ql.Schedule(issueDate, maturityDate, tenor, self.calendar, ql.Following, ql.Following,
                                   ql.DateGeneration.Backward, False)

            fixingDays = 0
            index = ql.IborIndex("Name", ql.Period(ql.Monthly), fixingDays, self.currency, self.calendar,
                                 ql.ModifiedFollowing, True, dayCount, shockedIndexHandle)
            up_index = ql.IborIndex("Name", ql.Period(ql.Monthly), fixingDays, self.currency, self.calendar,
                                    ql.ModifiedFollowing, True, dayCount, upIndexHandle)
            dn_index = ql.IborIndex("Name", ql.Period(ql.Monthly), fixingDays, self.currency, self.calendar,
                                    ql.ModifiedFollowing, True, dayCount, dnIndexHandle)

            floatingRateBond = ql.FloatingRateBond(0, faceValue, schedule, index, dayCount, ql.Following,
                                                   fixingDays, [], [float(spread)], [], [], False, float(faceValue), issueDate)
            upFloatingRateBond = ql.FloatingRateBond(0, faceValue, schedule, up_index, dayCount, ql.Following,
                                                     fixingDays, [], [float(spread)], [], [], False, float(faceValue), issueDate)
            dnFloatingRateBond = ql.FloatingRateBond(0, faceValue, schedule, dn_index, dayCount, ql.Following,
                                                     fixingDays, [], [float(spread)], [], [], False, float(faceValue), issueDate)

            bondEngine = ql.DiscountingBondEngine(shockedCurveHandle)
            upBondEngine = ql.DiscountingBondEngine(upCurveHandle)
            dnBondEngine = ql.DiscountingBondEngine(dnCurveHandle)

            floatingRateBond.setPricingEngine(bondEngine)
            upFloatingRateBond.setPricingEngine(upBondEngine)
            dnFloatingRateBond.setPricingEngine(dnBondEngine)

            npv = floatingRateBond.cleanPrice() * float(holding) / 100
            ytm = floatingRateBond.bondYield(dayCount, self.compounding, self.compoundingFrequency)

            price, duration, convexity = self._calculate_bond_metrics(floatingRateBond, upFloatingRateBond,
                                                                      dnFloatingRateBond, npv, holding)

            scenario_results[round(shock * 10000)] = {'NPV': npv,
                                                      'Price': price,
                                                      'YTM': ytm,
                                                      'Duration': duration,
                                                      'Dollar Duration': (npv / 100) * duration,
                                                      'Convexity': convexity,
                                                      'Dollar Convexity': (npv / 100) * convexity}

        return scenario_results

    def price_amortizing_floating(self, shocks, issueDate, maturityDate, tenor, spread, notionals,
                                      notional_dates, dayCount):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issueDate = self._convert_to_quantlib_date(issueDate)
        maturityDate = self._convert_to_quantlib_date(maturityDate)
        
        valid_notional_dates = [d for d in notional_dates if d and isinstance(d, str) and len(d.split('-')) == 3]
        print("Valid Notional Dates:", valid_notional_dates)
        notional_dates = [self._convert_to_quantlib_date(d) for d in valid_notional_dates]

        # Align the notionals with the bond's schedule
        schedule = ql.Schedule(issueDate, maturityDate, tenor, self.calendar, ql.Following, ql.Following,
                               ql.DateGeneration.Backward, False)
        aligned_notionals = [notionals[0]]
        for i in range(1, len(schedule)):
            date = schedule[i]
            if date in notional_dates:
                index = notional_dates.index(date)
                aligned_notionals.append(notionals[index])
            else:
                aligned_notionals.append(aligned_notionals[-1])

        return self.price_floating(shocks, issueDate, maturityDate, tenor, spread, aligned_notionals,
                                       aligned_notionals[0], dayCount)
