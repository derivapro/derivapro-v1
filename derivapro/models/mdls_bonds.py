import QuantLib as ql
import datetime
import logging
logger = logging.getLogger(__name__)

class NCFixedBonds:
    def __init__(self, value_date, spot_dates, spot_rates, shocks, day_count, calendar, interpolation,
                 compounding, compounding_frequency):
        self.value_date = self._convert_to_quantlib_date(value_date)
        ql.Settings.instance().evaluationDate = self.value_date
        self.spot_dates = [self._convert_to_quantlib_date(date.strip()) for date in spot_dates.split(',')]
        self.spot_rates = [float(rate.strip()) for rate in spot_rates.split(',')]
        self.shocks = [float(shock.strip()) for shock in shocks.split(',')]
        self.day_count = day_count
        self.calendar = calendar
        self.interpolation = interpolation
        self.compounding = compounding
        self.compounding_frequency = compounding_frequency
    
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
        # Flat-extrapolate beyond the last supplied pillar so a bond maturing
        # past the user's curve input (a common case - the curve is often
        # shorter than the bond) doesn't raise "time past max curve time"
        # instead of pricing.
        yield_curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(yield_curve)

    def fixed_rate(self, issue_date, maturity_date, tenor, coupon_rate, face_value):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issue_date = self._convert_to_quantlib_date(issue_date)
        maturity_date = self._convert_to_quantlib_date(maturity_date)
        
        results = {'Values': {'NPV': 'NPV', 'Price': 'Price', 'YTM': 'YTM', 'Duration': 'Duration',
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
            
            ytm = bond.bondYield(self.day_count, self.compounding, self.compounding_frequency)
            interest_rate = ql.InterestRate(ytm, self.day_count, self.compounding, self.compounding_frequency)
            
            results[round(shock * 10000)] = {
                'NPV': bond.NPV(),
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
        
        results = {'Values': {'NPV': 'NPV', 'Price': 'Price', 'YTM': 'YTM', 'Duration': 'Duration',
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
            
            ytm = bond.bondYield(self.day_count, self.compounding, self.compounding_frequency)
            interest_rate = ql.InterestRate(ytm, self.day_count, self.compounding, self.compounding_frequency)
            
            results[round(shock * 10000)] = {
                'NPV': bond.NPV(),
                'Price': bond.cleanPrice(),
                'YTM': ytm,
                'Duration': ql.BondFunctions.duration(bond, interest_rate),
                'Dollar Duration': (bond.NPV() / 100) * ql.BondFunctions.duration(bond, interest_rate),
                'Convexity': ql.BondFunctions.convexity(bond, interest_rate) / 100,
                'Dollar Convexity': (bond.NPV() / 100) * (ql.BondFunctions.convexity(bond, interest_rate) / 100),
            }
        
        return results


class NCFloatingBonds:
    def __init__(self, value_date, spotDates, spotRates, indexDates, indexRates, calendar, currency, interpolation,
                 compounding, compoundingFrequency, epsilon=0.001):
        self.value_date = self._convert_to_quantlib_date(value_date)
        ql.Settings.instance().evaluationDate = self.value_date
        self.spotDates = [self._convert_to_quantlib_date(date.strip()) for date in spotDates.split(',')]
        self.spotRates = [float(rate.strip()) for rate in spotRates.split(',')]
        self.indexDates = [self._convert_to_quantlib_date(date.strip()) for date in indexDates.split(',')]
        self.indexRates = [float(rate.strip()) for rate in indexRates.split(',')]
        self.calendar = calendar
        self.currency = currency
        self.interpolation = interpolation
        self.compounding = compounding
        self.compoundingFrequency = compoundingFrequency
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

    def _build_zero_curve(self, rates, dayCount, dates=None):
        """Build a zero curve from ``rates``.

        ``dates`` must be the pillar dates that belong to those rates. The index
        curve has its own pillars (``self.indexDates``), and silently reusing the
        discount-curve pillars raises an opaque ``new_ZeroCurve`` overload error
        whenever the two lists differ in length.
        """
        if dates is None:
            dates = self.spotDates
        if len(dates) != len(rates):
            raise ValueError(
                f"Curve build failed: {len(dates)} dates supplied for {len(rates)} rates. "
                "Provide one rate per date."
            )
        curve = ql.ZeroCurve(dates, rates, dayCount, self.calendar, self.interpolation,
                             self.compounding, self.compoundingFrequency)
        # Business-day adjustment pushes the final coupon a few days past the last
        # curve pillar, which otherwise aborts pricing with "time is past max
        # curve time". Flat extrapolation over that short stub is the market
        # convention and keeps a curve quoted to maturity usable.
        curve.enableExtrapolation()
        return curve

    def _build_yield_term_structure_handle(self, rates, dayCount, dates=None):
        curve = self._build_zero_curve(rates, dayCount, dates)
        return ql.YieldTermStructureHandle(curve)

    def _calculate_bond_metrics(self, bond, up_bond, dn_bond):
        """Shocked-price duration and convexity.

        Both statistics are ratios of prices, so the holding size cancels out and
        is not needed here.
        """
        price = bond.cleanPrice()
        up_price = up_bond.cleanPrice()
        dn_price = dn_bond.cleanPrice()

        duration = -1000 * ((up_price - dn_price) / (2 * price))
        convexity = (dn_price + up_price - 2 * price) / ((price * self.epsilon) ** 2)

        return price, duration, convexity

    def _build_floating_bond(self, schedule, index, dayCount, spread, issueDate, notionals):
        """Build a floating-rate bond, amortizing when more than one notional is given.

        ``notionals`` carries one notional per coupon period. QuantLib honours the
        stepped schedule natively through ``AmortizingFloatingRateBond``; a
        single-element list is the plain bullet case.
        """
        fixingDays = 0
        if len(notionals) == 1:
            faceValue = float(notionals[0])
            return ql.FloatingRateBond(0, faceValue, schedule, index, dayCount, ql.Following,
                                       fixingDays, [], [float(spread)], [], [], False,
                                       faceValue, issueDate)

        return ql.AmortizingFloatingRateBond(0, [float(n) for n in notionals], schedule, index,
                                             dayCount, ql.Following, fixingDays, [],
                                             [float(spread)], [], [], False, issueDate)

    def price_floating(self, shocks, issueDate, maturityDate, tenor, spread, holding, dayCount,
                       faceValue=100, notionals=None):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issueDate = self._convert_to_quantlib_date(issueDate)
        maturityDate = self._convert_to_quantlib_date(maturityDate)
        
        scenario_results = {'Values': {'NPV': 'NPV', 'Price': 'Price', 'YTM': 'YTM', 'Duration': 'Duration',
                                       'Dollar Duration': 'Dollar Duration', 'Convexity': 'Convexity',
                                       'Dollar Convexity': 'Dollar Convexity'}}
        
        shocks = [float(shock.strip()) for shock in shocks.split(',')]
        
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

            shockedIndexHandle = self._build_yield_term_structure_handle(shockedIndexRates, dayCount,
                                                                        self.indexDates)
            upIndexHandle = self._build_yield_term_structure_handle(up_shockedIndexRates, dayCount,
                                                                    self.indexDates)
            dnIndexHandle = self._build_yield_term_structure_handle(dn_shockedIndexRates, dayCount,
                                                                    self.indexDates)

            schedule = ql.Schedule(issueDate, maturityDate, tenor, self.calendar, ql.Following, ql.Following,
                                   ql.DateGeneration.Backward, False)

            fixingDays = 0
            index = ql.IborIndex("Name", ql.Period(ql.Monthly), fixingDays, self.currency, self.calendar,
                                 ql.ModifiedFollowing, True, dayCount, shockedIndexHandle)
            up_index = ql.IborIndex("Name", ql.Period(ql.Monthly), fixingDays, self.currency, self.calendar,
                                    ql.ModifiedFollowing, True, dayCount, upIndexHandle)
            dn_index = ql.IborIndex("Name", ql.Period(ql.Monthly), fixingDays, self.currency, self.calendar,
                                    ql.ModifiedFollowing, True, dayCount, dnIndexHandle)

            bondNotionals = notionals if notionals else [float(faceValue)]
            floatingRateBond = self._build_floating_bond(schedule, index, dayCount, spread,
                                                         issueDate, bondNotionals)
            upFloatingRateBond = self._build_floating_bond(schedule, up_index, dayCount, spread,
                                                           issueDate, bondNotionals)
            dnFloatingRateBond = self._build_floating_bond(schedule, dn_index, dayCount, spread,
                                                           issueDate, bondNotionals)

            bondEngine = ql.DiscountingBondEngine(shockedCurveHandle)
            upBondEngine = ql.DiscountingBondEngine(upCurveHandle)
            dnBondEngine = ql.DiscountingBondEngine(dnCurveHandle)

            floatingRateBond.setPricingEngine(bondEngine)
            upFloatingRateBond.setPricingEngine(upBondEngine)
            dnFloatingRateBond.setPricingEngine(dnBondEngine)

            if notionals:
                # The amortizing bond is built with real notionals, so its NPV is
                # already a currency amount and must not be rescaled by holding.
                npv = floatingRateBond.NPV()
            else:
                npv = floatingRateBond.cleanPrice() * float(holding) / 100
            ytm = floatingRateBond.bondYield(dayCount, self.compounding, self.compoundingFrequency)

            price, duration, convexity = self._calculate_bond_metrics(floatingRateBond, upFloatingRateBond,
                                                                      dnFloatingRateBond)

            scenario_results[round(shock * 10000)] = {'NPV': npv,
                                                      'Price': price,
                                                      'YTM': ytm,
                                                      'Duration': duration,
                                                      'Dollar Duration': (npv / 100) * duration,
                                                      'Convexity': convexity,
                                                      'Dollar Convexity': (npv / 100) * convexity}

        return scenario_results

    @staticmethod
    def _parse_amortization_schedule(notionals, notional_dates):
        """Normalise the amortization schedule into ``[(date_string, notional)]``.

        Accepts either comma-separated strings (what the web form posts) or ready
        made lists. Iterating a raw string character by character - which is what
        the previous implementation did - silently produced an empty schedule.
        """
        if isinstance(notionals, str):
            notionals = [part.strip() for part in notionals.split(',') if part.strip()]
        if isinstance(notional_dates, str):
            notional_dates = [part.strip() for part in notional_dates.split(',') if part.strip()]

        notionals = list(notionals)
        notional_dates = list(notional_dates)

        if not notionals:
            raise ValueError("Provide at least one notional.")
        if len(notionals) != len(notional_dates):
            raise ValueError(
                f"Provide one notional date per notional: got {len(notionals)} notionals "
                f"and {len(notional_dates)} dates."
            )

        parsed = []
        for raw_notional, raw_date in zip(notionals, notional_dates):
            try:
                amount = float(raw_notional)
            except (TypeError, ValueError):
                raise ValueError(f"'{raw_notional}' is not a valid notional amount.")
            if amount <= 0:
                raise ValueError("Enter a positive value for every notional.")
            parsed.append((raw_date, amount))

        return parsed

    def price_amortizing_floating(self, shocks, issueDate, maturityDate, tenor, spread, notionals,
                                      notional_dates, dayCount):
        # Ensure issue_date and maturity_date are in QuantLib Date format
        issueDate = self._convert_to_quantlib_date(issueDate)
        maturityDate = self._convert_to_quantlib_date(maturityDate)

        amortization = self._parse_amortization_schedule(notionals, notional_dates)
        steps = []
        for raw_date, amount in amortization:
            step_date = self._convert_to_quantlib_date(raw_date)
            if step_date < issueDate or step_date > maturityDate:
                raise ValueError(
                    f"Notional date {raw_date} falls outside the bond's life "
                    f"({issueDate.ISO()} to {maturityDate.ISO()})."
                )
            steps.append((step_date, amount))
        steps.sort(key=lambda step: step[0])
        logger.debug("Amortization steps: %s", [(d.ISO(), n) for d, n in steps])

        # Align the notionals with the bond's schedule: one notional per coupon
        # period, carrying forward the most recent step on or before the period
        # start. The old exact-date match meant a step that did not land on a
        # coupon date was dropped without warning.
        schedule = ql.Schedule(issueDate, maturityDate, tenor, self.calendar, ql.Following, ql.Following,
                               ql.DateGeneration.Backward, False)
        aligned_notionals = []
        for i in range(len(schedule) - 1):
            period_start = schedule[i]
            current = steps[0][1]
            for step_date, amount in steps:
                if step_date <= period_start:
                    current = amount
                else:
                    break
            aligned_notionals.append(current)

        if not aligned_notionals:
            raise ValueError("The issue and maturity dates produce no coupon periods.")

        return self.price_floating(shocks, issueDate, maturityDate, tenor, spread,
                                   holding=aligned_notionals[0], dayCount=dayCount,
                                   faceValue=aligned_notionals[0], notionals=aligned_notionals)
'''
test = NCFixedBonds('2024-11-24', '2024-11-24, 2027-11-24', 'spot_rates', shocks, day_count, calendar, interpolation, compounding, compounding_frequency)
'''