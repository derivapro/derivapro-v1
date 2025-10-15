# derivapro/models/daycount.py
import datetime as dt

class DayCount:
    @staticmethod
    def year_frac(d0: dt.date, d1: dt.date, conv: str = "30/360") -> float:
        c = conv.upper()
        if c == "30/360":
            D0 = min(30, d0.day); D1 = min(30, d1.day)
            return ((d1.year - d0.year)*360 + (d1.month - d0.month)*30 + (D1 - D0)) / 360.0
        if c == "ACT/365":
            return (d1 - d0).days / 365.0
        # default ACT/360
        return (d1 - d0).days / 360.0
