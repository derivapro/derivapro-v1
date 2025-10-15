# derivapro/models/schedule.py
import datetime as dt
from typing import List

def _add_months(d: dt.date, n: int) -> dt.date:
    y = d.year + (d.month - 1 + n)//12
    m = (d.month - 1 + n) % 12 + 1
    day = min(d.day, 28)
    return dt.date(y, m, day)

def build_schedule(start: dt.date, end: dt.date, pay_per_year: int) -> List[dt.date]:
    if pay_per_year not in (1,2,4): pay_per_year = 2
    months = 12 // pay_per_year
    dates, d = [], _add_months(start, months)
    while d <= end:
        dates.append(d)
        d = _add_months(d, months)
    if not dates or dates[-1] != end:
        dates.append(end)
    return dates
