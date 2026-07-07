from io import BytesIO, StringIO
import csv
from typing import List, Dict

from openpyxl import Workbook


def dicts_to_xlsx_bytes(rows: List[Dict], sheet_name: str = "Sheet1") -> BytesIO:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not rows:
        f = BytesIO()
        wb.save(f)
        f.seek(0)
        return f

    headers = list(rows[0].keys())
    ws.append(headers)

    for r in rows:
        ws.append([r.get(h) for h in headers])

    f = BytesIO()
    wb.save(f)
    f.seek(0)
    return f


def dicts_to_csv_bytes(rows: List[Dict]) -> BytesIO:
    si = StringIO()
    if not rows:
        si.write("")
        b = BytesIO(si.getvalue().encode("utf-8"))
        b.seek(0)
        return b

    headers = list(rows[0].keys())
    writer = csv.DictWriter(si, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in headers})

    b = BytesIO(si.getvalue().encode("utf-8"))
    b.seek(0)
    return b
