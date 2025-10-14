# derivapro/models/vol_surface.py
import os,  pandas as pd
from dataclasses import dataclass

@dataclass
class VolSurface:
    df: pd.DataFrame          # columns: expiry (Y), tenor (Y), vol
    model: str = "black"      # "black" or "bachelier"

    @staticmethod
    def from_csv(path: str, col_e="expiry", col_t="tenor", col_v="vol", model="black"):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Vol surface file not found: {path}")
        df = pd.read_csv(path).rename(columns={col_e:"expiry", col_t:"tenor", col_v:"vol"})
        if not {"expiry","tenor","vol"}.issubset(df.columns):
            raise ValueError("Vol surface CSV must have columns: expiry, tenor, vol")
        df = df[["expiry","tenor","vol"]].dropna()
        return VolSurface(df.reset_index(drop=True), model=model)

    @staticmethod
    def from_file(file_obj, col_e="expiry", col_t="tenor", col_v="vol", model="black"):
        df = pd.read_csv(file_obj).rename(columns={col_e:"expiry", col_t:"tenor", col_v:"vol"})
        df = df[["expiry","tenor","vol"]].dropna()
        return VolSurface(df.reset_index(drop=True), model=model)

    def get(self, expiry_y: float, tenor_y: float) -> float:
        if self.df.empty:
            raise ValueError("Vol surface is empty.")
        d = ((self.df["expiry"]-expiry_y).abs() + (self.df["tenor"]-tenor_y).abs())
        idx = int(d.idxmin())
        return float(self.df.loc[idx, "vol"])
