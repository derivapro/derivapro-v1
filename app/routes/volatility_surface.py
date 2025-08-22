from flask import Blueprint, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import yfinance as yf
import csv
from flask import Response

# Blueprint for Volatility Surface
volatility_surface_bp = Blueprint("volatility_surface", __name__)

# Function to fetch and process option chain
def fetch_option_chain(symbol, num_expirations=5):
    ticker = yf.Ticker(symbol)
    expirations = ticker.options[:num_expirations]
    
    if not expirations:
        raise ValueError("No options data found for the given symbol.")
    
    vol_data = []
    for exp in expirations:
        try:
            opt_chain = ticker.option_chain(exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            options = pd.concat([calls, puts])
            options = options[['strike', 'impliedVolatility']].dropna()
            options['expiration'] = exp
            vol_data.append(options)
        except Exception as e:
            print(f"Error fetching options for {exp}: {e}")
            continue
    
    if not vol_data:
        raise ValueError("No valid option data retrieved.")
    
    vol_df = pd.concat(vol_data, ignore_index=True)
    
    # Convert to numeric
    vol_df['impliedVolatility'] = pd.to_numeric(vol_df['impliedVolatility'], errors='coerce')
    vol_df['strike'] = pd.to_numeric(vol_df['strike'], errors='coerce')
    
    # Remove invalid values
    vol_df = vol_df.dropna()
    vol_df = vol_df[vol_df['impliedVolatility'] > 0]
    
    return vol_df

# Function to compute moneyness
def compute_moneyness(vol_df, symbol):
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="1d")
    if history.empty or 'Close' not in history.columns:
        raise ValueError("Could not retrieve stock price.")
    last_price = history['Close'].iloc[-1]
    
    vol_df['moneyness'] = vol_df['strike'] / last_price
    return vol_df

# Function to generate the implied volatility surface plot
def generate_volatility_surface(vol_df):
    vol_table = vol_df.pivot_table(index='moneyness', columns='expiration', values='impliedVolatility')

    if vol_table.empty:
        raise ValueError("Implied volatility data is empty after processing.")

    expiration_labels = vol_table.columns.astype(str)
    expiration_numeric = np.arange(len(expiration_labels))
    vol_table = vol_table.apply(pd.to_numeric, errors='coerce')

    for col in vol_table.columns:
        if vol_table[col].isnull().all():
            vol_table[col] = vol_table.mean().mean()
        else:
            vol_table[col] = vol_table[col].fillna(vol_table[col].mean())

    Z = np.nan_to_num(vol_table.values, nan=0)
    X, Y = np.meshgrid(expiration_numeric, vol_table.index)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xticks(expiration_numeric)
    ax.set_xticklabels(expiration_labels, rotation=45)
    ax.set_xlabel("Expiration Date")
    ax.set_ylabel("Moneyness")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Implied Volatility Surface")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close(fig)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return vol_table, plot_url

@volatility_surface_bp.route("/volatility_surface", methods=["GET", "POST"])
def volatility_surface():
    vol_table = None
    plot_url = None

    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper().strip()

        if not symbol:
            return jsonify({"error": "Please enter a valid stock symbol."})

        try:
            vol_df = fetch_option_chain(symbol)
            vol_df = compute_moneyness(vol_df, symbol)
            vol_table, plot_url = generate_volatility_surface(vol_df)
        except Exception as e:
            return jsonify({"error": str(e)})

    
    vol_table_html = vol_table.to_html(classes="table table-striped") if isinstance(vol_table, pd.DataFrame) and not vol_table.empty else None

    return render_template("volatility_surface.html", vol_table=vol_table_html, plot_url=plot_url)



@volatility_surface_bp.route("/download_vol_surface")
def download_vol_surface():
    symbol = request.args.get("symbol", "").upper().strip()

    if not symbol:
        return jsonify({"error": "Stock symbol is missing."})

    try:
        vol_df = fetch_option_chain(symbol)
        vol_df = compute_moneyness(vol_df, symbol)
        vol_table, _ = generate_volatility_surface(vol_df)  # Recompute vol_table

        # Convert DataFrame to CSV
        csv_data = vol_table.to_csv(index=True)

        # Create a response object with CSV data
        response = Response(csv_data, mimetype="text/csv")
        response.headers.set("Content-Disposition", f"attachment; filename=volatility_surface_{symbol}.csv")

        return response
    except Exception as e:
        return jsonify({"error": str(e)})

