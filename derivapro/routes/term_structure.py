# derivapro/routes/main_routes.py
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from flask import send_file
from fredapi import Fred
from ..models.mdls_term_structure import YieldTermStructure
from ..models.yieldterm_market_data import TreasuryRateProvider, SOFRRateProvider, FREDSwapRatesProvider
import QuantLib as ql
import uuid
import os
import matplotlib.pyplot as plt
from flask import session, current_app
import markdown
import numpy as np

term_structure_bp = Blueprint('term_structure', __name__)

API_KEY = 'a7a1a9c282ee0093003008999c337857'

@term_structure_bp.route('/calculate-term-structure', methods=['GET', 'POST'])

def calculate_term_structure():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'term_structure.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    form_data = {}
    plot = r2_ns_zero = r2_ns_df = r2_ns_fwd = r2_sv_zero = r2_sv_df = r2_sv_fwd = plot1 = r2_ns_zero1 = r2_ns_df1 = r2_ns_fwd1 = r2_sv_zero1 = r2_sv_df1 = r2_sv_fwd1 = None

    if request.method == 'POST':
        action = request.form.get('analysis_type')

        form_data = {
            'start_date': request.form['start_date'],
            'forward_tenor_str': request.form['forward_tenor_str'],
            'method_name': request.form['method_name'],
            'fit_selection': request.form['fit_selection'],
        }
        start_date_str = form_data['start_date'] 
        year, month, day = map(int, start_date_str.split('-'))
        start_date = ql.Date(day, month, year)

        # start_date = form_data['start_date']
        forward_tenor_str = form_data['forward_tenor_str']
        method_name = form_data['method_name']
        fit_selection = form_data['fit_selection']
        
        #day, month, year = map(int, start_date_str.split('-'))
        #start_date = ql.Date(day, month, year)
        forward_tenor = ql.Period(forward_tenor_str)

        # Initialize rate providers
        treasury_provider = TreasuryRateProvider(API_KEY)
        sofr_provider = SOFRRateProvider()
        swap_provider = FREDSwapRatesProvider(API_KEY)

        # Get market rates
        treasury_rates = treasury_provider.get_market_rates(start_date=start_date)
        sofr_rates = sofr_provider.get_market_rates(startDate=start_date)
        swap_rates = swap_provider.get_market_rates(start_date=start_date)

        if action == 'parallel':
            try:
                #start_date_str = form_data['start_date']
                #year, month, day = map(int, start_date_str.split('-'))
                #start_date = ql.Date(day, month, year)
                parallelShockValue = request.form.get('parallelShockValue')

                if parallelShockValue == '+50':
                    print('+50 read')
                    shocked_treasury_rates = [(tenor, rate + 0.005) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate + 0.005) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate + 0.005) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '+100':
                    print('+100 read')
                    shocked_treasury_rates = [(tenor, rate + 0.01) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate + 0.01) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate + 0.01) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '+150':
                    print('+150 read')
                    shocked_treasury_rates = [(tenor, rate + 0.015) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate + 0.015) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate + 0.015) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '+200':
                    print('+200 read')
                    shocked_treasury_rates = [(tenor, rate + 0.02) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate + 0.02) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate + 0.02) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '+250':
                    print('+250 read')
                    shocked_treasury_rates = [(tenor, rate + 0.025) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate + 0.025) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate + 0.025) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '+300':
                    print('+300 read')
                    shocked_treasury_rates = [(tenor, rate + 0.03) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate + 0.03) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate + 0.03) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '-50':
                    print('-50 read')
                    shocked_treasury_rates = [(tenor, rate - 0.005) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate - 0.005) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate - 0.005) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '-100':
                    print('-100 read')
                    shocked_treasury_rates = [(tenor, rate - 0.01) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate - 0.01) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate - 0.01) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '-150':
                    print('-150 read')
                    shocked_treasury_rates = [(tenor, rate - 0.015) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate - 0.015) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate - 0.015) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '-200':
                    print('-200 read')
                    shocked_treasury_rates = [(tenor, rate - 0.02) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate - 0.02) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate - 0.02) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '-250':
                    print('-250 read')
                    shocked_treasury_rates = [(tenor, rate - 0.025) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate - 0.025) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate - 0.025) for tenor, rate in swap_rates] if swap_rates else []
                elif parallelShockValue == '-300':
                    print('-300 read')
                    shocked_treasury_rates = [(tenor, rate - 0.03) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, rate - 0.03) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, rate - 0.03) for tenor, rate in swap_rates] if swap_rates else []
                else:
                    print('no shock read')
                    shocked_treasury_rates = treasury_rates
                    shocked_sofr_rates = sofr_rates
                    shocked_swap_rates = swap_rates
                
                # Initialize YieldTermStructure and append rates
                ts_shocked = YieldTermStructure()
                ts_shocked.append_market_rates(shocked_treasury_rates, source="treasury")
                ts_shocked.append_market_rates(shocked_sofr_rates, source="sofr")
                ts_shocked.append_market_rates(shocked_swap_rates, source="swap")

                ts_shocked.average_duplicate_rates(start_date)

                # 1. Create filename first
                safe_parallelShockValue = parallelShockValue.replace('+', 'plus').replace('-', 'minus') if parallelShockValue else "none"
                plot1_filename = f'shocked_yield_curve_plot_{safe_parallelShockValue}.png'
                #plot1_filename = f'shocked_yield_curve_plot_{parallelShockValue if parallelShockValue else "none"}.png'

                # 2. Build full path
                plot1_path = os.path.join(current_app.root_path, 'static', 'plots', plot1_filename)
                os.makedirs(os.path.dirname(plot1_path), exist_ok=True)

                # 3. Generate plot and save once
                fig1 = ts_shocked.yield_curve(start_date, fit_selection, method_name, forward_tenor)
                fig1.savefig(plot1_path)
                plt.close(fig1)
                # session['shocked_yield_curve_plot'] = {'filename': plot1_filename}
                session['shocked_yield_curve_plot_filename'] = plot1_filename

                # Bootstrap the curve
                curve1 = ts_shocked.bootstrap_curve(start_date, method_name)

                print('success1')

                # Conditionally calculate fit stats only if fit_selection is "yes"
                if fit_selection.lower() == "yes":
                    _, _, _, r2_zero_ns1, r2_df_ns1, r2_fwd_ns1 = ts_shocked.fit_nelson_siegel_curve(start_date, curve1)
                    _, _, _, r2_zero_sv1, r2_df_sv1, r2_fwd_sv1 = ts_shocked.fit_svensson_curve(start_date, curve1)
                    print('success2')
                    print("Plot filename sent to template:", plot1_filename)
                    print("File exists:", os.path.exists(os.path.join(current_app.root_path, 'static', 'plots', plot1_filename)))

                    return render_template(
                        'term_structure.html',
                        form_data=form_data,
                        plot=True,
                        md_content=md_content,
                        #start_date1=start_date_str,
                        #forward_tenor1=forward_tenor_str,
                        #method_name1=method_name,
                        #fit_selection1=fit_selection,
                        r2_ns_zero=r2_zero_ns1,
                        r2_ns_df=r2_df_ns1,
                        r2_ns_fwd=r2_fwd_ns1,
                        r2_sv_zero=r2_zero_sv1,
                        r2_sv_df=r2_df_sv1,
                        r2_sv_fwd=r2_fwd_sv1
                    )
                else:
                    print('success3')
                    # When fit_selection is "no", don't pass R2 values
                    return render_template(
                        'term_structure.html',
                        form_data=form_data,
                        plot=True,
                        md_content=md_content
                        #start_date1=start_date_str,
                        #forward_tenor1=forward_tenor_str,
                        #method_name1=method_name,
                        #fit_selection1=fit_selection
                    )
            except Exception as e:
                print('break1', e)
                return render_template(
                    'term_structure.html', 
                    form_data=form_data, 
                    plot=True, 
                    md_content=md_content,
                    error=str(e))
        
        elif action == 'non-parallel':
            try:
                #start_date_str = form_data['start_date']
                #year, month, day = map(int, start_date_str.split('-'))
                #start_date = ql.Date(day, month, year)
                nonParallelShockValue = request.form.get('nonParallelShockValue')
                nonParallelAbsoluteShock = request.form.get('nonParallelAbsoluteShock')
                tenor_translation = {
                        "1D": 1/365,
                        "1M": 1/12,
                        "3M": 3/12,
                        "6M": 6/12,
                        "1Y": 1,
                        "2Y": 2,
                        "3Y": 3,
                        "4Y": 4,
                        "5Y": 5,
                        "7Y": 7,
                        "10Y": 10,
                        "20Y": 20,
                        "30Y": 30
                    }
                def apply_steepener_shock(tenor, rate):
                    tenor_value = tenor_translation[str(tenor)]
                    shock = (-0.65 * int(nonParallelAbsoluteShock) * np.exp(-tenor_value / 4)) + (0.9 * int(nonParallelAbsoluteShock) * (1 - np.exp(-tenor_value / 4)))
                    rate = rate + shock / 10000
                    return rate
                
                def apply_flattener_shock(tenor, rate):
                    tenor_value = tenor_translation[str(tenor)]
                    shock = (0.8 * int(nonParallelAbsoluteShock) * np.exp(-tenor_value / 4)) - (0.6 * int(nonParallelAbsoluteShock) * (1 - np.exp(-tenor_value / 4)))
                    rate = rate + shock / 10000
                    return rate
                
                if nonParallelShockValue == 'Steepener':
                    print('steepener read')
                    shocked_treasury_rates = [(tenor, apply_steepener_shock(tenor, rate)) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, apply_steepener_shock(tenor, rate)) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, apply_steepener_shock(tenor, rate)) for tenor, rate in swap_rates] if swap_rates else []
                elif nonParallelShockValue == 'Flattener':
                    print('flattener read')
                    shocked_treasury_rates = [(tenor, apply_flattener_shock(tenor, rate)) for tenor, rate in treasury_rates]
                    shocked_sofr_rates = [(tenor, apply_flattener_shock(tenor, rate)) for tenor, rate in sofr_rates]
                    shocked_swap_rates = [(tenor, apply_flattener_shock(tenor, rate)) for tenor, rate in swap_rates] if swap_rates else []
                else:
                    print('no shock read')
                    shocked_treasury_rates = treasury_rates
                    shocked_sofr_rates = sofr_rates
                    shocked_swap_rates = swap_rates
                
                # Initialize YieldTermStructure and append rates
                ts_shocked = YieldTermStructure()
                ts_shocked.append_market_rates(shocked_treasury_rates, source="treasury")
                ts_shocked.append_market_rates(shocked_sofr_rates, source="sofr")
                ts_shocked.append_market_rates(shocked_swap_rates, source="swap")

                ts_shocked.average_duplicate_rates(start_date)
                #Create a replica of this in here (average duplicate rates worked for baseline)

                # 1. Create filename first
                safe_nonParallelShockValue = nonParallelShockValue if nonParallelShockValue else "none"
                plot2_filename = f'shocked_yield_curve_plot_{safe_nonParallelShockValue}.png'
                #plot1_filename = f'shocked_yield_curve_plot_{parallelShockValue if parallelShockValue else "none"}.png'

                # 2. Build full path
                plot2_path = os.path.join(current_app.root_path, 'static', 'plots', plot2_filename)
                os.makedirs(os.path.dirname(plot2_path), exist_ok=True)

                # 3. Generate plot and save once
                print('Early Checkpoint')
                fig2 = ts_shocked.yield_curve(start_date, fit_selection, method_name, forward_tenor)
                print('Checkpoint')
                fig2.savefig(plot2_path)
                plt.close(fig2)
                # session['shocked_yield_curve_plot'] = {'filename': plot1_filename}
                session['shocked_yield_curve_plot_filename'] = plot2_filename

                # Bootstrap the curve
                curve2 = ts_shocked.bootstrap_curve(start_date, method_name)

                print('success1')

                # Conditionally calculate fit stats only if fit_selection is "yes"
                if fit_selection.lower() == "yes":
                    _, _, _, r2_zero_ns2, r2_df_ns2, r2_fwd_ns2 = ts_shocked.fit_nelson_siegel_curve(start_date, curve2)
                    _, _, _, r2_zero_sv2, r2_df_sv2, r2_fwd_sv2 = ts_shocked.fit_svensson_curve(start_date, curve2)
                    print('success2')
                    print("Plot filename sent to template:", plot2_filename)
                    print("File exists:", os.path.exists(os.path.join(current_app.root_path, 'static', 'plots', plot2_filename)))

                    return render_template(
                        'term_structure.html',
                        form_data=form_data,
                        plot=True,
                        md_content=md_content,
                        #start_date1=start_date_str,
                        #forward_tenor1=forward_tenor_str,
                        #method_name1=method_name,
                        #fit_selection1=fit_selection,
                        r2_ns_zero=r2_zero_ns2,
                        r2_ns_df=r2_df_ns2,
                        r2_ns_fwd=r2_fwd_ns2,
                        r2_sv_zero=r2_zero_sv2,
                        r2_sv_df=r2_df_sv2,
                        r2_sv_fwd=r2_fwd_sv2
                    )
                else:
                    print('success3')
                    # When fit_selection is "no", don't pass R2 values
                    return render_template(
                        'term_structure.html',
                        form_data=form_data,
                        plot=True,
                        md_content=md_content,
                        #start_date1=start_date_str,
                        #forward_tenor1=forward_tenor_str,
                        #method_name1=method_name,
                        #fit_selection1=fit_selection
                    )
            except Exception as e:
                print('break2', e)
                return render_template(
                    'term_structure.html', 
                    form_data=form_data, 
                    plot=True,
                    md_content=md_content,
                    error=str(e))

        else:
            try:
                # Initialize YieldTermStructure and append rates
                print('plot = true')
                ts = YieldTermStructure()
                ts.append_market_rates(treasury_rates, source="treasury")
                ts.append_market_rates(sofr_rates, source="sofr")
                ts.append_market_rates(swap_rates, source="swap")

                ts.average_duplicate_rates(start_date)

                # Generate and save the plot
                fig = ts.yield_curve(start_date, fit_selection, method_name, forward_tenor)
                plot_filename = f'yield_curve_plot_{uuid.uuid4().hex}.png'
                plot_path = os.path.join(current_app.root_path, 'static', 'plots', plot_filename)
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                fig.savefig(plot_path)
                plt.close(fig)

                # Save plot filename in session
                session['yield_curve_plot'] = {'filename': plot_filename}

                # Bootstrap the curve
                curve = ts.bootstrap_curve(start_date, method_name)

                # Conditionally calculate fit stats only if fit_selection is "yes"
                if fit_selection.lower() == "yes":
                    print('fit selection = yes')
                    _, _, _, r2_zero_ns, r2_df_ns, r2_fwd_ns = ts.fit_nelson_siegel_curve(start_date, curve)
                    _, _, _, r2_zero_sv, r2_df_sv, r2_fwd_sv = ts.fit_svensson_curve(start_date, curve)

                    return render_template(
                        'term_structure.html',
                        form_data=form_data,
                        plot=True,
                        md_content=md_content,
                        #start_date=start_date_str,
                        #forward_tenor=forward_tenor_str,
                        #method_name=method_name,
                        #fit_selection=fit_selection,
                        r2_ns_zero=r2_zero_ns,
                        r2_ns_df=r2_df_ns,
                        r2_ns_fwd=r2_fwd_ns,
                        r2_sv_zero=r2_zero_sv,
                        r2_sv_df=r2_df_sv,
                        r2_sv_fwd=r2_fwd_sv
                    )
                else:
                    # When fit_selection is "no", don't pass R2 values
                    print('fit selection = no')
                    return render_template(
                        'term_structure.html',
                        form_data=form_data,
                        plot=True,
                        md_content=md_content
                        #start_date1=start_date_str,
                        #forward_tenor1=forward_tenor_str,
                        #method_name1=method_name,
                        #fit_selection1=fit_selection
                    )
            except Exception as e:
                print('break3', e)
                return render_template(
                    'term_structure.html', 
                    form_data=form_data, 
                    plot=None,
                    md_content=md_content,
                    error=str(e))

    else:
        # Defaults for form on GET
        return render_template(
            'term_structure.html',
            form_data=form_data,
            plot=None,
            md_content=md_content,
            forward_tenor='3M',
            method_name='PiecewiseFlatForward',
            fit_selection='yes'
        )