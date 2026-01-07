
def register_routes(app):
    from .index import index_bp
    from .vanilla_options import vanilla_options_bp
    from .exotic_options import exotic_options_bp
    from .futures_forwards import futures_forwards_bp
    from .swaps import swaps_bp
    from .swaptions import swaptions_bp
    from .volatility_derivatives import volatility_derivatives_bp
    from .credit_derivatives import credit_derivatives_bp
    from .equity_derivatives import equity_derivatives_bp
    from .bonds import nc_bonds_bp
    from .volatility_surface import volatility_surface_bp
    from .prepayment import prepayment_bp
    from .term_structure import term_structure_bp
    from .prepayment_v2 import prepayment_v2_bp
    from .rates import rates_bp
    from .rates_api import rates_api_bp
    from .swap_swaptions import swap_swaptions_bp
    from .extract_market_data import extract_market_data_bp
    from .reports import reports_generated_bp
    app.register_blueprint(index_bp, url_prefix='/')
    app.register_blueprint(vanilla_options_bp, url_prefix='/vanilla-options')
    app.register_blueprint(exotic_options_bp, url_prefix='/exotic-options')
    app.register_blueprint(futures_forwards_bp, url_prefix='/futures-forwards')
    app.register_blueprint(swaps_bp, url_prefix='/swaps')
    app.register_blueprint(swaptions_bp, url_prefix='/swaptions')
    app.register_blueprint(volatility_derivatives_bp, url_prefix='/volatility-derivatives')
    app.register_blueprint(credit_derivatives_bp, url_prefix='/credit-derivatives')
    app.register_blueprint(equity_derivatives_bp, url_prefix='/equity-derivatives')
    app.register_blueprint(nc_bonds_bp, url_prefix='/noncallable-bonds')
    # Register the volatility surface blueprint
    app.register_blueprint(volatility_surface_bp, url_prefix="/volatility_surface")
    app.register_blueprint(prepayment_bp, url_prefix="/prepayment")
    app.register_blueprint(term_structure_bp, url_prefix="/term-structure")
    app.register_blueprint(prepayment_v2_bp, url_prefix="/prepayment-v2")
    app.register_blueprint(rates_bp)        # /rates/swap, /rates/swaption
    app.register_blueprint(rates_api_bp)    # /api/rates/...
    app.register_blueprint(swap_swaptions_bp, url_prefix="/swap_swaptions")
    app.register_blueprint(extract_market_data_bp, url_prefix="/extract-market-data")
    app.register_blueprint(reports_generated_bp, url_prefix = "/reports-generated")

