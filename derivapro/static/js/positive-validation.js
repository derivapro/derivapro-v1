/*
 * Shared numeric-input validation for DerivaPro product pages.
 *
 * Product forms across the app accept economically impossible inputs (negative
 * strikes, spot prices, notionals, volatilities) which either produce nonsense
 * output or raise a raw 500 from the pricing library. Rather than hand-editing
 * the ~200 numeric inputs spread over the templates, this module classifies
 * every `input[type=number]` by its `name` and enforces the right bound.
 *
 * Behaviour is deliberately friendly: the browser's native validation bubble is
 * suppressed and replaced with an inline `.field-error` message next to the
 * offending field, telling the user what to enter instead of throwing an error.
 *
 * Escape hatches, in priority order:
 *   - `data-skip-validation` on an input   -> ignored entirely
 *   - `data-allow-negative="true"`         -> no lower bound applied
 *   - an author-supplied `min` attribute   -> respected as-is, never overwritten
 */
(function () {
    'use strict';

    // Strictly greater than zero: a zero value is as meaningless as a negative one.
    var POSITIVE = [
        'strike_price', 'market_strike', 'spot_price', 'settlement_price',
        'notional', 'nominal', 'structured_notional', 'pay_notional',
        'rec_notional', 'vega_notional', 'face_value', 'principal',
        'sigma', 'volatility', 'strike_vol', 'new_strike_vol',
        'num_steps', 'number_of_steps', 'num_paths', 'num_mc_steps',
        'num_mc_paths', 'mc_steps', 'steps', 'quantity', 'num_contracts',
        'multiplier', 'contract_multiplier', 'maturity', 'time_to_maturity',
        'structured_maturity', 'range_span', 'step_range', 'sens_step_range',
        'obs', 'payoff_points', 'target_standard_error'
    ];

    // Zero is a legitimate value, negatives are not.
    var NON_NEGATIVE = [
        'dividend_yield', 'q', 'structured_q', 'spread', 'pay_spread',
        'rec_spread', 'coupon_rate', 'pay_coupon_rate', 'rec_coupon_rate',
        'structured_coupon_rate', 'storage_cost', 'contract_fee',
        'loan_age', 'orig_ltv', 'orig_fico', 'random_seed', 'seed',
        'barrier', 'structured_protection_barrier', 'structured_coupon_barrier',
        'structured_autocall_barrier', 'payoff_floor', 'payoff_ceiling',
        'maintenance_margin_pct', 'tranche_lower_1', 'tranche_lower_2',
        'tranche_lower_3', 'tranche_upper_1', 'tranche_upper_2',
        'tranche_upper_3', 'kappa'
    ];

    // Probabilities and correlations, bounded to [0, 1].
    var UNIT_INTERVAL = ['recovery_rate', 'structured_correlation'];

    // Rates and shock sizes are genuinely allowed to be negative.
    var ALLOW_NEGATIVE = [
        'r', 'risk_free', 'risk_free_rate', 'structured_r', 'market_rate',
        'orig_rate', 'convenience_yield', 'theta', 'rho', 'price_change'
    ];

    function classify(name) {
        if (!name) return null;
        if (ALLOW_NEGATIVE.indexOf(name) !== -1) return null;
        if (/_shock$/.test(name) || /^shock/.test(name)) return null;
        if (UNIT_INTERVAL.indexOf(name) !== -1) return { min: 0, max: 1, strict: false };
        if (POSITIVE.indexOf(name) !== -1) return { min: 0, strict: true };
        if (NON_NEGATIVE.indexOf(name) !== -1) return { min: 0, strict: false };
        return null;
    }

    function labelFor(input) {
        var explicit = input.getAttribute('data-label');
        if (explicit) return explicit;

        var label = null;
        if (input.id) {
            label = document.querySelector('label[for="' + input.id + '"]');
        }
        if (!label && input.parentElement) {
            label = input.parentElement.querySelector('label');
        }
        if (label) {
            return label.textContent.replace(/[:*]\s*$/, '').trim();
        }
        return (input.name || 'This field')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
    }

    function messageFor(input, rule) {
        var label = labelFor(input);
        if (input.validity.valueMissing) {
            return 'Enter a value for ' + label + '.';
        }
        if (input.validity.badInput) {
            return 'Enter a valid number for ' + label + '.';
        }
        if (!rule) {
            return 'Enter a valid value for ' + label + '.';
        }
        if (rule.max !== undefined) {
            return 'Enter a value between ' + rule.min + ' and ' + rule.max
                + ' for ' + label + '.';
        }
        if (rule.strict) {
            return 'Enter a positive value for ' + label + '.';
        }
        return 'Enter a value of ' + rule.min + ' or greater for ' + label + '.';
    }

    function errorElementFor(input) {
        // Reuse a hand-written .field-error already present in the field group so
        // pages that ship their own message do not end up showing two.
        var group = input.closest('.form-group') || input.parentElement;
        if (!group) return null;

        var existing = group.querySelector('.field-error');
        if (existing) return existing;

        var created = document.createElement('div');
        created.className = 'field-error';
        created.setAttribute('data-auto-error', 'true');
        created.style.display = 'none';
        if (input.nextSibling) {
            input.parentNode.insertBefore(created, input.nextSibling);
        } else {
            input.parentNode.appendChild(created);
        }
        return created;
    }

    function showError(input, text) {
        var target = errorElementFor(input);
        if (!target) return;
        // Only overwrite messages this module owns; keep bespoke page copy intact.
        if (target.getAttribute('data-auto-error') === 'true' || !target.textContent.trim()) {
            target.textContent = text;
        }
        target.style.display = 'block';
        input.setAttribute('aria-invalid', 'true');
    }

    function clearError(input) {
        var group = input.closest('.form-group') || input.parentElement;
        if (!group) return;
        var target = group.querySelector('.field-error');
        if (target) target.style.display = 'none';
        input.removeAttribute('aria-invalid');
    }

    function applyRule(input) {
        if (input.hasAttribute('data-skip-validation')) return;
        if (input.type === 'hidden' || input.disabled) return;
        if (input.getAttribute('data-allow-negative') === 'true') return;

        var rule = classify(input.name);
        if (!rule) return;

        input.dataset.dpRule = JSON.stringify(rule);

        if (!input.hasAttribute('min')) {
            if (rule.strict) {
                // A strictly-positive field needs a min the browser can enforce.
                // Derive it from the declared step so integer counters get min=1
                // and continuous quantities get an arbitrarily small positive floor.
                var step = parseFloat(input.getAttribute('step'));
                input.setAttribute('min', (step && step >= 1) ? String(step) : '0.000001');
            } else {
                input.setAttribute('min', String(rule.min));
            }
        }
        if (rule.max !== undefined && !input.hasAttribute('max')) {
            input.setAttribute('max', String(rule.max));
        }

        input.addEventListener('invalid', function (event) {
            // Suppress the native bubble in favour of the inline message.
            event.preventDefault();
            showError(input, messageFor(input, rule));
        });

        input.addEventListener('input', function () {
            if (input.checkValidity()) clearError(input);
        });

        input.addEventListener('blur', function () {
            if (input.value === '') return;
            if (input.checkValidity()) {
                clearError(input);
            } else {
                showError(input, messageFor(input, rule));
            }
        });
    }

    /**
     * Validate a form and surface inline messages. Returns true when the form is
     * safe to submit. Exposed so the AJAX `submitForm()` helpers duplicated across
     * the product templates can gate their POSTs on it.
     */
    function validateForm(form) {
        if (typeof form === 'string') {
            form = document.getElementById(form);
        }
        if (!form || typeof form.checkValidity !== 'function') return true;

        // checkValidity() fires `invalid` on each offending control, which our
        // handlers turn into inline messages.
        var valid = form.checkValidity();
        if (!valid) {
            var firstInvalid = form.querySelector(':invalid');
            if (firstInvalid && typeof firstInvalid.focus === 'function') {
                firstInvalid.focus();
                if (typeof firstInvalid.scrollIntoView === 'function') {
                    firstInvalid.scrollIntoView({ block: 'center', behavior: 'smooth' });
                }
            }
        }
        return valid;
    }

    function scan(root) {
        (root || document).querySelectorAll('input[type="number"]').forEach(applyRule);
    }

    function init() {
        scan(document);

        // Product pages swap result panels in via innerHTML after AJAX posts, so
        // re-scan whenever new inputs appear.
        if (window.MutationObserver) {
            new MutationObserver(function (mutations) {
                mutations.forEach(function (mutation) {
                    mutation.addedNodes.forEach(function (node) {
                        if (node.nodeType !== 1) return;
                        if (node.matches && node.matches('input[type="number"]')) {
                            applyRule(node);
                        } else if (node.querySelectorAll) {
                            scan(node);
                        }
                    });
                });
            }).observe(document.body, { childList: true, subtree: true });
        }
    }

    window.validatePositiveInputs = validateForm;
    window.derivaProValidation = { validateForm: validateForm, scan: scan };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
