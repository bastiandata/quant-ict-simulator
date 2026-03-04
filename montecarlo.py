"""
montecarlo.py (V2) — Monte Carlo consciente de régimen de mercado y volatilidad EWMA.
Evita el sesgo ciego del drift histórico usando el sesgo direccional y señales de reversión ICT.
"""
import numpy as np


TIMEFRAME_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
    '12h': 720, '1d': 1440,
}


def simulate(df, entry_price, tp, sl, direction, confluence_score,
             directional_bias, ict_reversal_strength,
             simulations=5000, steps=150, timeframe='1h'):
    """
    Simulación Monte Carlo (V2).
    En lugar de usar pct_change().mean() entero (que sesga si el mercado viene cayendo),
    usamos la volatilidad reciente (EWMA) y forzamos el drift (mu) basándonos en:
    1. directional_bias (tendencia macro BOS/EMA)
    2. ict_reversal_strength (señal local FVG/OB/RSI)
    3. confluence_score (fuerza general del setup)
    """
    
    # ── 1. Volatilidad Dinámica (EWMA) ──
    # Usar volatilidad reciente, no histórica completa
    if 'ewma_vol' in df.columns:
        sigma = df['ewma_vol'].iloc[-1]
    else:
        returns = df['close'].pct_change().dropna()
        sigma = returns.ewm(span=30).std().iloc[-1]
        
    # Salvar el valor absoluto de mu histórico estricto solo como magnitud base
    raw_returns = df['close'].pct_change().dropna()
    base_mu_mag = abs(raw_returns.mean()) * 2.0  # Multiplicador empírico para que el drift afecte
    if base_mu_mag < 0.0001: base_mu_mag = 0.0005 # Piso mínimo de drift
    
    # ── 2. Calcular Drift Ajustado (mu_adjusted) ──
    # El drift inicial viene del sesgo direccional macro (-1 a +1)
    macro_drift = directional_bias * base_mu_mag
    
    # Si hay una señal fuerte de reversión ICT, esta SOBREESCRIBE el macro_drift
    # (ej: mercado bajista (-1) pero toca Bullish FVG+OB+Oversold (+1))
    if abs(ict_reversal_strength) > 0.4:
        # Reversa agresivamente hacia la señal ICT
        mu_adjusted = ict_reversal_strength * base_mu_mag * 1.5
        drift_source = "Señal ICT de Reversión"
    else:
        # Sigue la tendencia macro, moderada por la confluencia
        mu_adjusted = macro_drift
        drift_source = "Tendencia Direccional (BOS + EMA)"
        
        # Si la confluencia es alta (>70), empujar levemente hacia 'direction'
        if confluence_score >= 70:
            if direction == 'LONG':
                mu_adjusted += (base_mu_mag * 0.5)
            else:
                mu_adjusted -= (base_mu_mag * 0.5)
            drift_source += " + Boost de Confluencia"
            
    # ── 3. Ejecutar Simulación ──
    success_count = 0
    fail_count = 0
    neutral_count = 0
    success_steps = []

    for _ in range(simulations):
        price = entry_price
        hit = False
        for step in range(1, steps + 1):
            shock = np.random.normal(loc=mu_adjusted, scale=sigma)
            price = price * (1 + shock)

            if direction == 'LONG':
                if price >= tp:
                    success_count += 1
                    success_steps.append(step)
                    hit = True
                    break
                elif price <= sl:
                    fail_count += 1
                    hit = True
                    break
            else:  # SHORT
                if price <= tp:
                    success_count += 1
                    success_steps.append(step)
                    hit = True
                    break
                elif price >= sl:
                    fail_count += 1
                    hit = True
                    break

        if not hit:
            neutral_count += 1

    prob_success = (success_count / simulations) * 100
    prob_fail = (fail_count / simulations) * 100

    # ── 4. Tiempo Estimado ──
    mins_per_step = TIMEFRAME_MINUTES.get(timeframe, 60)
    avg_steps = np.mean(success_steps) if success_steps else 0
    med_steps = np.median(success_steps) if success_steps else 0
    
    return {
        'probability': prob_success,
        'fail_pct': prob_fail,
        'neutral_pct': (neutral_count / simulations) * 100,
        'mu_adjusted': mu_adjusted,
        'sigma': sigma,
        'drift_source': drift_source,
        'avg_steps_to_tp': avg_steps,
        'median_steps_to_tp': med_steps,
        'time_str': _format_time(avg_steps * mins_per_step),
        'median_time_str': _format_time(med_steps * mins_per_step)
    }


def _format_time(total_minutes):
    if np.isnan(total_minutes) or total_minutes == 0: return "N/A"
    if total_minutes < 60: return f"{total_minutes:.0f} mins"
    hours = total_minutes / 60
    if hours < 24:
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h}h {m}m" if m > 0 else f"{h}h"
    days = hours / 24
    d = int(days)
    h = int((days - d) * 24)
    return f"{d}d {h}h" if h > 0 else f"{d}d"
