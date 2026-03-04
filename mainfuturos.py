"""
mainfuturos.py (V2) — Orquestador avanzado ICT con Detección de Régimen y Predicciones Corregidas.
"""

from data_fetcher import fetch_ohlcv
from indicators import (calc_rsi, calc_macd, calc_ema, calc_bollinger,
                         calc_fibonacci_levels, find_best_fibonacci_tp,
                         calc_ewma_volatility, generate_indicator_signals)
from ict_analysis import (identify_swings, get_swing_sequence, detect_bos_choch,
                          detect_ict_reversal_signal, identify_fvg, identify_order_blocks,
                          identify_liquidity_pools, detect_liquidity_sweeps,
                          estimate_liquidation_zones, find_nearest_liquidation_magnets)
from market_regime import determine_directional_bias
from confluence import calculate_confluence_score
from montecarlo import simulate

# ============================================================
#  CONFIGURACIÓN V2
# ============================================================
SYMBOL        = 'AMZN'         # Par Crypto (ej: 'ETH/USDT') o Stock (ej: 'AMZN')
TIMEFRAME     = '1d'           # Temporalidad
ENTRY_PRICE   = 206            # Precio de entrada
LEVERAGE      = 1              # Apalancamiento (1 = SPOT)
DIRECTION     = 'LONG'         # 'LONG' o 'SHORT'
LIMIT         = 2000           # Velas
SIMULATIONS   = 5000           # Rutas Monte Carlo
STEPS         = 150            # Pasos al futuro
# ============================================================


def calculate_levels(entry, leverage, direction):
    """Calcula niveles forzando una relación Riesgo:Beneficio de 1:3."""
    if leverage == 1:
        # Modo SPOT 1:3 Ratio
        if direction == 'LONG':
            sl = entry * 0.97
            dist = entry - sl
            tp = entry + (dist * 3) # 1:3
        else:
            sl = entry * 1.03
            dist = sl - entry
            tp = entry - (dist * 3) # 1:3
        return None, sl, tp
    
    # Modo FUTUROS (Risk = Distancia al Liquidation, Reward = Riesgo x 3)
    margin = 1 / leverage
    if direction == 'LONG':
        liq = entry * (1 - margin)
        dist = entry - liq
        tp = entry + (dist * 3)
    else:
        liq = entry * (1 + margin)
        dist = liq - entry
        tp = entry - (dist * 3)
    return liq, liq, tp  # liq, sl, tp


if __name__ == '__main__':
    DIR = DIRECTION.upper()
    dir_emoji = '🟢' if DIR == 'LONG' else '🔴'
    
    market_type = "SPOT" if LEVERAGE == 1 else f"FUTURES {LEVERAGE}x"

    print(f"{'='*65}")
    print(f"  📊 ICT SIMULATOR — V2 (REGIME-AWARE)")
    print(f"  {dir_emoji} {DIR} {SYMBOL}  |  {TIMEFRAME}  |  {market_type}")
    print(f"{'='*65}")

    # 1. DATOS & INDICADORES V2
    print("\n⏳ Descargando datos y procesando indicadores O(N)...")
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, LIMIT)
    last_price = df['close'].iloc[-1]
    
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_ema(df)
    df = calc_bollinger(df)
    df = calc_ewma_volatility(df)
    
    # 2. ICT ESTRUCTURA (BOS/CHoCH)
    print("\n🔍 Analizando Estructura de Mercado ICT (BOS/CHoCH)...")
    df = identify_swings(df, window=10)
    swings = get_swing_sequence(df, lookback=150)
    bos_choch_trend, last_signal, all_signals = detect_bos_choch(df, swings)
    
    print(f"  Tendencia de Estructura: {bos_choch_trend.upper()}")
    if all_signals:
        for s in all_signals[-3:]:
            print(f"    {'✅' if s['direction']=='bullish' else '🔴'} {s['type']} ({s['direction']}): {s['detail']}")

    # 3. ZONAS Y SEÑAL DE REVERSIÓN ICT
    bullish_fvgs, bearish_fvgs = identify_fvg(df)
    bullish_obs, bearish_obs = identify_order_blocks(df)
    
    rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else None
    reversal_strength = detect_ict_reversal_signal(
        ENTRY_PRICE, DIR, 
        bullish_fvgs, bearish_fvgs, bullish_obs, bearish_obs, 
        rsi_val, bos_choch_trend
    )

    if reversal_strength > 0.3:
        print(f"\n  🚀 SEÑAL ICT DE REVERSIÓN ALCISTA DETECTADA (Fuerza: {reversal_strength:.2f})")
    elif reversal_strength < -0.3:
        print(f"\n  🩸 SEÑAL ICT DE REVERSIÓN BAJISTA DETECTADA (Fuerza: {reversal_strength:.2f})")

    # 4. RÉGIMEN DIRECCIONAL
    print("\n⚖️  Calculando Sesgo Direccional (Regime)...")
    directional_bias, bias_details = determine_directional_bias(df, bos_choch_trend, rsi_val)
    print(f"  Sesgo final: {directional_bias:.2f} (-1 bajista, +1 alcista)")
    for d in bias_details: print(d)

    # 5. LIQUIDEZ Y NIVELES
    liq_zones = estimate_liquidation_zones(last_price)
    
    # En Spot no consideramos imanes de liquidación como relevantes al precio de entrada, pero los imprimimos
    liq_magnets = find_nearest_liquidation_magnets(ENTRY_PRICE, liq_zones, DIR) if LEVERAGE > 1 else []
    
    eq_h, eq_l = identify_liquidity_pools(df)
    sweeps = detect_liquidity_sweeps(df, eq_h, eq_l)
    
    liq, sl, tp_sym = calculate_levels(ENTRY_PRICE, LEVERAGE, DIR)
    
    fib_lvls = calc_fibonacci_levels(df['last_swing_high'].iloc[-1], df['last_swing_low'].iloc[-1])
    fib_tp, fib_ratio, _ = find_best_fibonacci_tp(df, ENTRY_PRICE, fib_lvls, DIR)
    # FORZAR RATIO 3:1 IGNORANDO EL FIBONACCI TP DEMASIADO CERCANO
    tp_final = tp_sym
    
    indicators_sigs = generate_indicator_signals(df, DIR)

    # 6. CONFLUENCIA V2
    print(f"\n{'─'*65}")
    print("🎯 CONFLUENCIA V2 (Ponderada)")
    print(f"{'─'*65}")
    
    score, conf_det = calculate_confluence_score(
        ENTRY_PRICE, tp_final, sl, DIR,
        bullish_fvgs, bearish_fvgs, bullish_obs, bearish_obs,
        indicators_sigs, fib_tp, sweeps, liq_magnets, bos_choch_trend
    )
    for d in conf_det: print(d)
    print(f"\n  🎯 SCORE V2: {score}/100")

    # 7. MONTECARLO V2
    print(f"\n{'─'*65}")
    print(f"🎲 MONTECARLO V2 (Regime-Aware + EWMA Vol)")
    print(f"{'─'*65}")
    
    mc = simulate(
        df, ENTRY_PRICE, tp_final, sl, DIR, score,
        directional_bias, reversal_strength,
        simulations=SIMULATIONS, steps=STEPS, timeframe=TIMEFRAME
    )
    
    print(f"  Motor de Drift:  {mc['drift_source']}")
    print(f"  μ_adjusted V2:   {mc['mu_adjusted']:.6f}")
    print(f"  σ_ewma (vol):    {mc['sigma']:.6f}")
    print(f"\n  📈 Probabilidad ÉXITO ({DIR}): {mc['probability']:.1f}%")
    if LEVERAGE == 1:
        print(f"  📉 Probabilidad Tocar SL:  {mc['fail_pct']:.1f}%")
    else:
        print(f"  📉 Probabilidad SL/Liq:    {mc['fail_pct']:.1f}%")
    print(f"  ⏱️  Estimación Tiempo:      {mc['time_str']} (Mediana: {mc['median_time_str']})")

    # 8. RESUMEN FINAL
    pnltp = abs(tp_final - ENTRY_PRICE) / ENTRY_PRICE * LEVERAGE * 100
    pnlsl = abs(sl - ENTRY_PRICE) / ENTRY_PRICE * LEVERAGE * 100
    
    print(f"\n{'='*65}")
    print(f"  RESUMEN: {dir_emoji} {DIR} {SYMBOL} @ ${ENTRY_PRICE:.2f} | {market_type}")
    
    sl_label = "SL (-3%)" if LEVERAGE == 1 else "Liq"
    print(f"  TP: ${tp_final:.2f} (+{pnltp:.1f}%) | SL: ${sl:.2f} ({sl_label} -{pnlsl:.1f}%)")
    print(f"  PROBABILIDAD M.C.: {mc['probability']:.2f}% | CONFLUENCIA: {score}/100")
    print(f"{'='*65}")
