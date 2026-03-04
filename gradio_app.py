"""
gradio_app.py — Interfaz Gráfica para ICT Simulator V2
Permite ingresar datos fácilmente y muestra resultados comparativos.
"""
import gradio as gr
import pandas as pd
from data_fetcher import fetch_ohlcv
from indicators import (calc_rsi, calc_macd, calc_ema, calc_bollinger,
                         calc_fibonacci_levels, find_multiple_fibonacci_tps,
                         calc_ewma_volatility, generate_indicator_signals)
from ict_analysis import (identify_swings, get_swing_sequence, detect_bos_choch,
                          detect_ict_reversal_signal, identify_fvg, identify_order_blocks,
                          identify_liquidity_pools, detect_liquidity_sweeps,
                          estimate_liquidation_zones, find_nearest_liquidation_magnets)
from market_regime import determine_directional_bias
from confluence import calculate_confluence_score
from montecarlo import simulate


def calculate_levels(entry, leverage, direction):
    """Lógica original de MainFuturos V2 para SL/TP simétrico/Liq"""
    if leverage == 1:
        if direction == 'LONG':
            sl = entry * 0.97
            tp = entry * 1.05
        else:
            sl = entry * 1.03
            tp = entry * 0.95
        return None, sl, tp
    
    margin = 1 / leverage
    if direction == 'LONG':
        liq = entry * (1 - margin)
        dist = entry - liq
        tp = entry + dist
    else:
        liq = entry * (1 + margin)
        dist = liq - entry
        tp = entry - dist
    return liq, liq, tp


def run_analysis(symbol, entry_price, timeframe, leverage, limit, sims, steps,
                 w_bos, w_zones, w_fib, w_sweeps, w_rsi, w_trend, w_rr, w_magnet):
    """Ejecuta el análisis V2 para AMBAS direcciones para mostrarlas en tabla"""
    
    weights_dict = {
        'bos': w_bos,
        'zones': w_zones,
        'fib': w_fib,
        'sweeps': w_sweeps,
        'rsi': w_rsi,
        'trend': w_trend,
        'rr': w_rr,
        'magnet': w_magnet
    }
    
    # 1. Fetch Df
    try:
        df = fetch_ohlcv(symbol, timeframe, limit)
    except Exception as e:
        return f"### Error descargando datos: {str(e)}", None

    last_price = df['close'].iloc[-1]
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_ema(df)
    df = calc_bollinger(df)
    df = calc_ewma_volatility(df)
    
    # 2. ICT Base
    df = identify_swings(df, window=10)
    swings = get_swing_sequence(df, lookback=150)
    bos_choch_trend, _, _ = detect_bos_choch(df, swings)
    
    bullish_fvgs, bearish_fvgs = identify_fvg(df)
    bullish_obs, bearish_obs = identify_order_blocks(df)
    rsi_val = df['rsi'].iloc[-1] if 'rsi' in df.columns else None
    
    directional_bias, _ = determine_directional_bias(df, bos_choch_trend, rsi_val)
    liq_zones = estimate_liquidation_zones(last_price)
    eq_h, eq_l = identify_liquidity_pools(df)
    sweeps = detect_liquidity_sweeps(df, eq_h, eq_l)
    fib_lvls = calc_fibonacci_levels(df['last_swing_high'].iloc[-1], df['last_swing_low'].iloc[-1])

    results = []
    messages = []
    
    market_type = "SPOT" if leverage == 1 else f"FUTURES {leverage}x"
    messages.append(f"### 📊 Resultados para {symbol} | {timeframe} | {market_type} | Entrada: ${entry_price}\n---")
    messages.append(f"**Estructura Macro (BOS/CHoCH):** {bos_choch_trend.upper()}")
    messages.append(f"**Sesgo de Régimen (Drift Base):** {directional_bias:.2f} (-1 Bajista, +1 Alcista)")

    # 3. Loop por Direccion para comparativa
    for DIR in ['LONG', 'SHORT']:
        reversal_strength = detect_ict_reversal_signal(
            entry_price, DIR, 
            bullish_fvgs, bearish_fvgs, bullish_obs, bearish_obs, 
            rsi_val, bos_choch_trend
        )
        
        liq_magnets = find_nearest_liquidation_magnets(entry_price, liq_zones, DIR) if leverage > 1 else []
        liq, sl, tp_sym = calculate_levels(entry_price, leverage, DIR)
        
        # Buscar TPs múltiples
        viable_tps = find_multiple_fibonacci_tps(df, entry_price, fib_lvls, DIR, max_tps=3)
        
        if not viable_tps:
            # Fallback si no hay niveles Fibonacci viables
            viable_tps = [{'level': tp_sym, 'ratio': 'Simétrico'}]
            
        indicators_sigs = generate_indicator_signals(df, DIR)
        dir_label = "🟢 LONG" if DIR == 'LONG' else "🔴 SHORT"

        for i, tp_data in enumerate(viable_tps):
            tp_final = tp_data['level']
            tp_name = f"Fib {tp_data['ratio']}" if tp_data['ratio'] != 'Simétrico' else 'Simétrico'
            row_label = f"{dir_label} (TP{i+1}: {tp_name})"
            
            score, details = calculate_confluence_score(
                entry_price, tp_final, sl, DIR,
                bullish_fvgs, bearish_fvgs, bullish_obs, bearish_obs,
                indicators_sigs, tp_final, sweeps, liq_magnets, bos_choch_trend,
                weights=weights_dict
            )
            
            mc = simulate(
                df, entry_price, tp_final, sl, DIR, score,
                directional_bias, reversal_strength,
                simulations=sims, steps=steps, timeframe=timeframe
            )
            
            pnltp = abs(tp_final - entry_price) / entry_price * leverage * 100
            pnlsl = abs(sl - entry_price) / entry_price * leverage * 100
            
            sl_label = f"${sl:.2f} (-{pnlsl:.1f}%)" if leverage == 1 else f"${sl:.2f} (Liq)"
            details_str = "\n".join(details)
            
            results.append({
                "Dirección / Escenario": row_label,
                "Prob (Éxito)": f"{mc['probability']:.2f}%",
                "Prob (SL)": f"{mc['fail_pct']:.2f}%",
                "Confluencia V2": f"{score}/100",
                "Desglose Confluencia": details_str,
                "Tiempo Estimado": mc['time_str'],
                "Take Profit": f"${tp_final:.2f} (+{pnltp:.1f}%)",
                "Stop Loss": sl_label,
                "Señal Reversión": f"{reversal_strength:.2f}",
                "Motor Drift": mc['drift_source']
            })

    df_results = pd.DataFrame(results)
    
    # Determinar ganador general (promediando escenarios o tomando el TP1)
    long_probs = [float(r["Prob (Éxito)"].strip('%')) for r in results if "LONG" in r["Dirección / Escenario"]]
    short_probs = [float(r["Prob (Éxito)"].strip('%')) for r in results if "SHORT" in r["Dirección / Escenario"]]
    
    avg_long = sum(long_probs) / len(long_probs) if long_probs else 0
    avg_short = sum(short_probs) / len(short_probs) if short_probs else 0
    
    if avg_long > avg_short + 5:
        messages.append(f"**💡 Veredicto Global:** El modelo favorece claramente al **🟢 LONG**")
    elif avg_short > avg_long + 5:
        messages.append(f"**💡 Veredicto Global:** El modelo favorece claramente al **🔴 SHORT**")
    else:
        messages.append(f"**💡 Veredicto Global:** Mercado **choppy/incierto**")
        
    return "\n".join(messages), df_results


# =====================================================================
# GUI GRADIO
# =====================================================================

css = """
body { font-family: 'Inter', sans-serif; }
table { width: 100% !important; border-collapse: collapse; table-layout: auto; }
th { background-color: #2b2b2b !important; color: #4ade80 !important; text-align: center !important; font-size: 14px !important; padding: 10px !important; white-space: normal !important; word-wrap: break-word !important; vertical-align: middle !important; }
td { text-align: left !important; font-size: 13px !important; padding: 8px !important; border-bottom: 1px solid #444 !important; white-space: pre-wrap !important; vertical-align: top !important; }
tr:hover { background-color: #333 !important; }
.gradio-container { background-color: #121212 !important; }
.gr-button-primary { background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%); border: none; font-weight: bold; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css, title="ICT Quant Simulator V2") as demo:
    gr.Markdown("# 📈 ICT Quant Futures & Spot Simulator V2")
    
    with gr.Accordion("📚 ¿Qué es ICT? Guía Rápida de Conceptos", open=False):
        gr.Markdown('''
        **Inner Circle Trader (ICT)** se basa en operar como lo hacen los algoritmos institucionales (Smart Money), cazando stop-losses ajenos y entrando en zonas clave, no usando medias móviles aleatorias.
        - **BOS (Break of Structure):** Confirmación de que la tendencia principal continúa a favor.
        - **CHoCH (Change of Character):** La primera señal fuerte de que el mercado se dio la vuelta (reversión).
        - **FVG (Fair Value Gap):** Ineficiencias o vacíos en el precio originados por inyecciones violentas de capital institucional. El precio vuelve al FVG como un imán para rebalancear órdenes.
        - **OB (Order Block):** La última vela bajista antes del gran impulso alcista (o viceversa). Son zonas de soporte/resistencia institucional de alta precisión (Golden Zones).
        - **Sweeps de Liquidez (Cacería de Stops):** Ocurren cuando el precio cae apenas bajo el mínimo anterior (sacando a los retail traders) sólo para luego dispararse hacia arriba con la liquidez robada.
        ''')
    
    gr.Markdown("Ingresa tus parámetros para analizar el mercado usando lógicas Bottom-Up (ICT Order Blocks/FVG) y Top-Down (Regime-Aware Monte Carlo).")
    
    with gr.Row():
        with gr.Column(scale=1):
            symbol_in = gr.Textbox(label="Activo (ej: ETH/USDT, BTC/USDT, AMZN)", value="ETH/USDT")
            price_in = gr.Number(label="Precio de Entrada ($)", value=1910)
            tf_in = gr.Dropdown(label="Temporalidad", choices=['15m', '1h', '4h', '1d'], value='1h')
            leverage_in = gr.Slider(label="Apalancamiento (1x = SPOT)", minimum=1, maximum=125, step=1, value=35)
            
            with gr.Accordion("⚙️ Configuración Avanzada / Simulador", open=False):
                limit_in = gr.Slider(label="Velas Históricas", minimum=500, maximum=10000, step=500, value=2000)
                sims_in = gr.Slider(label="Rutas Monte Carlo", minimum=1000, maximum=10000, step=1000, value=5000)
                steps_in = gr.Slider(label="Pasos al Futuro (Velas a Proyectar)", minimum=10, maximum=500, step=10, value=150)
                
            with gr.Accordion("🛠️ Seteos de Ponderación (Confluencia V2)", open=False):
                gr.Markdown("Ajusta el peso máximo (puntos) que entrega cada señal al sistema de confluencia (Max 100 recomendado):")
                w_bos = gr.Number(label="Estructura BOS/CHoCH", value=20)
                w_zones = gr.Number(label="Golden Z. (FVG+OB)", value=15)
                w_sweeps = gr.Number(label="Sweeps Escudos", value=15)
                w_fib = gr.Number(label="TP Fibonacci Validado", value=10)
                w_rsi = gr.Number(label="Alineación RSI", value=10)
                w_trend = gr.Number(label="Alineación Tendencias Clásicas", value=10)
                w_rr = gr.Number(label="Risk/Reward Bueno (>1.5)", value=10)
                w_magnet = gr.Number(label="Imán de Liquidez cerca", value=10)
                
            submit_btn = gr.Button("Analizar Mercado 🚀", variant="primary")
            
        with gr.Column(scale=2):
            markdown_out = gr.Markdown("Esperando ejecución...")
            table_out = gr.Dataframe(headers=[
                "Dirección / Escenario", "Prob (Éxito)", "Prob (SL)", 
                "Confluencia V2", "Desglose Confluencia", "Tiempo Estimado", 
                "Take Profit", "Stop Loss", "Señal Reversión", "Motor Drift"
            ], interactive=False, wrap=True)
            
            # Botones pre-armados para testing rapido
            gr.Markdown("### ⚡ Casos de Prueba Rápidos")
            with gr.Row():
                btn_eth_1h = gr.Button("ETH 1h (Caso Original)")
                btn_eth_4h = gr.Button("ETH 4h (Macro View)")
                btn_amzn_1d = gr.Button("AMZN Spot 1d")

    # Mapeo de eventos
    submit_btn.click(
        fn=run_analysis,
        inputs=[symbol_in, price_in, tf_in, leverage_in, limit_in, sims_in, steps_in,
                w_bos, w_zones, w_fib, w_sweeps, w_rsi, w_trend, w_rr, w_magnet],
        outputs=[markdown_out, table_out]
    )
    
    btn_eth_1h.click(lambda: ("ETH/USDT", 1910, "1h", 35), outputs=[symbol_in, price_in, tf_in, leverage_in])
    btn_eth_4h.click(lambda: ("ETH/USDT", 1910, "4h", 35), outputs=[symbol_in, price_in, tf_in, leverage_in])
    btn_amzn_1d.click(lambda: ("AMZN", 206, "1d", 1), outputs=[symbol_in, price_in, tf_in, leverage_in])

if __name__ == "__main__":
    demo.launch(inbrowser=True, quiet=True, server_port=7861)
