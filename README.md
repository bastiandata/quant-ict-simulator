# 📈 ICT Quant Trading Simulator (V2)

A Python-based quantitative trading simulator designed to analyze financial markets (Cryptocurrency Futures & Traditional Stock/SPOT Markets) using a hybrid approach: **Institutional Order Flow (Smart Money Concepts / ICT)** combined with **Stochastic Probability Models (Regime-Aware Monte Carlo Simulations)**.

---

## 🚀 Overview

This project was developed to bridge the gap between technical execution and mathematical probabilities. As a Commercial Engineer/Economist, the objective was to not just find "trading signals", but to model risk, estimate time-horizons, and quantify the probability of success using statistical methods.

### Key Features
- **Bottom-Up Analysis (ICT / Smart Money Concepts):** 
  - Mathematical detection of Break of Structure (BOS) and Change of Character (CHoCH).
  - Identification of Fair Value Gaps (FVGs) and Order Blocks (Golden Zones).
  - Aggregation of Liquidity Pools and detection of institutional Stop-Loss hunting (Sweeps).
- **Top-Down Quantitative Modeling (Monte Carlo):**
  - **Regime-Aware Drift:** Unlike standard Monte Carlo simulations that strictly use historical variance, this model adjusts the mathematical *drift* based on the current market regime (Bullish/Bearish Bias) and penalizes/rewards the drift based on the strength of the ICT Reversal Signals.
  - Dynamically calculates probability of success vs likelihood of hitting Stop-Loss/Liquidation.
- **Dynamic Risk Management:** 
  - Supports Leverage Futures (Liquidations) and SPOT trading (Fixed % Risk). 
  - Enforces Risk/Reward modeling and calculates multiple Take-Profit scenarios based on historical Fibonacci extensions.
- **Cross-Market Capability:** Integrates with `ccxt` (Binance API) for Crypto and `yfinance` (Yahoo Finance) for Stocks and Indices.
- **Gradio Web Interface:** A sleek, interactive dashboard to run simulations in real-time and visualize comparative tables (LONG vs SHORT).

---

## 🧠 Architectural Modules

The V2 architecture is modularized for scalability and maintainability:

1. `mainfuturos.py` / `gradio_app.py`: The Main Orchestrators (CLI and Web Interface).
2. `market_regime.py`: Evaluates the Macro Directional Bias (-1.0 to +1.0) using structural pivots.
3. `ict_analysis.py`: Core logic for detecting Institutional concepts (FVG, OB, Sweeps, BOS/CHoCH).
4. `montecarlo.py`: The stochastic simulator overriding historical drift with technical regime inputs.
5. `confluence.py`: A customizable weighted scoring system that merges all technical variables into a unified 0-100 probability score.
6. `indicators.py`: Traditional technical analysis and Volatility modeling (VWAP, ATR, EWMA, RSI, MACD).
7. `data_fetcher.py`: Asynchronous multi-exchange data ingestion.

---

## 📊 Example: The UI Dashboard
The project includes a local Web Application built with `Gradio`. 
It processes thousands of historical candles and simulates up to 10,000 future paths in seconds, returning a side-by-side comparison of Long vs Short outcomes, projected time horizons, and a breakdown of the Confluence Score.

### Installation & Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/quant-ict-simulator.git
   cd quant-ict-simulator/V2
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Web Interface:
   ```bash
   python gradio_app.py
   ```
4. Open the provided localhost link (e.g., `http://127.0.0.1:7861`) in your browser.

---

## 💼 Why this matters (Economic & Quant Perspective)
In quantitative finance, retail traders often fail due to psychological biases and poor risk management. This tool forces a probabilistic mindset. By defining trading opportunities strictly through the lens of expected value and statistical recurrence (Monte Carlo), it treats the financial markets not as a casino, but as a system of manageable risks and measurable edges.

<img width="1725" height="821" alt="image" src="https://github.com/user-attachments/assets/74d7906a-3588-42e5-8699-f11a2e33eaa0" />


***Note:** This software is for educational and simulation purposes. It does not constitute financial advice.*
