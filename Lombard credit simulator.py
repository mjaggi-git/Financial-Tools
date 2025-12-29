import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# Deaktiviere Warnungen für saubere Ausgabe
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. KONFIGURATION
# ==============================================================================
# Eingaben basierend auf den vom Benutzer bereitgestellten Werten, 
# aber korrigiert für Floats und Potenzierung.

# Hinweis: Die Eingabe über input() wurde entfernt, um das Skript 
# direkt ausführbar und reproduzierbar zu machen.

# *Kredit- und Basisparameter*
SUM_LOMB = 50000.0          # Lombardkreditbetrag
INTEREST_LOMB = 0.04        # 4% Jahreszins (als Float)
DURATION_LOMB = 5           # Dauer in Jahren
CURRENT_PORTFOLIO = 50000.0 # Eigenkapital
TOTAL_INVESTED = SUM_LOMB + CURRENT_PORTFOLIO

# *Risiko-Parameter*
CHANCE_BEING_FIRED_ANNUAL = 0.05 # 5% jährliches Risiko für Jobverlust
YEARLY_VOLATILITY = 0.15    # Jährliche Volatilität (Standardabweichung)
MAINTENANCE_MARGIN = 0.35   # 35% Maintenance Margin (vom Portfolio)

# *Simulation & Szenarien*
REPEATS = 2500
SEED = 42

# **Die zu analysierenden Regime:** Erwartete jährliche Rendite des Investments
SCENARIOS_MEAN_RETURN = {
    "Regime_1 (Niedrig)": 0.03, # 3% erwartete Rendite
    "Regime_2 (Mittel)": 0.05, # 5% erwartete Rendite
    "Regime_3 (Hoch)": 0.08    # 8% erwartete Rendite
}
# ==============================================================================


def calculate_lombard_profit_probability(mean_return: float):
    """
    Führt die Monte Carlo Simulation für ein einzelnes Rendite-Regime durch.
    Beinhaltet das Risiko des Margin Calls bei Jobverlust.
    """
    
    np.random.seed(SEED)

    # Korrekte Berechnung des Gesamtkredit-Endwerts (L_T) mit Float-Potenzierung
    total_loan_repayment = SUM_LOMB * ((1 + INTEREST_LOMB) ** DURATION_LOMB)
    
    # Schwellenwert für die Zwangsliquidation (Margin Call)
    # Portfolio-Wert, bei dem die Sicherheitsanforderungen verletzt werden:
    liquidation_threshold = SUM_LOMB / (1 - MAINTENANCE_MARGIN)

    net_terminal_values = []

    for _ in range(REPEATS):
        current_portfolio_value = TOTAL_INVESTED
        
        # Jede Iteration ist ein Jahr
        for year in range(1, DURATION_LOMB + 1):
            
            # --- 1. Marktentwicklung (Geometrische Brownsche Bewegung) ---
            # Rendite wird aus Normalverteilung gezogen
            yearly_return = np.random.normal(mean_return, YEARLY_VOLATILITY)
            current_portfolio_value *= (1 + yearly_return)
            
            # --- 2. Risiko-Event Prüfung (Jobverlust & Margin Call) ---
            
            # Jährliches Risiko: Job verloren?
            is_job_lost = np.random.rand() < CHANCE_BEING_FIRED_ANNUAL
            
            # Prüfung auf Margin Call Level
            is_below_margin = current_portfolio_value < liquidation_threshold

            if is_job_lost and is_below_margin:
                # Zwangsliquidation: Trade wird beendet
                
                # Zinsen bis zum Liquidation-Jahr berechnen
                loan_value_t = SUM_LOMB * ((1 + INTEREST_LOMB) ** year)
                net_value = current_portfolio_value - loan_value_t
                net_terminal_values.append(net_value)
                
                # Beende diese Simulation (Break)
                break
        
        # --- Normaler Abschluss der Laufzeit (falls keine Liquidation) ---
        else:
            # Die volle Laufzeit wurde erreicht
            net_value = current_portfolio_value - total_loan_repayment
            net_terminal_values.append(net_value)

    # Ergebnisse auswerten
    final_results = np.array(net_terminal_values)
    
    # 1. Profitwahrscheinlichkeit (Netto-Gewinn > 0)
    profitable_runs = np.sum(final_results > 0)
    profit_probability = profitable_runs / REPEATS
    
    # 2. Value at Risk (VaR) - 5% Perzentil (Worst 5% Runs)
    var_95 = np.percentile(final_results, 5)
    
    # 3. Erwarteter Wert (Mean)
    mean_net_value = np.mean(final_results)

    # 4. Zwangsliquidationen
    liquidation_count = np.sum(final_results < (liquidation_threshold - SUM_LOMB)) 
    
    return profit_probability, mean_net_value, var_95, liquidation_count


if __name__ == "__main__":
    
    results = {}
    
    print("--- Lombardkredit Szenarioanalyse gestartet ---")
    print(f"Basis-Parameter: Kredit {SUM_LOMB:,.0f} @ {INTEREST_LOMB*100}% für {DURATION_LOMB} Jahre.")
    print(f"Jobverlust-Risiko: {CHANCE_BEING_FIRED_ANNUAL*100}% p.a. | Investment Volatilität: {YEARLY_VOLATILITY*100}%.")
    print("-" * 50)
    
    # Iteration über alle definierten Rendite-Regime
    for scenario_name, mean_return in SCENARIOS_MEAN_RETURN.items():
        prob, mean_val, var, liq_count = calculate_lombard_profit_probability(mean_return)
        
        results[scenario_name] = {
            "Erwartete Rendite": f"{mean_return:.1%}",
            "Profitwahrscheinlichkeit": f"{prob:.2%}",
            "Erwarteter Netto-Gewinn": f"{mean_val:,.0f} CHF",
            "VaR 95% (Worst 5%)": f"{var:,.0f} CHF",
            "Zwangsliquidationen": f"{liq_count} Runs"
        }
        
    # Ergebnisse als DataFrame anzeigen
    df_results = pd.DataFrame.from_dict(results, orient='index')
    print("\nERGEBNISSE DER REGIME-SIMULATION (2500 Runs pro Szenario):")
    print(df_results)
    plt.show(calculate_lombard_profit_probability, mean)
