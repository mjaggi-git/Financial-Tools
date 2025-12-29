# This python file does simulate a leveraged portfolio with the the inclusion of likeiyhood of being fired (absence of steady main cash flow)






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LombardRiskSimulator:
    def __init__(self, loan_sum=50000, loan_interest=0.04, duration_yrs=5, 
                 portfolio_value=390000, margin_level=0.6, job_loss_risk=0.05, 
                 volatility=0.15, repeats=5000):
        self.loan_sum = loan_sum
        self.loan_interest = loan_interest
        self.duration_yrs = duration_yrs
        self.portfolio_start = portfolio_value + loan_sum
        self.margin_threshold = portfolio_value * margin_level
        self.job_loss_risk = job_loss_risk
        self.volatility = volatility
        self.repeats = repeats
        self.results_df = None

    def run_simulation(self, regimes):
        simulation_data = {}
        
        for name, mean_return in regimes.items():
            net_values = []
            liquidations = 0
            
            for _ in range(self.repeats):
                current_val = self.portfolio_start
                liquidated = False
                
                for year in range(1, self.duration_yrs + 1):
                    yearly_return = np.random.normal(mean_return, self.volatility)
                    current_val *= (1 + yearly_return)
                    
                    is_job_lost = np.random.rand() < self.job_loss_risk
                    is_below_margin = current_val < self.margin_threshold

                    if is_job_lost and is_below_margin:
                        loan_at_t = self.loan_sum * ((1 + self.loan_interest) ** year)
                        net_values.append(current_val - loan_at_t)
                        liquidations += 1
                        liquidated = True
                        break
                
                if not liquidated:
                    total_repayment = self.loan_sum * ((1 + self.loan_interest) ** self.duration_yrs)
                    net_values.append(current_val - total_repayment)

            simulation_data[name] = np.array(net_values)
        
        self.results_df = pd.DataFrame(simulation_data)
        return self._summarize(regimes)

    def _summarize(self, regimes):
        summary = {}
        for name in regimes.keys():
            data = self.results_df[name]
            summary[name] = {
                "Profit Prob.": f"{(data > self.portfolio_start - self.loan_sum).mean():.2%}",
                "Expected Value": f"{data.mean():,.0f} CHF",
                "VaR 95%": f"{np.percentile(data, 5):,.0f} CHF",
                "Liquidation Risk": f"{(self.results_df[name] < self.margin_threshold).mean():.2%}"
            }
        return pd.DataFrame(summary).T

    def plot_results(self):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 7))
        
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        for i, col in enumerate(self.results_df.columns):
            sns.kdeplot(self.results_df[col], fill=True, label=col, color=colors[i], alpha=0.5)

        plt.axvline(x=self.portfolio_start - self.loan_sum, color='black', linestyle='--', label='Break-even')
        plt.title("Monte Carlo Simulation: Lombard Credit Risk Scenarios", fontsize=16)
        plt.xlabel("Net Portfolio Value in Mio. (CHF) after 5 Years", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    regimes = {
        "Low Return (3%)": 0.03,
        "Mid Return (5%)": 0.05,
        "High Return (8%)": 0.08
    }

    simulator = LombardRiskSimulator(repeats=10000)
    summary_stats = simulator.run_simulation(regimes)
    
    print("\n--- Simulation Summary Statistics ---")
    print(summary_stats)
    
    simulator.plot_results()

