import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from scipy.stats import ttest_1samp, chi2, norm
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
np.random.seed(27029)


class RandomWalk:
    """
    Generates different types of Random Walks.

    Parameters:
    - Type: Type of process.
        "A" - Standard Brownian Motion dx(t)=dW(t)
        "B" - Arithmetic Brownian Motion dx(t)=adt+bdW(t)
        "C" - Multiplicative Linear Process dx(t)=bx(t)dW(t)
        "D" - Geometric Brownian Motion dx(t)=ax(t)dt+bx(t)dW(t)
        "E" - Ornstein-Uhlenbeck Process dx(t)=-a(x(t)-mu)dt+bdW(t)
    - x0: Initial value.
    - N_traiettorie: Number of trajectories.
    - Step_Temporali: Number of time steps.
    - dt: Time step size.
    - a: Drift coefficient.
    - b: Instantaneous Standard Deviation/Volatility.
    - mu: Stationary mean of the OU process.
    """
    def __init__(self, Type="A", x0=1, N_traiettorie=100, Step_Temporali=500, dt=0.01, a=0.1, b=0.2, mu=0):
        """
        Initializes a RandomWalk object with the specified parameters.
        """
        self.Type = Type
        self.x0 = x0
        self.N_traiettorie = N_traiettorie
        self.Step_Temporali = Step_Temporali
        self.dt = dt
        self.a = a
        self.b = b
        self.mu = mu

        # Initialize main variables
        self.time = np.linspace(dt, Step_Temporali * dt, Step_Temporali)
        self.positions = np.zeros((N_traiettorie, Step_Temporali))
        self.numerical_variance = np.zeros(Step_Temporali)
        self.log_returns = np.zeros((N_traiettorie, Step_Temporali)) if Type in ["C", "D"] else None
        self.theoretical_mean = None
        self.theoretical_variance = None

    def generate(self):
        """Generates the trajectories of the Random Walk."""
        for i in range(self.Step_Temporali):
            if self.Type == "A":
                dx = np.random.normal(0, self.b * np.sqrt(self.dt), self.N_traiettorie)
                self.positions[:, i] = self.positions[:, i-1] + dx if i > 0 else self.x0

            elif self.Type == "B":
                deterministic = self.a * self.dt
                stochastic = np.random.normal(0, self.b * np.sqrt(self.dt), self.N_traiettorie)
                dx = deterministic + stochastic
                self.positions[:, i] = self.positions[:, i-1] + dx if i > 0 else self.x0

            elif self.Type in ["C", "D"]:
                drift = -0.5 * (self.b ** 2) * self.dt if self.Type == "C" else self.a * self.dt - 0.5 * (self.b ** 2) * self.dt
                stochastic = self.b * np.random.normal(0, np.sqrt(self.dt), self.N_traiettorie)
                dlnx = drift + stochastic
                self.log_returns[:, i] = self.log_returns[:, i-1] + dlnx if i > 0 else np.log(self.x0)
                self.positions[:, i] = np.exp(self.log_returns[:, i])

            elif self.Type == "E":
                mu = self.mu
                deterministic = -self.a * (self.positions[:, i-1] - mu) * self.dt if i > 0 else 0
                stochastic = np.random.normal(0, self.b * np.sqrt(self.dt), self.N_traiettorie)
                dx = deterministic + stochastic
                self.positions[:, i] = self.positions[:, i-1] + dx if i > 0 else self.x0

            # Calculate numerical variance
            self.numerical_variance[i] = np.var(self.positions[:, i])

        # Calculate theoretical mean
        self.calculate_theoretical_mean()

    def calculate_theoretical_mean(self):
        """Calculates the theoretical mean and variance."""
        if self.Type == "A":
            self.theoretical_mean = np.full_like(self.time, self.x0)
            self.theoretical_variance = (self.b ** 2) * self.time
        elif self.Type == "B":
            self.theoretical_mean = self.x0 + self.a * self.time
            self.theoretical_variance = (self.b ** 2) * self.time
        elif self.Type == "C":
            if self.a != 0:
                self.a = 0
            self.theoretical_mean = self.x0 * np.exp(self.a * self.time)
            self.theoretical_variance = (self.x0**2)*(np.exp((self.b**2)*self.time)-1)
        elif self.Type == "D":
            self.theoretical_mean = self.x0 * np.exp(self.a * self.time)
            self.theoretical_variance = (self.x0**2)*np.exp(2*self.a*self.time)*(np.exp((self.b**2)*self.time)-1)
        elif self.Type == "E":
            mu = self.mu
            self.theoretical_mean = mu + (self.x0 - mu) * np.exp(-self.a * self.time)
            self.theoretical_variance = (self.b ** 2) / (2 * self.a) * (1 - np.exp(-2 * self.a * self.time))

    def plot_mean_variance(self):
        """Plots empirical and theoretical mean/variance."""
        ensemble_mean = np.mean(self.positions, axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(self.time, ensemble_mean, label='Empirical Mean', color='blue')
        plt.plot(self.time, self.theoretical_mean, label='Theoretical Mean', color='red', linestyle='--')
        
        # Add confidence bands
        plt.fill_between(self.time, ensemble_mean - np.sqrt(self.numerical_variance),
                         ensemble_mean + np.sqrt(self.numerical_variance),
                         color='blue', alpha=0.2, label='Empirical ± std')
        plt.fill_between(self.time, self.theoretical_mean - np.sqrt(self.theoretical_variance),
                         self.theoretical_mean + np.sqrt(self.theoretical_variance),
                         color='red', alpha=0.2, label='Theoretical ± std')
        
        plt.xlabel('Time')
        plt.ylabel('Mean and Variance')
        plt.title(f'Mean and Variance ({self.Type})')
        plt.legend(framealpha=0.5)
        plt.show()

    def plot_trajectories(self, n_shown=10):
        """Plots the trajectories."""
        plt.figure(figsize=(10, 5))
        for j in range(n_shown):
            plt.plot(self.time, self.positions[j, :], lw=0.8, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('$x(t)$')
        plt.title(f'Trajectories with Confidence Band - Type {self.Type}')
        plt.show()

    def plot_log_returns(self):
        """Plots log-returns for Type C and D."""
        if self.Type not in ["C", "D"]:
            print("Log-returns are only defined for types C and D.")
            return
        
        log_mean_theoretical = (np.log(self.x0) - 0.5 * (self.b ** 2) * self.time) if self.Type == "C" else (np.log(self.x0) + self.a * self.time - 0.5 * (self.b ** 2) * self.time)
        log_mean_empirical = np.mean(self.log_returns, axis=0)
        log_variance_empirical = np.var(self.log_returns, axis=0)
        log_variance_theoretical = (self.b ** 2) * self.time

        plt.figure(figsize=(10, 5))
        plt.plot(self.time, log_mean_empirical, label='Empirical Log-Mean', color='blue')
        plt.plot(self.time, log_mean_theoretical, label='Theoretical Log-Mean', color='red', linestyle='--')
        
        # Add confidence bands
        plt.fill_between(self.time, log_mean_empirical - np.sqrt(log_variance_empirical),
                         log_mean_empirical + np.sqrt(log_variance_empirical),
                         color='blue', alpha=0.3, label='Empirical Log ± std')
        plt.fill_between(self.time, log_mean_theoretical - np.sqrt(log_variance_theoretical),
                         log_mean_theoretical + np.sqrt(log_variance_theoretical),
                         color='red', alpha=0.2, label='Theoretical Log ± std')
        
        plt.xlabel('Time')
        plt.ylabel('$\ln(x(t))$')
        plt.title(f'Log-Returns Mean and Variance ({self.Type})')
        plt.legend(framealpha=0.5)
        plt.show()


    def asymptotic_distributions(self, t=None, bins=None):
        """
        Compares the empirical distribution of prices (or log-prices for Type C and D) at time t (default last step)
        with a Gaussian distribution having the same mean and variance.
        Includes a statistical test for mean and variance.

        Parameters:
        - t: Time step to consider (default last step).
        - bins: Number of bins for the histogram (default automatic).
        """
        if t is None:
            t = self.Step_Temporali - 1  # Last time step

        # Determine which dataset to use (prices or log-prices)
        if self.Type in ["C", "D"]:
            # For Type C and D, work with the logarithm of the data
            data_at_t = np.log(self.positions[:, t])
            theoretical_mean = (
                np.log(self.x0) - 0.5 * (self.b ** 2) * self.time[t]
                if self.Type == "C"
                else np.log(self.x0) + self.a * self.time[t] - 0.5 * (self.b ** 2) * self.time[t]
            )
            theoretical_variance = (self.b ** 2) * self.time[t]
        else:
            # For other types, use prices directly
            data_at_t = self.positions[:, t]
            theoretical_mean = self.theoretical_mean[t]
            theoretical_variance = self.theoretical_variance[t]

        # Compute empirical mean and variance
        empirical_mean = np.mean(data_at_t)
        empirical_variance = np.var(data_at_t, ddof=1)  # Sample variance (ddof=1)

        # Statistical test for the mean (t-test)
        t_stat, p_value_mean = ttest_1samp(data_at_t, theoretical_mean)

        # Statistical test for the variance (chi-squared test)
        n = len(data_at_t)
        chi2_stat = (n - 1) * empirical_variance / theoretical_variance
        p_value_variance = 1 - chi2.cdf(chi2_stat, df=n - 1)

        # Number of bins (automatic if bins=None)
        if bins is None:
            bins = int(np.sqrt(len(data_at_t)))  # Square root rule

        # Histogram of the empirical distribution
        plt.hist(data_at_t, bins=bins, density=True, alpha=0.6, color="b", label="Empirical Distribution")

        # Theoretical Gaussian distribution with the same mean and variance
        x = np.linspace(min(data_at_t), max(data_at_t), 1000)
        y_gaussian = norm.pdf(x, theoretical_mean, np.sqrt(theoretical_variance))
        plt.plot(x, y_gaussian, 'r-', lw=2, label="Theoretical Gaussian Distribution")

        # Title, labels, and legend
        plt.title(f"Asymptotic Distributions at Time t={self.time[t]:.2f}")
        plt.xlabel("$\\ln[x(t)]$" if self.Type in ["C", "D"] else "Price $x(t)$")
        plt.ylabel("Probability Density")

        # Add test results to the legend
        legend_test = (
            # f"Mean Test: p-value={p_value_mean:.3f}\n"
            # f"Variance Test: p-value={p_value_variance:.3f}"
        )
        plt.legend(framealpha=0.5, loc="upper left", title=legend_test)

        # Show the plot
        plt.show()







    def log_normal_distributions(self, t=None, N_bins=None):
        """
        Compares the empirical distribution of prices with a theoretical log-normal distribution
        at time t (default is the last one) for Types C and D.

        Parameters:
        - t: Time step to consider (default is the last one).
        - bins: Number of bins for the histogram (default is automatic).
        """
        if self.Type not in ["C", "D"]:
            print("This method is defined only for Types C and D.")
            return

        if t is None:
            t = self.Step_Temporali - 1  # Last time step

        if N_bins is None:
            N_bins = int(np.sqrt(len(self.positions[:, t])))  # Square root rule

        # Empirical data
        data_at_t = self.positions[:, t]

        # Parameters for the theoretical log-normal distribution
        mean_ln = (np.log(self.x0) - 0.5 * (self.b ** 2) * self.time[t] if self.Type == "C" else np.log(self.x0) + self.a * self.time[t] - 0.5 * (self.b ** 2) * self.time[t])
        var_ln = (self.b ** 2) * self.time[t]
        sigma_ln = np.sqrt(var_ln)

        # Empirical mean and variance of the logarithm of the data
        log_data = np.log(data_at_t)
        empirical_mean_ln = np.mean(log_data)
        empirical_var_ln = np.var(log_data, ddof=1)  # Sample variance

        # Statistical test for the mean (t-test)
        t_stat_ln, p_value_ln_mean = ttest_1samp(log_data, mean_ln)

        # Generate theoretical x values (logarithmic scale)
        x_theoretical = np.logspace(np.log10(min(data_at_t)), np.log10(max(data_at_t)), 100)
        #x_theoretical = np.logspace(0.01, np.log10(max(data_at_t)), 100)

        y_theoretical = lognorm.pdf(x_theoretical, s=sigma_ln, scale=np.exp(mean_ln))

        # Empirical histogram with logarithmic binning
        bins = np.logspace(np.log10(min(data_at_t)), np.log10(max(data_at_t)), N_bins)
        #bins = np.logspace(0.01, np.log10(max(data_at_t)), N_bins)
        plt.hist(data_at_t, bins=bins, density=True, alpha=0.6, color='blue', label="Empirical Distribution")

        # Plot the theoretical distribution
        plt.plot(x_theoretical, y_theoretical, 'r-', label="Theoretical Distribution (Lognormal)")

        # Plot settings
        plt.xscale('log')  # Logarithmic scale on the x-axis
        plt.yscale('log')  # Logarithmic scale on the y-axis
        plt.xlabel("Value")
        plt.ylabel("Density (log)")
        plt.title(f"Lognormal Distribution (Type {self.Type}) at time t = {self.time[t]:.2f}")
        # Add test results to the legend
        test_legend = (
            #f"Log Mean Test: p-value={p_value_ln_mean:.3f}\n"
            #f"Log Variance Test: p-value={p_value_ln_var:.3f}"
        )
        plt.legend(framealpha=0.5, loc="upper right", title=test_legend)
        #plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()


# Creazione dell'oggetto RandomWalk
rw = RandomWalk(Type="A", x0=5, N_traiettorie=1000, Step_Temporali=1000, dt=0.01, a=0.1, b=1, mu=0)
#N.B: per C e D riduci di tanto gli step temporali!! tipo 500 (altrimenti per gli altri fai 10000)

# Generazione delle traiettorie
rw.generate()

# Plot dei risultati
rw.plot_mean_variance()
rw.plot_trajectories()
rw.plot_log_returns()
rw.asymptotic_distributions()
       
        
rw = RandomWalk(Type="C", x0=5, N_traiettorie=10000, Step_Temporali=450, dt=0.01, a=0.1, b=1, mu=0)

# Generazione delle traiettorie
rw.generate()

# Plot dei risultati
rw.plot_mean_variance()
rw.plot_trajectories()
rw.plot_log_returns()
rw.asymptotic_distributions()
rw.log_normal_distributions()
