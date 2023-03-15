import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# IMPORT DATA AND SET CONSTANTS

files = ["MLA/0-2MLA", "MB/0-6MB", "ML/0-8ML", "MLA/1-MLA", "MLA/1-2MLA", "ML/1-5ML", "ML/2-0ML"]
samples = {}
volume_solute = [0.2, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]  # mL (cm^3)
volume_solvent = 10/1000  # Liters

ConcentrationInSolute = 0.05
Density = 1.049  # g/cm^3
MolarMass = 60.052  # g/mol

# 1) GRAPH THE RESULTS

for i in range(len(files)):

    # Read and transform to float
    data = pd.read_csv("./experimental_data/csv/" + files[i] + ".csv", names=["x", "y"])
    data = (data.iloc[86:])
    data = data.astype(float)
    data = data.loc[data["x"] < 300]

    # Get the positions of the peaks
    peak_y = float(max(data.loc[data["x"] > 260]["y"]))
    peak_x = float(data.loc[data["y"] == peak_y]["x"])
    peak = {"x": peak_x, "y": peak_y}
    print(peak)

    # Calculate the Concentration (Molarity : Molar Concentration)
    MolesOfSolute = (volume_solute[i]*Density)/MolarMass * ConcentrationInSolute
    Molarity = MolesOfSolute/(volume_solvent + volume_solute[i]/1000)

    # Join information in dictionary
    samples[files[i]] = {"peak": peak, "molarity": Molarity}

    # Plot data
    plt.plot(data["x"], data["y"])


plt.title("ESPECTROMETRÍA UV-VIS DE SOLUCIÓN DE ÁCIDO ACÉTICO")
plt.ylabel("Absorbancia")
plt.xlabel("Longitud de Onda (nm)")
plt.legend(files)
plt.grid()
plt.show()


# 2) GRAPH OG BEER-LAMBERT'S LAW

molarity_x = []
absorbance_y = []

for i in range(0, len(files)):
    molarity_x.append(samples[files[i]]["molarity"])
    absorbance_y.append(samples[files[i]]["peak"]["y"])

# Linear Regression
molarity_x = np.array(molarity_x).reshape((-1, 1))
absorbance_y = np.array(absorbance_y)
lineal_reg = LinearRegression().fit(molarity_x, absorbance_y)

x_fit = np.linspace(min(molarity_x), max(molarity_x), 10)
y_fit = lineal_reg.predict(x_fit)

score = "{0:.2f}".format(lineal_reg.score(molarity_x, absorbance_y))
coef = "{0:.2f}".format(lineal_reg.coef_[0])
intercept = "{0:.2f}".format(lineal_reg.intercept_)

# Draw
plt.scatter(molarity_x, absorbance_y)
plt.plot(x_fit, y_fit, c="r")
plt.title("LEY DE BEER-LAMBERT (SCORE: R^2 = " + str(score) + ")")
plt.ylabel("Absorbancia")
plt.xlabel("Molaridad (mol/L)")
plt.grid()
plt.legend(["Datos Experimentales", "Ajuste Lineal [y = " + str(coef) + "x + " + str(intercept) + "]"])
plt.show()
