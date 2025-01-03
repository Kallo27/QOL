import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.optimize import curve_fit

def calculate_threshold(current, output_power):
  # Define regions for non-lasing and lasing (manually)
  non_lasing_region = current < 0.04
  lasing_region = current > 0.04

  # Linear regression for both regions
  result1 = linregress(current[non_lasing_region], output_power[non_lasing_region])
  result2 = linregress(current[lasing_region], output_power[lasing_region])

  # Threshold is the intersection point
  num = (result2.intercept - result1.intercept)
  den = (result1.slope - result2.slope)
  threshold_current = num / den 

  # Error propagation for threshold current
  err_num = (result1.intercept_stderr + result2.intercept_stderr)
  err_den = (result1.stderr + result2.stderr)
  threshold_error = np.abs(threshold_current * ((err_num / num) + (err_den / den)))
  
  return threshold_current, threshold_error

def plot_LI_curve(current, output_power, threshold_current, threshold_error):
  plt.figure(figsize=(8, 5))
  plt.scatter(current, output_power, label='LI curve')
  plt.axvline(threshold_current, color='r', linestyle='--', label=f'Threshold: {threshold_current:.4f} +/- {threshold_error:.4f} mA')
  plt.xlabel('Current [A]')
  plt.ylabel('Output power [W]')
  plt.legend()
  plt.title('LI curve')
  
def fit_lasing_region(current, output_power, threshold_current):
  # Select the lasing region (current > threshold_current)
  lasing_region = current > threshold_current
  fit_result = linregress(current[lasing_region], output_power[lasing_region])
  
  # Extract fit parameters and uncertainties
  slope = fit_result.slope  # Differential efficiency
  intercept = fit_result.intercept
  slope_error = fit_result.stderr
  intercept_error = fit_result.intercept_stderr
  
  return slope, intercept, slope_error, intercept_error

def plot_fitted_lasing_curve(current, output_power, threshold_current, slope, intercept):
  x_fit = np.linspace(threshold_current, max(current), 500)
  y_fit = slope * x_fit + intercept

  plt.figure(figsize=(8, 5))
  plt.scatter(current, output_power, label='Data', color='blue')
  plt.plot(x_fit, y_fit, label='Linear fit', color='orange')
  plt.axvline(threshold_current, color='r', linestyle='--', label=f'Threshold: {threshold_current:.4f} A')
  plt.xlabel('Current [A]')
  plt.ylabel('Output power [W]')
  plt.legend()
  plt.title('LI curve')
  plt.show()
  
  
def plot_VI_curve(current, voltage):
  plt.figure(figsize=(8, 5))
  plt.plot(current, voltage, marker='o', linestyle='-', color='blue', label='VI curve')
  plt.xlabel('Current [A]')
  plt.ylabel('Voltage [V]')
  plt.title('VI curve')
  plt.grid(True)
  plt.legend()
  plt.show()
  
def fit_VI_threshold(current, voltage, threshold_current):  
  # Filter data for current > threshold_current
  lasing_region = current > threshold_current
  fit_result1 = linregress(current[lasing_region], voltage[lasing_region])
  
  # Filter data for current < threshold_current
  non_lasing_region = current < threshold_current
  fit_result2 = linregress(current[non_lasing_region], voltage[non_lasing_region])
  
  return fit_result1, fit_result2

def plot_fit_VI_threshold(current, voltage, threshold_current, fit_1, fit_2):
  # Generate fit line
  x_fit1 = np.linspace(threshold_current, max(current), 500)
  x_fit2 = np.linspace(min(current), threshold_current, 500)
  y_fit1 = fit_1.slope * x_fit1 + fit_1.intercept
  y_fit2 = fit_2.slope * x_fit2 + fit_2.intercept

  plt.figure(figsize=(8, 5))
  plt.scatter(current, voltage, label='Data', color='blue', alpha=0.6)
  plt.plot(x_fit1, y_fit1, label=f'Fit (above threshold): V = {fit_1.slope:.3f} I + {fit_1.intercept:.3f}', color='orange')
  plt.plot(x_fit2, y_fit2, label=f'Fit (under threshold): V = {fit_2.slope:.3f} I + {fit_2.intercept:.3f}', color='orange')
  plt.axvline(threshold_current, color='r', linestyle='--', label=f'Threshold: {threshold_current:.3f} A')
  plt.xlabel('Current [A]')
  plt.ylabel('Voltage [V]')
  plt.title('VI curve with fit and threshold')
  plt.grid(True)
  plt.legend()
  plt.show()
  
def wall_plug_efficiency(current, voltage, output_power):
  electrical_power = current * voltage
  wpe = output_power / electrical_power
  return wpe

def plot_wpe(current, threshold_current, wpe):
  plt.figure(figsize=(8, 5))
  plt.plot(current, wpe, marker='o', label='Wall-Plug efficiency', color='orange')
  plt.axvline(threshold_current, color='r', linestyle='--', label=f'Threshold: {threshold_current:.3f} A')
  plt.xlabel('Current [A]')
  plt.ylabel('Efficiency [%]')
  plt.legend()
  plt.title('Efficiency')
  plt.show()
  
def fit_wpe(current, threshold_current, wpe):
  # Exponential Saturation Model
  def exp_saturation(I, A, B, C):
    return A / (1 + B * np.exp(-C * I))
  
  # Fit WPE as a function of current
  params, covariance = curve_fit(exp_saturation, current, wpe, p0=[max(wpe), 1, 1])
  
  # Extract fitted parameters and errors
  A_fit, B_fit, C_fit = params
  A_error, B_error, C_error = np.sqrt(np.diag(covariance))

  WPE_fit = exp_saturation(current, *params)
  
  # Plot the WPE curve and fit
  plt.figure(figsize=(8, 5))
  plt.plot(current, wpe, marker='o', label='Measured WPE', linestyle='')
  plt.plot(current, WPE_fit, label=f'Fit: A / (1 + B * exp(-C * I))')
  plt.axvline(threshold_current, color='r', linestyle='--', label=f'Threshold: {threshold_current:.3f} A')
  plt.xlabel('Current [A]')
  plt.ylabel('WPE')
  plt.title('Wall-Plug efficiency vs current')
  plt.grid(True)
  plt.legend(loc='lower right')
  plt.show()
  
  # Print fit parameters
  print("Fit parameters:")
  print(f"A = {A_fit:.3f} +/- {A_error:.3f}, B = {B_fit:.2f} +/- {B_error:.2f}, C = {C_fit:.2f} +/- {C_error:.2f}")