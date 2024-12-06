import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# Extract red_channel
def extract_red(filename, noise = False):
  img = Image.open(filename).convert("RGB")
  red_channel = np.array(img)[:, :, 0]
  red_channel = red_channel / 255.0
  
  if noise:
    red_channel= gaussian_filter(red_channel, sigma=1)
  
  return red_channel

# Plot image
def visualize_image(image, title="Image", cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.axis("off")
    plt.show()
  
# 2D Gaussian Function
def gaussian_2d(coords, I0, x0, y0, sigma_x, sigma_y):
  x, y = coords
  return I0 * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))).ravel()

# Fit Gaussian
def fit_gaussian(image):
  x = np.arange(image.shape[1])
  y = np.arange(image.shape[0])
  x, y = np.meshgrid(x, y)
  
  y0, x0 = np.unravel_index(np.argmax(image), image.shape)
  initial_guess = (image.max(), x0, y0, 10, 10)
  params, _ = curve_fit(gaussian_2d, (x, y), image.ravel(), p0=initial_guess)
  return params

# Visualize Fit
def visualize_fit(image, params):
  x = np.arange(image.shape[1])
  y = np.arange(image.shape[0])
  x, y = np.meshgrid(x, y)
  
  fit_image = gaussian_2d((x, y), *params).reshape(image.shape)
  plt.imshow(image, cmap="gray")
  plt.contour(fit_image, colors="red")
  plt.title("Gaussian Fit")
  plt.axis("off")
  plt.show()

# Plot Cross-Sections
def plot_cross_sections(image):
  center_x = image.shape[1] // 2
  center_y = image.shape[0] // 2

  horizontal = image[center_y, :]
  vertical = image[:, center_x]

  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.plot(horizontal, label="Horizontal")
  plt.title("Horizontal Cross-Section")
  plt.xlabel("Pixel")
  plt.ylabel("Intensity")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(vertical, label="Vertical")
  plt.title("Vertical Cross-Section")
  plt.xlabel("Pixel")
  plt.ylabel("Intensity")
  plt.legend()
  
  plt.tight_layout()
  plt.show()

# Calculate Beam Parameters
def calculate_beam_parameters(params):
  I0, x0, y0, sigma_x, sigma_y = params
  
  return {
    "Peak Intensity": I0,
    "Center (x, y)": (x0, y0),
    "Beam Width (sigma_x)": sigma_x,
    "Beam Width (sigma_y)": sigma_y,
    "Ellipticity (sigma_x / sigma_y)": sigma_x / sigma_y
  }

# Fourier Analysis
def fourier_analysis(image):
  fft_image = np.fft.fftshift(np.fft.fft2(image))
  magnitude = np.abs(fft_image)
  plt.imshow(np.log1p(magnitude), cmap="viridis")
  plt.title("Fourier Transform (Log Magnitude)")
  plt.colorbar(label="Log Intensity")
  plt.axis("off")
  plt.show()
  return magnitude

# Total Power Distribution
def calculate_total_power(image):
  return np.sum(image)

