import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate normal, noisy sensor data
np.random.seed(42)
time_steps = np.arange(100)
normal_vibration = np.random.normal(loc=10, scale=2, size=100)

# 2. Introduce an anomaly (e.g., a bearing failing at step 70)
normal_vibration[70:] += np.random.normal(loc=0, scale=8, size=30)

df = pd.DataFrame({'time': time_steps, 'vibration': normal_vibration})

# 3. Apply rolling windows (Window Size = 10)
window_size = 10
df['rolling_mean'] = df['vibration'].rolling(window=window_size).mean()
df['rolling_std'] = df['vibration'].rolling(window=window_size).std()

# 4. Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['vibration'], alpha=0.4, label='Raw Sensor Data (Noisy)', color='gray')
plt.plot(df['time'], df['rolling_mean'], label='Rolling Mean (Smoothed)', color='blue', linewidth=2)
plt.plot(df['time'], df['rolling_std'], label='Rolling Std Dev (Anomaly Indicator)', color='red', linewidth=2)
plt.axvline(x=70, color='black', linestyle='--', label='Anomaly Starts Here')

plt.title('Sensor Vibration: Raw Data vs. Rolling Windows')
plt.xlabel('Time Steps')
plt.ylabel('Vibration Amplitude')
plt.legend()
plt.grid(True)
plt.show()