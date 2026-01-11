import pandas as pd
import matplotlib.pyplot as plt
import datetime

from core_simulator import SimulatorLogic

# 1. SETUP: Create a Mock Logic Instance
# We pass 'None' for the petri net arguments because generate_arrivals doesn't use them.
# Ensure 'schedule_dict' is defined in your environment before running this!
test_logic = SimulatorLogic(None, None, None)

print("--- 1. Generating Arrivals (This may take a few seconds) ---")
arrivals = test_logic.generate_arrivals()
print(f"Total Cases Generated: {len(arrivals)}")

# 2. CONVERT TO DATAFRAME
# Extract just the timestamps and attributes we want to check
data = []
for item in arrivals:
    data.append({
        'timestamp': item['arrival_time'],
        'season_factor': item['attributes']['SeasonalityFactor']
    })

df = pd.DataFrame(data)
df['date'] = df['timestamp'].dt.date
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday
df['hour'] = df['timestamp'].dt.hour

# --- TEST 1: CHECK HOLIDAYS (The "Closed Bank" Test) ---
print("\n--- TEST 1: HOLIDAY CHECK ---")
# Check specific Dutch holidays (e.g., King's Day April 27)
kings_day = datetime.date(2016, 4, 27)
kings_day_count = df[df['date'] == kings_day].shape[0]

if kings_day_count == 0:
    print(f"✅ SUCCESS: King's Day ({kings_day}) has 0 arrivals.")
else:
    print(f"❌ FAILURE: King's Day has {kings_day_count} arrivals! (Should be 0)")

# --- TEST 2: CHECK SEASONALITY (The "September Peak" Test) ---
print("\n--- TEST 2: SEASONALITY CHECK ---")
monthly_counts = df.groupby('month').size()
jan_count = monthly_counts.get(1, 0)
sep_count = monthly_counts.get(9, 0)

print(f"January Count: {jan_count}")
print(f"September Count: {sep_count}")

if sep_count > jan_count * 1.3:
    print("✅ SUCCESS: September is significantly busier than January.")
else:
    print("❌ FAILURE: Seasonality multiplier might not be working.")

# --- TEST 3: CHECK WEEKLY RHYTHM (The "Monday vs. Sunday" Test) ---
print("\n--- TEST 3: WEEKLY RHYTHM CHECK ---")
# Count arrivals by weekday (0=Mon, 6=Sun)
weekday_counts = df.groupby('weekday').size()
print(weekday_counts)

# Visual Confirmation
plt.figure(figsize=(15, 5))

# Plot 1: Monthly Trend
plt.subplot(1, 2, 1)
monthly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Monthly Volume (Check for Seasonality)")
plt.xlabel("Month")
plt.ylabel("Total Arrivals")

# Plot 2: Weekly Hourly Pattern (First Week Only)
plt.subplot(1, 2, 2)
# Filter for first 7 days to see the clear "Camel Humps"
first_week = df[df['timestamp'] < datetime.datetime(2016, 1, 8)]
first_week_hourly = first_week.groupby(first_week['timestamp'].dt.floor('h')).size()
first_week_hourly.plot()
plt.title("Hourly Volume (First Week)")
plt.xlabel("Date/Time")
plt.ylabel("Arrivals per Hour")

plt.tight_layout()
plt.show()