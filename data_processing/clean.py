import pandas as pd

input_csv = "/Users/ananygupta/Desktop/MADDPG_SOLAR_TRADER/15minute_data_california.csv"
output_csv = "cleaned_california_data.csv"


df = pd.read_csv(input_csv)


columns_to_keep = [
    "dataid",
    "local_15min",
    "grid",
    "solar",
    "solar2"
]
df = df[columns_to_keep]

df = df.fillna(0)

df["total_solar"] = df["solar"] + df["solar2"]

df.drop(columns=["solar", "solar2"], inplace=True)

df = df[["dataid", "local_15min", "grid", "total_solar"]]
df.to_csv(output_csv, index=False)

print(f"✅ Data cleaning complete! Saved as '{output_csv}'")
