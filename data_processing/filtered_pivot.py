import pandas as pd
import random

def generate_new_ids(num_ids, existing_ids, min_id=100, max_id=999):

    existing_set = set(existing_ids)  # for fast lookups
    new_ids = []

    while len(new_ids) < num_ids:
        candidate = random.randint(min_id, max_id)
        # only add if not already existing
        if candidate not in existing_set:
            new_ids.append(candidate)
            existing_set.add(candidate)

    return new_ids
df = pd.read_csv("cleaned_austin.csv")

df["local_15min"] = pd.to_datetime(df["local_15min"], utc=True).dt.tz_convert("America/Chicago")

start_date = pd.Timestamp("2018-03-21", tz="America/Chicago")
end_date   = pd.Timestamp("2018-05-21 23:59:59", tz="America/Chicago")
df = df[(df["local_15min"] >= start_date) & (df["local_15min"] <= end_date)]

original_10_ids = [661, 1642, 2335, 2361, 2818, 4373, 5746, 7901, 7951, 8386]
df = df[df["dataid"].isin(original_10_ids)]

df["total_solar"] = df["total_solar"].fillna(0)

df_pivot = df.pivot_table(
    index="local_15min",
    columns="dataid",
    values=["grid", "total_solar"],
    fill_value=0
)

df_pivot.columns = [f"{col}_{house}" for col, house in df_pivot.columns]

# Move timestamps back to columns
df_pivot.reset_index(inplace=True)

desired_total_houses = 100  # Could be 25, 50, 100, etc.

current_count = len(original_10_ids)
if desired_total_houses < current_count:
    raise ValueError(
        f"You already have {current_count} houses, which is more than desired_total_houses={desired_total_houses}!"
    )

num_new_needed = desired_total_houses - current_count
existing_ids = set(original_10_ids)

new_ids = generate_new_ids(num_new_needed, existing_ids)

print(f"Generating {num_new_needed} new house IDs to reach {desired_total_houses} total.\n"
      f"New IDs = {new_ids}\n")


for i, new_id in enumerate(new_ids):
    old_id = original_10_ids[i % len(original_10_ids)]
    df_pivot[f"grid_{new_id}"] = df_pivot[f"grid_{old_id}"]
    df_pivot[f"total_solar_{new_id}"] = df_pivot[f"total_solar_{old_id}"]


time_col = ["local_15min"]
grid_cols = [c for c in df_pivot.columns if c.startswith("grid_")]
solar_cols = [c for c in df_pivot.columns if c.startswith("total_solar_")]

def get_house_id(col_name, prefix):
    # e.g. "grid_661" -> "661", then int -> 661
    return int(col_name.replace(prefix, ""))

grid_cols_sorted  = sorted(grid_cols,  key=lambda x: get_house_id(x, "grid_"))
solar_cols_sorted = sorted(solar_cols, key=lambda x: get_house_id(x, "total_solar_"))

final_columns = time_col + grid_cols_sorted + solar_cols_sorted

df_pivot = df_pivot[final_columns]

output_filename = f"filtered_pivoted_{desired_total_houses}_houses.csv"
df_pivot.to_csv(output_filename, index=False)

print(f"✅ Duplication complete! Saved as {output_filename}")
