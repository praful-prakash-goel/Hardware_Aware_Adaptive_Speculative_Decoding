import pandas as pd
import os

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results/')

PYTHIA_ADAPTIVE_NEW_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_pythia.csv")
PYTHIA_ADAPTIVE_OLD_PATH = os.path.join(RESULTS_DIR, "adaptive_overall_pythia_old.csv")

pythia_new_df = pd.read_csv(PYTHIA_ADAPTIVE_NEW_PATH)
pythia_old_df = pd.read_csv(PYTHIA_ADAPTIVE_OLD_PATH)

tps_cols = [col for col in pythia_new_df if col.endswith("TPS")]
target_cols = ['run_id', 'generation_length'] + tps_cols

pythia_new_df = pythia_new_df[target_cols]
pythia_old_df = pythia_old_df[target_cols]

grouped_new = pythia_new_df.groupby('generation_length')[tps_cols].agg(['mean', 'std'])
grouped_new.columns = [f"{col} {stat.capitalize()}" for col, stat in grouped_new.columns]
pythia_new_stats_df = grouped_new.reset_index()

grouped_old = pythia_old_df.groupby('generation_length')[tps_cols].agg(['mean', 'std'])
grouped_old.columns = [f"{col} {stat.title()}" for col, stat in grouped_old.columns]
pythia_old_stats_df = grouped_old.reset_index()

print("New controller : ")
print(pythia_new_stats_df)
print("\n Old controller :")
print(pythia_old_stats_df)