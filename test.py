import pandas as pd, pprint, textwrap, os

PV_2023_FILE = r"C:\Users\TommasoMalaguti\Desktop\fv_sim_project\Timeseries_44.852_11.207_SA3_6kWp_crystSi_14_39deg_0deg_2023_2023.csv"

df = pd.read_csv(
    PV_2023_FILE,
    sep=";",      # se hai dubbi sul separatore, prova anche ',' o '\t'
    comment="#",
    nrows=5,      # solo prime 5 righe
    header=None   # evita che le prime righe commento diventino header
)

print("\n>>> PRIME 5 RIGHE RAW:")
print(df.to_string(index=False, header=False)[:300])  # stampa tagliata

# ora prova con prima riga come header vera:
df2 = pd.read_csv(PV_2023_FILE, sep=";", comment="#", nrows=0)
print("\n>>> HEADER (interpretato):")
print(df2.columns.tolist())
