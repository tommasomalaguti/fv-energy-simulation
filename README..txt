# Dashboard Consumi & FV – Streamlit

Questa app mostra consumi e produzione FV (12 mesi) e consente una simulazione batteria.

## Avvio locale
pip install -r requirements.txt
streamlit run main.py

## CSV di esempio
- Il file di default caricato dall'app è: combined_Feb2024_to_Jan2025_hourly_F123_withNetGrid_lower.csv

## Deploy su Streamlit Community Cloud
1. Crea un nuovo repo su GitHub con questi file.
2. Vai su streamlit.io → Deploy app → collega il repo, scegli main.py.
3. Condividi l'URL pubblico.