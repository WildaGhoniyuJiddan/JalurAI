# JalurAI

Smart logistics risk prediction system.

## Data layout

- `../jalurai_orders.csv` — 20.848 order e-commerce Indonesia. Fitur order
  dan ongkir bersumber dari data nyata; label keterlambatan dan kelebihan
  biaya disimulasikan dengan seed tetap karena event distributor tidak tersedia.
- `../last_mile_delivery.csv` — data mentah titik/jalan last-mile Pontianak
  dalam format wide dengan delimiter `;`.
- `../delivery_points_long.csv` — bentuk normalized dari data Pontianak,
  satu baris per titik jalan untuk EDA/geografi.
- `../JalurAI_dataset_EDA.ipynb` — notebook pengambilan, pembersihan,
  pembentukan label, dan EDA kedua lapisan data.

Data Pontianak tidak digabungkan ke setiap order karena grain-nya berbeda:
data tersebut adalah titik/jalan, bukan riwayat order. Model order dilatih
hanya dari order aktif (`is_cancelled=0`).

## Model training

Jalankan dari root repo setelah dependency backend tersedia:

```powershell
python file/train.py
```

Training menghasilkan:

- `models/xgb_classifier.json` — risiko gabungan telat atau over-cost;
- `models/xgb_delay_regressor.json` — estimasi hari keterlambatan untuk order
  yang berlabel telat;
- `models/xgb_cost_regressor.json` — estimasi nominal kelebihan biaya untuk
  order yang berlabel over-cost;
- `models/metadata.json` — urutan fitur, mapping kategori, scaler, dan metrik.

Target-derived columns seperti `prob_telat`, `label_telat`,
`hari_keterlambatan`, `label_kelebihan_biaya`, dan
`nilai_kelebihan_biaya_idr` tidak dipakai sebagai fitur.

## Run

```bash
docker compose up --build
```

API docs: `http://localhost:8000/docs`  
Dashboard: `http://localhost:3000`

## Optional Local Resolver Agent

JalurAI memakai Ollama dengan model `llama3.2` jika tersedia. Jika Ollama
offline atau model belum ada, API otomatis memakai fallback resolver
deterministik.
