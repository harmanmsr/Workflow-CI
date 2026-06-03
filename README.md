# Workflow-CI — KMeans Womens Shoes

Repository ini berisi MLflow Project untuk re-training model KMeans clustering sepatu wanita secara otomatis menggunakan GitHub Actions CI.

## Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── MLProject/
│   ├── modelling.py            # Script training utama
│   ├── conda.yaml              # Environment dependencies
│   ├── MLProject               # MLflow Project config
│   └── data/
│       └── Womens_Shoes_Clean.csv
└── README.md
```

## Cara Menjalankan

### Lokal
```bash
mlflow run MLProject -P k_optimal=4 -P random_state=42 --env-manager=local
```

### Via GitHub Actions
Workflow akan otomatis berjalan ketika:
- Push ke branch `main` pada folder `MLProject/`
- Pull request ke branch `main`
- Manual trigger via **Actions** → **Run workflow**

## Parameter

| Parameter | Default | Keterangan |
|---|---|---|
| `k_optimal` | 4 | Jumlah cluster K |
| `random_state` | 42 | Random seed |
| `data_path` | `data/Womens_Shoes_Clean.csv` | Path dataset |

## Docker Hub
Image tersedia di: `docker pull <username>/kmeans-womens-shoes:latest`

## MLflow Tracking
Set environment variable sebelum menjalankan:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```
