# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="pants-backend")

# 允許前端（Github Pages）跨網域呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 如果之後有自訂網域，可改成你的網域
    allow_methods=["*"],
    allow_headers=["*"],
)

# 一些假資料，先讓前端可以通
SAMPLE_ITEMS = [
    {
        "tumor": 1,
        "sex": "F",
        "age": 66,
        "ct phase": "Non-contrast",
        "manufacturer": "SIEMENS",
        "manufacturer model": "Sensation 16",
        "study year": "2000",
        "PanTS ID": "PanTS_00000001",
        "case_id": "PanTS_00000001",
    },
    {
        "tumor": 0,
        "sex": "M",
        "age": 60,
        "ct phase": "Venous",
        "manufacturer": "SIEMENS",
        "manufacturer model": "somatom definition edge",
        "study year": "2012",
        "PanTS ID": "PanTS_00000002",
        "case_id": "PanTS_00000002",
    },
]

@app.get("/")
def root():
    return {"ok": True, "service": "pants-backend"}

@app.get("/api/search")
def search(page: int = 1, per_page: int = 20):
    total = len(SAMPLE_ITEMS)
    return {"items": SAMPLE_ITEMS, "total": total, "page": 1, "pages": 1}

@app.get("/api/facets")
def facets(fields: str = "", top_k: int = 10, guarantee: int = 1):
    return {
        "facets": {
            "ct_phase": [
                {"value": "Non-contrast", "count": 100},
                {"value": "Venous", "count": 80},
                {"value": "Arterial", "count": 60},
            ],
            "manufacturer": [
                {"value": "SIEMENS", "count": 120},
                {"value": "GE MEDICAL SYSTEMS", "count": 70},
                {"value": "Philips", "count": 50},
            ],
        },
        "year_range": {"min": 2010, "max": 2024},
    }

@app.get("/api/random")
def random_endpoint(scope: str = "filtered", n: int = 3):
    return {"items": SAMPLE_ITEMS[:n]}
