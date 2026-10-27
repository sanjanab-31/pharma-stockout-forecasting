"""api.py - FastAPI service that serves risk scores and recommendations (report Section 5.5).

    uvicorn api:app --reload        (run inside the src/ folder, then open http://127.0.0.1:8000/docs)
"""
import json
from fastapi import FastAPI, HTTPException
from config import OUTPUTS
import predict as P

app = FastAPI(title="PharmaTrack Live API", version="1.0",
              description="Real-time pharmaceutical supply-chain stockout forecasting")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/alerts")
def alerts(limit: int = 10):
    """Highest-risk facility-medicine pairs on the latest snapshot."""
    return P.top_risks(limit)


@app.get("/risk/{facility_id}/{drug_id}")
def risk(facility_id: str, drug_id: str):
    try:
        return P.predict(facility_id.upper(), drug_id.upper())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics")
def metrics():
    m = json.load(open(OUTPUTS / "metrics.json"))
    return {"classification": m["classification"], "simulation": json.load(open(OUTPUTS / "simulation.json"))}
