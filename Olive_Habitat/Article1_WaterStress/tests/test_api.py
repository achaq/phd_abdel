"""
OliveGuard API - Simple Endpoint Tests
======================================
Run this script to verify the API is working.

Usage:
  1. Start the API:  uvicorn api.main:app --reload --port 8000
  2. In another terminal:  python tests/test_api.py

Or with pytest:  pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests

BASE_URL = "http://localhost:8000"


def test_root():
    """GET / - Returns API info."""
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "name" in data, "Response should have 'name'"
    assert "version" in data, "Response should have 'version'"
    assert data["name"] == "OliveGuard API", f"Unexpected name: {data['name']}"
    print("  GET /         -> OK")
    return data


def test_health():
    """GET /health - Returns health status."""
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", f"Expected status=ok, got {data}"
    print("  GET /health   -> OK")
    return data


def test_docs_available():
    """GET /docs - Swagger UI should be reachable."""
    r = requests.get(f"{BASE_URL}/docs")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print("  GET /docs     -> OK")


def test_analyze():
    """POST /analyze - Accepts geometry, returns AnalysisResult."""
    # Ghafsai bounding box (small polygon)
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-4.85, 34.65],
                    [-4.80, 34.65],
                    [-4.80, 34.70],
                    [-4.85, 34.70],
                    [-4.85, 34.65],
                ]
            ],
        }
    }
    r = requests.post(f"{BASE_URL}/analyze", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "anomalyScore" in data
    assert "ndviSeries" in data
    assert "idealSeries" in data
    assert data["stressStatus"] in ("Healthy", "Warning", "Critical")
    assert 0 <= data["anomalyScore"] <= 1
    assert len(data["ndviSeries"]) > 0
    print("  POST /analyze -> OK")


def test_compare():
    """POST /research/compare - Comparative analysis."""
    payload = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-4.85, 34.65], [-4.80, 34.65], [-4.80, 34.70], [-4.85, 34.70], [-4.85, 34.65]]]
        },
        "models": ["LSTM-AE", "Heuristic-NDWI", "Statistical-ZScore"]
    }
    r = requests.post(f"{BASE_URL}/research/compare", json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert "agreement_score" in data
    print("  POST /research/compare -> OK")


def run_all():
    """Run all tests and print summary."""
    print("\n" + "=" * 50)
    print("OliveGuard API - Endpoint Tests")
    print("=" * 50)
    print(f"\nBase URL: {BASE_URL}")
    print("\nRunning tests...\n")

    try:
        root_data = test_root()
        test_health()
        test_docs_available()
        test_analyze()
        test_compare()

        print("\n" + "-" * 50)
        print("All tests passed!")
        print("-" * 50)
        print(f"\nAPI Info: {root_data['name']} v{root_data['version']}")
        print(f"Docs:    {BASE_URL}/docs")
        print()
        return 0

    except requests.exceptions.ConnectionError:
        print("\n  ERROR: Could not connect to the API.")
        print("  Make sure the server is running:")
        print("    uvicorn api.main:app --reload --port 8000")
        print()
        return 1

    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit(run_all())
