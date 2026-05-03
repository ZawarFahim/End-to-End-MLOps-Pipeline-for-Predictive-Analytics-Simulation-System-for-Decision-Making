import urllib.request, json, sys

base = "http://127.0.0.1:8000"

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

# TEST 1 — Multan (arid) coffee must NOT appear
print("TEST 1: Multan  temp=30 rain=150 hum=40  [arid]")
r = post("/classify-yield", {"temperature": 30, "rainfall": 150, "humidity": 40})
print(json.dumps(r, indent=2))
assert r["region_type"] == "arid", f"Wrong region: {r['region_type']}"
assert "coffee" not in r["top_crops"], f"FAIL coffee in arid: {r['top_crops']}"
assert r["risk_level"] in ("low", "medium", "high")
assert "climate_score" in r
print("  PASS - coffee excluded, region=arid\n")

# TEST 2 — Manaus (tropical)
print("TEST 2: Manaus  temp=27 rain=2300 hum=85  [tropical]")
r = post("/classify-yield", {"temperature": 27, "rainfall": 2300, "humidity": 85})
print(json.dumps(r, indent=2))
assert r["region_type"] == "tropical"
print("  PASS - tropical region\n")

# TEST 3 — London (temperate) coffee must NOT appear
print("TEST 3: London  temp=12 rain=600 hum=76  [temperate]")
r = post("/classify-yield", {"temperature": 12, "rainfall": 600, "humidity": 76})
print(json.dumps(r, indent=2))
assert r["region_type"] == "temperate"
assert "coffee" not in r["top_crops"], f"FAIL coffee in temperate: {r['top_crops']}"
print("  PASS - coffee excluded, region=temperate\n")

# TEST 4 — Cluster untouched
print("TEST 4: /cluster")
r = post("/cluster", {"samples": [{"rainfall": 620, "temperature": 25, "N": 88, "P": 45, "K": 42}]})
assert "clusters" in r
print(f"  PASS: {r}\n")

# TEST 5 — Forecast untouched
print("TEST 5: /forecast")
r = post("/forecast", {"city": "Lahore"})
assert "forecast" in r
print(f"  PASS: {r}\n")

print("ALL 5 TESTS PASSED")
