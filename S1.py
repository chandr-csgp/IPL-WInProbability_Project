from pathlib import Path
import json

DATA_ROOT = Path.home() / "Documents" / "DSP" / "ipl_json"

print("[Preflight] data_root:", DATA_ROOT)
assert DATA_ROOT.exists(), "Dataset folder not found."

files = sorted(DATA_ROOT.glob("*.json"))
print("[Preflight] .json files found:", len(files))
assert len(files) > 0, "No JSON files found."

# Peek one file
with open(files[0], "r", encoding="utf-8") as f:
    sample = json.load(f)

print("[Preflight] top-level keys:", list(sample.keys()))
assert "info" in sample and "innings" in sample, "Unexpected JSON structure."

info = sample["info"]
print("[Preflight] sample info keys:", list(info.keys())[:12])
print("[Preflight] dates field:", info.get("dates"))
print("[Preflight] teams field:", info.get("teams"))
print("[Preflight] venue field:", info.get("venue"))
print("[Preflight] city field:", info.get("city"))
print("[Preflight] outcome:", info.get("outcome"))
print("[Preflight] sample innings keys:", list(sample["innings"][0].keys()))