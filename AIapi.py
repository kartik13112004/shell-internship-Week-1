# AIapi.py
import os
import requests
from typing import Dict

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
HF_KEY = os.getenv("HF_API_KEY")
HF_ROUTER = "https://router.huggingface.co/v1"
HF_MODEL_DEFAULT = "google/flan-t5-large"
GEMINI_MODEL_DEFAULT = "gemini-1.5"

class EVAIAssistant:
    def __init__(self, gemini_model: str = GEMINI_MODEL_DEFAULT, hf_model: str = HF_MODEL_DEFAULT, temperature: float = 0.7):
        self.gemini_model = gemini_model
        self.hf_model = hf_model
        self.temperature = temperature
        self.hf_headers = {"Authorization": f"Bearer {HF_KEY}"} if HF_KEY else {}
        self.gemini_headers = {"Authorization": f"Bearer {GEMINI_KEY}"} if GEMINI_KEY else {}

    def _fallback(self, prompt: str) -> str:
        p = (prompt or "").lower()
        if "maintenance" in p:
            return ("Battery maintenance tips:\n"
                    "1. Avoid frequent 0–100% cycles; keep daily charging around 20–80%.\n"
                    "2. Use slow AC charging when possible.\n"
                    "3. Avoid extreme temperatures.\n")
        if "charging" in p or "charge" in p:
            return ("Charging strategy:\n- Charge at home overnight during off-peak hours.\n"
                    "- Use fast chargers sparingly on long trips.\n"
                    "- Keep daily charging around 20–80%.\n")
        if "compare" in p:
            return ("Comparison: Compare range, price, and efficiency. "
                    "High range = good for long trips.\n")
        return ("EV range = battery kWh × efficiency km/kWh. "
                "Higher values mean longer range.\n")

    def _call_gemini(self, prompt: str, max_tokens: int = 300) -> str:
        if not GEMINI_KEY:
            raise RuntimeError("No Gemini key found.")
        url = f"https://generativelanguage.googleapis.com/v1beta2/models/{self.gemini_model}:generateText"
        body = {
            "prompt": {"text": prompt},
            "temperature": float(self.temperature),
            "max_output_tokens": int(max_tokens)
        }
        try:
            resp = requests.post(url, headers={"Authorization": f"Bearer {GEMINI_KEY}",
                                               "Content-Type":"application/json"},
                                 json=body, timeout=15)
            resp.raise_for_status()
            j = resp.json()
            if "candidates" in j and j["candidates"]:
                cand = j["candidates"][0]
                for k in ("output","content","text"):
                    if k in cand:
                        return str(cand[k])
            for k in ("output","generated_text","content"):
                if k in j:
                    return str(j[k])
            return str(j)[:1500]
        except Exception as e:
            return f"(Gemini error: {e})\n" + self._fallback(prompt)

    def _call_hf(self, prompt: str, max_tokens: int = 250) -> str:
        if not HF_KEY:
            raise RuntimeError("No HF key found.")
        url = f"{HF_ROUTER}/models/{self.hf_model}"
        payload = {"inputs": prompt,
                   "parameters": {"max_new_tokens": max_tokens,
                                  "temperature": float(self.temperature)}}
        try:
            r = requests.post(url, headers=self.hf_headers, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return str(data)[:1500]
        except Exception as e:
            return f"(HF error: {e})\n" + self._fallback(prompt)

    def generate_text(self, prompt: str, max_tokens: int = 250) -> str:
        if GEMINI_KEY:
            try:
                return self._call_gemini(prompt, max_tokens=max_tokens)
            except:
                pass
        if HF_KEY:
            try:
                return self._call_hf(prompt, max_tokens=max_tokens)
            except:
                pass
        return self._fallback(prompt)

    def recommend_vehicle(self, specs: Dict, predicted_range_km: float) -> str:
        prompt = (f"Specs: {specs}, predicted range {predicted_range_km} km. "
                  f"Write 3 lines recommendation.")
        return self.generate_text(prompt)

    def maintenance_tips(self, specs: Dict) -> str:
        prompt = f"Give 4 maintenance tips for EV with specs {specs}."
        return self.generate_text(prompt)

    def charging_strategy(self, daily_km: float, predicted_range_km: float) -> str:
        prompt = (f"Daily km: {daily_km}, Range: {predicted_range_km}. "
                  f"Give charging strategy and best practices.")
        return self.generate_text(prompt)
