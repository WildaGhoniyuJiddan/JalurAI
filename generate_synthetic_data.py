"""Generate synthetic Indonesian logistics shipment data for JalurAI ML model training.

Distributions derived from guidebook & research values:
  - Jawa vs Luar-Jawa cost disparity: 20-40% (Suara.com, 2026)
  - E-commerce contribution to digital economy: ~77% (Suara.com, 2026)
  - Daily packages processed: ~25M (Suara.com, 2026)
  - Target Rlog: 14.29% -> 8% by 2045 (Kemenhub)
"""

import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)

# --- Domain Constants ---
ORIGINS_JAWA = [
    "Jakarta", "Bandung", "Surabaya", "Semarang", "Yogyakarta",
    "Solo", "Malang", "Blitar", "Tangerang", "Bekasi", "Depok", "Bogor"
]
ORIGINS_OUTER_JAWA = [
    "Medan", "Palembang", "Makassar", "Banjarmasin", "Pontianak",
    "Manado", "Padang", "Pekanbaru", "Jambi", "Bengkulu", "Lampung",
    "Balikpapan", "Samarinda", "Ternate", "Ambon", "Mataram", "Kupang"
]

DESTINATIONS_JAWA = ORIGINS_JAWA[:]
DESTINATIONS_OUTER_JAWA = ORIGINS_OUTER_JAWA[:]

CARRIERS = ["JNE", "J&T Express", "SiCepat", "TIKI", "Anteraja", "Ninja Xpress", "JX Express", "Wahana"]
ARMADA_TYPES = ["truk", "kapal_laut", "pesawat", "motor"]

# Weight (kg): log-normal centered around 2-5kg for typical e-commerce
def sample_weight():
    return round(random.lognormvariate(math.log(3.0), 0.6), 2)

def sample_volume(weight):
    # 0.3-0.6 cubic meter per kg typical, with noise
    return round(weight * random.uniform(0.3, 0.6), 4)

def sample_value(weight):
    # Rp 50k-500k per kg with some outliers
    base = random.uniform(80000, 150000)
    return round(weight * base * random.uniform(0.7, 1.3), 0)

def is_jawa_city(city):
    return city in ORIGINS_JAWA or city in DESTINATIONS_JAWA

def base_cost_per_km(origin, dest):
    """Base cost model: cheaper inside Jawa, higher for cross-island."""
    if is_jawa_city(origin) and is_jawa_city(dest):
        return random.uniform(800, 1800)  # Rp per km, intra-Jawa cheaper
    elif is_jawa_city(origin) or is_jawa_city(dest):
        return random.uniform(2500, 5000)  # Cross-island outbound
    else:
        return random.uniform(1500, 3500)  # Outer-Jawa to Outer-Jawa

def distance_km(origin, dest):
    if origin == dest:
        return random.uniform(5, 30)
    if (is_jawa_city(origin) and is_jawa_city(dest)):
        return random.uniform(50, 800)
    elif (is_jawa_city(origin) or is_jawa_city(dest)):
        return random.uniform(500, 2500)
    else:
        return random.uniform(200, 1800)

def shipment_delay_prob(weight, dist, carrier):
    """Probability of delay influenced by weight, distance, carrier reliability."""
    base = 0.05  # 5% base delay rate
    base += min(dist / 3000, 0.20)  # Longer distance = higher delay risk
    base += min(weight / 20, 0.10)  # Heavy items slightly more risky
    # Carrier quality varies
    carrier_risk = {"JNE": 0.03, "J&T Express": 0.05, "SiCepat": 0.06, "TIKI": 0.04,
                    "Anteraja": 0.05, "Ninja Xpress": 0.07, "JX Express": 0.04, "Wahana": 0.05}
    base += carrier_risk.get(carrier, 0.05)
    return min(base, 0.50)

def generate_delay_days(prob):
    """Generate delay in days based on probability."""
    if random.random() < prob:
        return round(random.expovariate(1.0 / 3.0), 1)  # avg 3 days delay
    return 0.0

def generate_extra_cost_pct(dist, prob_delay):
    """Extra cost percentage (over base rate) based on route difficulty."""
    base_extra = prob_delay * random.uniform(1.15, 1.40)  # 15-40% overage when delayed
    route_penalty = min(dist / 2000, 0.25)  # Long routes add up to 25% baseline extra
    return round((base_extra + route_penalty) * 100, 2)

# --- Main Generation ---
N_SAMPLES = 5000
OUTPUT_PATH = "E:/AIC/synthetic_shipments.csv"

rows = []
for i in range(N_SAMPLES):
    # Sample origin/destination pair - avoid same if unrealistic
    origin = random.choice(ORIGINS_JAWA + ORIGINS_OUTER_JAWA)
    dest = random.choice(DESTINATIONS_JAWA + DESTINATIONS_OUTER_JAWA)
    while dest == origin:
        dest = random.choice(DESTINATIONS_JAWA + DESTINATIONS_OUTER_JAWA)

    carrier = random.choice(CARRIERS)
    armada = random.choice(ARMADA_TYPES)
    weight = sample_weight()
    volume = sample_volume(weight)
    value_goods = sample_value(weight)
    dist = distance_km(origin, dest)
    base_cost_km = base_cost_per_km(origin, dest)
    base_shipping_cost = round(dist * base_cost_km, 0)

    delay_prob = shipment_delay_prob(weight, dist, carrier)
    delay_days = generate_delay_days(delay_prob)
    extra_cost_pct = generate_extra_cost_pct(dist, delay_prob) if delay_days > 0 else random.uniform(0, 5)
    extra_cost_amount = round(base_shipping_cost * (extra_cost_pct / 100), 0)

    # Target variable: risk flag (high risk = delay OR extra cost > 15%)
    is_high_risk = 1 if (delay_days > 0 or extra_cost_pct > 15) else 0

    # Feature: ratio of shipping cost to goods value (proxy for loss sensitivity)
    cost_value_ratio = round(base_shipping_cost / max(value_goods, 10000), 4)
    # Feature: distance per kg (intensity metric)
    dist_per_kg = round(dist / max(weight, 0.1), 2)
    # Cost per km adjusted for armada type
    armada_multiplier = {"truk": 1.0, "kapal_laut": 0.6, "pesawat": 2.5, "motor": 0.4}
    effective_cost = base_shipping_cost * armada_multiplier.get(armada, 1.0)

    # Date within 6 months prior (realistic historical window)
    days_ago = random.randint(1, 180)
    ship_date = (datetime(2026, 6, 15) - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    rows.append({
        "shipment_id": f"SHP-{i+1:05d}",
        "ship_date": ship_date,
        "origin_city": origin,
        "dest_city": dest,
        "carrier": carrier,
        "armada_type": armada,
        "weight_kg": weight,
        "volume_m3": volume,
        "value_goods_rp": int(value_goods),
        "distance_km": round(dist, 1),
        "base_shipping_cost_rp": int(base_shipping_cost),
        "cost_value_ratio": cost_value_ratio,
        "dist_per_kg": dist_per_kg,
        "delay_days": delay_days,
        "extra_cost_pct": extra_cost_pct,
        "extra_cost_amount_rp": int(extra_cost_amount),
        "is_high_risk": is_high_risk,
        "jawa_origin": 1 if is_jawa_city(origin) else 0,
        "jawa_dest": 1 if is_jawa_city(dest) else 0,
        "cross_island": 1 if (is_jawa_city(origin) != is_jawa_city(dest)) else 0,
    })

# Write CSV
fieldnames = list(rows[0].keys())
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> {OUTPUT_PATH}")

# Quick stats
import collections
risk_counts = collections.Counter(r["is_high_risk"] for r in rows)
print(f"Risk distribution: {dict(risk_counts)}")
print(f"Avg delay (high-risk): {sum(r['delay_days'] for r in rows if r['is_high_risk'] > 0) / max(risk_counts[1], 1):.2f} days")
print(f"Avg extra cost pct (high-risk): {sum(r['extra_cost_pct'] for r in rows if r['is_high_risk'] > 0) / max(risk_counts[1], 1):.2f}%")
