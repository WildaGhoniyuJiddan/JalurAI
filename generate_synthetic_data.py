"""Generate reproducible synthetic logistics data for JalurAI.

The risk label is sampled from a latent probability built only from
information available before shipment: dimensions, value, route, and armada.
Post-shipment outcomes such as delay and extra cost are never model inputs.
"""

import collections
import csv
import math
import random
from datetime import datetime, timedelta


RANDOM_SEED = 42
N_SAMPLES = 5000
OUTPUT_PATH = "E:/AIC/synthetic_shipments.csv"

random.seed(RANDOM_SEED)

ORIGINS_JAWA = [
    "Jakarta",
    "Bandung",
    "Surabaya",
    "Semarang",
    "Yogyakarta",
    "Solo",
    "Malang",
    "Blitar",
    "Tangerang",
    "Bekasi",
    "Depok",
    "Bogor",
]
ORIGINS_OUTER_JAWA = [
    "Medan",
    "Palembang",
    "Makassar",
    "Banjarmasin",
    "Pontianak",
    "Manado",
    "Padang",
    "Pekanbaru",
    "Jambi",
    "Bengkulu",
    "Lampung",
    "Balikpapan",
    "Samarinda",
    "Ternate",
    "Ambon",
    "Mataram",
    "Kupang",
]

DESTINATIONS_JAWA = ORIGINS_JAWA[:]
DESTINATIONS_OUTER_JAWA = ORIGINS_OUTER_JAWA[:]

CARRIERS = [
    "JNE",
    "J&T Express",
    "SiCepat",
    "TIKI",
    "Anteraja",
    "Ninja Xpress",
    "JX Express",
    "Wahana",
]
ARMADA_TYPES = ["truk", "kapal_laut", "pesawat", "motor"]
ARMADA_RISK_EFFECT = {
    "truk": 0.15,
    "kapal_laut": 0.35,
    "pesawat": -0.25,
    "motor": 0.45,
}
ARMADA_COST_EFFECT = {
    "truk": 2.0,
    "kapal_laut": 4.0,
    "pesawat": 8.0,
    "motor": 3.0,
}


def sample_weight():
    return round(random.lognormvariate(math.log(3.0), 0.6), 2)


def sample_volume(weight):
    return round(weight * random.uniform(0.3, 0.6), 4)


def sample_value(weight):
    base = random.uniform(80000, 150000)
    return round(weight * base * random.uniform(0.7, 1.3), 0)


def is_jawa_city(city):
    return city in ORIGINS_JAWA


def distance_km(origin, dest):
    origin_is_jawa = is_jawa_city(origin)
    dest_is_jawa = is_jawa_city(dest)

    if origin == dest:
        return random.uniform(5, 30)
    if origin_is_jawa and dest_is_jawa:
        return random.uniform(50, 800)
    if origin_is_jawa != dest_is_jawa:
        return random.uniform(500, 2500)
    return random.uniform(200, 1800)


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def shipment_risk_probability(
    weight,
    volume,
    distance,
    cost_value_ratio,
    dist_per_kg,
    cross_island,
    armada,
):
    """Estimate latent pre-shipment risk from observable features."""
    distance_signal = min(distance / 2500, 1.0)
    weight_signal = min(weight / 15, 1.0)
    volume_signal = min(volume / 6, 1.0)
    cost_pressure = min(cost_value_ratio / 10, 1.0)
    distance_intensity = min(dist_per_kg / 800, 1.0)

    log_odds = (
        -4.0
        + 2.2 * distance_signal
        + 0.9 * weight_signal
        + 0.5 * volume_signal
        + 2.3 * cross_island
        + 0.5 * cost_pressure
        + 0.4 * distance_intensity
        + ARMADA_RISK_EFFECT[armada]
    )
    return min(max(sigmoid(log_odds), 0.02), 0.95)


def generate_delay_days(risk_probability, cross_island):
    base_delay = 0.5 + 5.0 * risk_probability + 1.5 * cross_island
    return round(max(random.gauss(base_delay, 0.8), 0.5), 1)


def generate_extra_cost_pct(
    risk_probability,
    distance,
    weight,
    cross_island,
    armada,
):
    distance_signal = min(distance / 2500, 1.0)
    weight_signal = min(weight / 15, 1.0)
    expected_extra = (
        5.0
        + 18.0 * risk_probability
        + 10.0 * distance_signal
        + 6.0 * weight_signal
        + 8.0 * cross_island
        + ARMADA_COST_EFFECT[armada]
    )
    return round(max(random.gauss(expected_extra, 3.0), 0.0), 2)


rows = []
for index in range(N_SAMPLES):
    origin = random.choice(ORIGINS_JAWA + ORIGINS_OUTER_JAWA)
    dest = random.choice(DESTINATIONS_JAWA + DESTINATIONS_OUTER_JAWA)
    while dest == origin:
        dest = random.choice(
            DESTINATIONS_JAWA + DESTINATIONS_OUTER_JAWA
        )

    carrier = random.choice(CARRIERS)
    armada = random.choice(ARMADA_TYPES)
    weight = sample_weight()
    volume = sample_volume(weight)
    value_goods = sample_value(weight)
    distance = distance_km(origin, dest)

    jawa_origin = int(is_jawa_city(origin))
    jawa_dest = int(is_jawa_city(dest))
    cross_island = int(jawa_origin != jawa_dest)

    # Keep training features aligned with backend build_feature_vector().
    base_shipping_cost = round(distance * 2000, 0)
    cost_value_ratio = round(
        base_shipping_cost / max(value_goods, 10000), 4
    )
    dist_per_kg = round(distance / max(weight, 0.1), 2)

    risk_probability = shipment_risk_probability(
        weight,
        volume,
        distance,
        cost_value_ratio,
        dist_per_kg,
        cross_island,
        armada,
    )
    is_high_risk = int(random.random() < risk_probability)

    if is_high_risk:
        delay_days = generate_delay_days(
            risk_probability, cross_island
        )
        extra_cost_pct = generate_extra_cost_pct(
            risk_probability,
            distance,
            weight,
            cross_island,
            armada,
        )
    else:
        delay_days = 0.0
        extra_cost_pct = round(random.uniform(0, 5), 2)

    extra_cost_amount = round(
        base_shipping_cost * extra_cost_pct / 100, 0
    )
    days_ago = random.randint(1, 180)
    ship_date = (
        datetime(2026, 6, 15) - timedelta(days=days_ago)
    ).strftime("%Y-%m-%d")

    rows.append(
        {
            "shipment_id": f"SHP-{index + 1:05d}",
            "ship_date": ship_date,
            "origin_city": origin,
            "dest_city": dest,
            "carrier": carrier,
            "armada_type": armada,
            "weight_kg": weight,
            "volume_m3": volume,
            "value_goods_rp": int(value_goods),
            "distance_km": round(distance, 1),
            "base_shipping_cost_rp": int(base_shipping_cost),
            "cost_value_ratio": cost_value_ratio,
            "dist_per_kg": dist_per_kg,
            "delay_days": delay_days,
            "extra_cost_pct": extra_cost_pct,
            "extra_cost_amount_rp": int(extra_cost_amount),
            "is_high_risk": is_high_risk,
            "jawa_origin": jawa_origin,
            "jawa_dest": jawa_dest,
            "cross_island": cross_island,
        }
    )


fieldnames = list(rows[0].keys())
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

risk_counts = collections.Counter(row["is_high_risk"] for row in rows)
high_risk_rows = [
    row for row in rows if row["is_high_risk"] == 1
]

print(f"Generated {len(rows)} rows -> {OUTPUT_PATH}")
print(f"Risk distribution: {dict(risk_counts)}")
print(
    "Avg delay (high-risk): "
    f"{sum(row['delay_days'] for row in high_risk_rows) / len(high_risk_rows):.2f} days"
)
print(
    "Avg extra cost pct (high-risk): "
    f"{sum(row['extra_cost_pct'] for row in high_risk_rows) / len(high_risk_rows):.2f}%"
)
