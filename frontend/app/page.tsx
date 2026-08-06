"use client";

import "./globals.css";

import {
  ChangeEvent,
  FormEvent,
  useMemo,
  useState,
} from "react";


type FormDataState = {
  origin_city: string;
  dest_city: string;
  weight: string;
  value_of_goods: string;
  estimated_shipping_cost: string;
  qty: string;
  tier_layanan: string;
  kurir: string;
};

type ShapFeature = {
  feature: string;
  impact: number;
  direction: string;
};

type PredictionResult = {
  risk_score: number;
  risk_category: "Normal" | "Risiko Tinggi";
  estimated_extra_cost: number;
  estimated_delay_days: number;
  shap_features: ShapFeature[];
  resolution_narrative: string;
  recommended_action: string;
  created_at: string;
};

type HistoryEntry = {
  id: string;
  timestamp: string;
  origin: string;
  destination: string;
  result: PredictionResult;
};


const initialFormData: FormDataState = {
  origin_city: "Tangerang",
  dest_city: "",
  weight: "",
  value_of_goods: "",
  estimated_shipping_cost: "",
  qty: "1",
  tier_layanan: "Reguler",
  kurir: "SPX",
};

const cityCoordinates: Record<string, readonly [number, number]> = {
  Jakarta: [-6.2088, 106.8456],
  Bandung: [-6.9175, 107.6191],
  Surabaya: [-7.2575, 112.7521],
  Semarang: [-6.9667, 110.4167],
  Yogyakarta: [-7.7978, 110.3695],
  Solo: [-7.5646, 110.8212],
  Malang: [-7.9667, 112.6333],
  Blitar: [-8.0983, 112.1667],
  Tangerang: [-6.1783, 106.6319],
  Bekasi: [-6.2349, 106.9933],
  Depok: [-6.4025, 106.835],
  Bogor: [-6.595, 106.8167],
  Medan: [3.5952, 98.6722],
  Palembang: [-2.9833, 104.7167],
  Makassar: [-5.1477, 119.4327],
  Banjarmasin: [-3.3167, 114.5908],
  Pontianak: [-0.025, 109.3333],
  Manado: [1.4743, 124.8425],
  Padang: [-0.9481, 100.3543],
  Pekanbaru: [0.2667, 101.1333],
  Jambi: [-1.6101, 103.6131],
  Bengkulu: [-3.8004, 102.2656],
  Lampung: [-5.1167, 105.2333],
  Balikpapan: [-1.2667, 116.8],
  Samarinda: [-0.4862, 117.15],
  Ternate: [0.6917, 127.3833],
  Ambon: [-3.6954, 128.1814],
  Mataram: [-8.5833, 116.1167],
  Kupang: [-10.1713, 123.6067],
  Papua: [-2.5337, 140.7181],
};

const cityNames = Object.keys(cityCoordinates);
const cityIslands: Record<string, string> = {
  Jakarta: "JAWA",
  Bandung: "JAWA",
  Surabaya: "JAWA",
  Semarang: "JAWA",
  Yogyakarta: "JAWA",
  Solo: "JAWA",
  Malang: "JAWA",
  Blitar: "JAWA",
  Tangerang: "JAWA",
  Bekasi: "JAWA",
  Depok: "JAWA",
  Bogor: "JAWA",
  Medan: "SUMATERA",
  Palembang: "SUMATERA",
  Padang: "SUMATERA",
  Pekanbaru: "SUMATERA",
  Jambi: "SUMATERA",
  Bengkulu: "SUMATERA",
  Lampung: "SUMATERA",
  Banjarmasin: "KALIMANTAN",
  Pontianak: "KALIMANTAN",
  Balikpapan: "KALIMANTAN",
  Samarinda: "KALIMANTAN",
  Makassar: "SULAWESI",
  Manado: "SULAWESI",
  Ternate: "MALUKU-PAPUA",
  Ambon: "MALUKU-PAPUA",
  Mataram: "BALI-NUSA",
  Kupang: "BALI-NUSA",
  Papua: "MALUKU-PAPUA",
};
const logisticsRouteFactor = 1.25;

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const rupiahFormatter = new Intl.NumberFormat("id-ID", {
  style: "currency",
  currency: "IDR",
  maximumFractionDigits: 0,
});

const dateFormatter = new Intl.DateTimeFormat("id-ID", {
  dateStyle: "medium",
  timeStyle: "short",
});

function haversine(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
) {
  const earthRadiusKm = 6371;
  const toRadians = (degrees: number) => (degrees * Math.PI) / 180;
  const latitudeDelta = toRadians(lat2 - lat1);
  const longitudeDelta = toRadians(lon2 - lon1);
  const startLatitude = toRadians(lat1);
  const endLatitude = toRadians(lat2);

  const haversineValue =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(startLatitude) *
      Math.cos(endLatitude) *
      Math.sin(longitudeDelta / 2) ** 2;
  const angularDistance =
    2 *
    Math.atan2(
      Math.sqrt(haversineValue),
      Math.sqrt(1 - haversineValue),
    );

  return earthRadiusKm * angularDistance;
}

function getDistance(origin: string, destination: string) {
  const originCoordinates = cityCoordinates[origin];
  const destinationCoordinates = cityCoordinates[destination];

  if (!originCoordinates || !destinationCoordinates) {
    return null;
  }
  if (origin === destination) {
    return 10;
  }

  const airDistance = haversine(
    originCoordinates[0],
    originCoordinates[1],
    destinationCoordinates[0],
    destinationCoordinates[1],
  );

  return Math.round(airDistance * logisticsRouteFactor * 10) / 10;
}


export default function DashboardPage() {
  const [formData, setFormData] =
    useState<FormDataState>(initialFormData);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [expandedHistoryId, setExpandedHistoryId] =
    useState<string | null>(null);

  const distanceKm = useMemo(
    () => getDistance(formData.origin_city, formData.dest_city),
    [formData.origin_city, formData.dest_city],
  );
  const sameCity =
    formData.origin_city !== "" &&
    formData.origin_city === formData.dest_city;

  const maxShapImpact = useMemo(
    () =>
      Math.max(
        ...((result?.shap_features ?? []).map((feature) =>
          Math.abs(feature.impact),
        )),
        0.0001,
      ),
    [result],
  );

  function handleChange(
    event: ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) {
    const { name, value } = event.target;
    setFormData((current) => ({ ...current, [name]: value }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (sameCity || distanceKm === null) {
      setError(
        sameCity
          ? "Kota asal dan tujuan tidak boleh sama."
          : "Kombinasi kota tidak tersedia, silahkan pilih kota lain.",
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin_city: formData.origin_city.trim(),
          dest_city: formData.dest_city.trim(),
          dest_island: cityIslands[formData.dest_city] ?? "LAINNYA",
          weight: Number(formData.weight),
          value_of_goods: Number(formData.value_of_goods),
          distance_km: distanceKm,
          estimated_shipping_cost: formData.estimated_shipping_cost
            ? Number(formData.estimated_shipping_cost)
            : Math.round(distanceKm * 2000),
          qty: Number(formData.qty),
          jumlah_kategori: 1,
          tier_layanan: formData.tier_layanan,
          kurir: formData.kurir,
        }),
      });

      if (!response.ok) {
        const payload = await response
          .json()
          .catch(() => ({ detail: "Respons API tidak valid." }));
        throw new Error(
          payload.detail ?? `Permintaan gagal (${response.status}).`,
        );
      }

      const prediction = (await response.json()) as PredictionResult;
      const timestamp = prediction.created_at || new Date().toISOString();
      setResult(prediction);
      setHistory((current) =>
        [
          {
            id: `${timestamp}-${Date.now()}`,
            timestamp,
            origin: formData.origin_city.trim(),
            destination: formData.dest_city.trim(),
            result: prediction,
          },
          ...current,
        ].slice(0, 10),
      );
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "Tidak dapat terhubung ke API JalurAI. Pastikan backend berjalan di port 8000."
          : requestError instanceof Error
          ? requestError.message
          : "Tidak dapat terhubung ke API JalurAI.",
      );
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFormData(initialFormData);
    setResult(null);
    setError(null);
  }

  const isHighRisk = result?.risk_category === "Risiko Tinggi";
  const statusClasses = isHighRisk
    ? "border-red-300 bg-red-100 text-red-800"
    : "border-green-300 bg-green-100 text-green-800";

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-8 text-slate-950 sm:px-6 sm:py-12">
      <div className="mx-auto flex max-w-[640px] flex-col gap-4">
        <header className="mb-2">
          <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Smart Logistics System
          </p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">
            JalurAI
          </h1>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Prediksi risiko sebelum pesanan diberangkatkan dengan fitur order
            nyata dan label operasional tersimulasi.
          </p>
        </header>

        <section
          aria-labelledby="shipment-form-title"
          className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm sm:p-6"
        >
          <div className="mb-5">
            <h2
              id="shipment-form-title"
              className="text-lg font-bold"
            >
              Detail Pesanan
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Lengkapi fitur yang tersedia sebelum dispatch untuk menghitung risiko.
            </p>
          </div>

          <form
            className="grid grid-cols-1 gap-4 sm:grid-cols-2"
            onSubmit={handleSubmit}
          >
            <CitySelect
              label="Kota Asal"
              name="origin_city"
              value={formData.origin_city}
              onChange={handleChange}
              placeholder="Pilih kota asal"
            />
            <CitySelect
              label="Kota Tujuan"
              name="dest_city"
              value={formData.dest_city}
              onChange={handleChange}
              placeholder="Pilih kota tujuan"
            />
            <InputField
              label="Berat"
              name="weight"
              type="number"
              step="0.1"
              min="0.1"
              value={formData.weight}
              onChange={handleChange}
              placeholder="kg"
            />
            <InputField
              label="Kuantitas"
              name="qty"
              type="number"
              step="1"
              min="1"
              value={formData.qty}
              onChange={handleChange}
              placeholder="unit"
            />
            <InputField
              label="Perkiraan ongkir"
              name="estimated_shipping_cost"
              type="number"
              min="0"
              value={formData.estimated_shipping_cost}
              onChange={handleChange}
              placeholder="otomatis bila kosong"
            />
            <InputField
              label="Nilai barang"
              name="value_of_goods"
              type="number"
              min="0"
              value={formData.value_of_goods}
              onChange={handleChange}
              placeholder="Rp"
            />
            <label className="flex flex-col gap-2">
              <span className="text-sm font-semibold text-slate-700">
                Tier layanan
              </span>
              <select
                className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm outline-none transition focus:border-slate-700 focus:ring-2 focus:ring-slate-200"
                name="tier_layanan"
                value={formData.tier_layanan}
                onChange={handleChange}
              >
                <option value="Instan">Instan</option>
                <option value="Same Day">Same Day</option>
                <option value="Next Day">Next Day</option>
                <option value="Reguler">Reguler</option>
                <option value="Hemat">Hemat</option>
                <option value="Kargo">Kargo</option>
              </select>
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-sm font-semibold text-slate-700">
                Kurir
              </span>
              <select
                className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm outline-none transition focus:border-slate-700 focus:ring-2 focus:ring-slate-200"
                name="kurir"
                value={formData.kurir}
                onChange={handleChange}
              >
                <option value="SPX">SPX</option>
                <option value="JNE">JNE</option>
                <option value="J&T">J&amp;T</option>
                <option value="GOSEND">GOSEND</option>
                <option value="GRAB">GRAB</option>
                <option value="LAINNYA">Lainnya</option>
              </select>
            </label>

            <div
              aria-live="polite"
              className={`rounded-lg border px-4 py-3 text-sm sm:col-span-2 ${
                sameCity
                  ? "border-red-300 bg-red-50 font-semibold text-red-800"
                  : "border-slate-200 bg-slate-50 text-slate-600"
              }`}
            >
              {sameCity ? (
                "Kota asal dan tujuan tidak boleh sama."
              ) : distanceKm !== null ? (
                <>
                  Jarak terdeteksi:{" "}
                  <strong className="font-mono text-slate-950">
                    {distanceKm.toLocaleString("id-ID", {
                      maximumFractionDigits: 1,
                    })}{" "}
                    km
                  </strong>
                </>
              ) : formData.origin_city && formData.dest_city ? (
                "Kombinasi kota tidak tersedia, silahkan pilih kota lain."
              ) : (
                "Pilih kota asal dan tujuan untuk menghitung jarak otomatis."
              )}
            </div>

            <button
              className="mt-1 flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-slate-950 px-5 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400 sm:col-span-2"
              type="submit"
              disabled={
                loading ||
                sameCity ||
                distanceKm === null ||
                !formData.origin_city ||
                !formData.dest_city
              }
            >
              {loading ? (
                <>
                  <span
                    aria-hidden="true"
                    className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
                  />
                  Memproses prediksi...
                </>
              ) : (
                "Prediksi Risiko"
              )}
            </button>
          </form>

          {error && (
            <div
              role="alert"
              className="mt-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm font-medium text-red-800"
            >
              {error}
            </div>
          )}
        </section>

        {result && (
          <section
            aria-labelledby="prediction-result-title"
            className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm sm:p-6"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Hasil prediksi
                </p>
                <h2
                  id="prediction-result-title"
                  className="mt-1 text-xl font-bold"
                >
                  Status Pengiriman
                </h2>
              </div>
              <span
                className={`rounded-full border px-4 py-2 text-base font-bold ${statusClasses}`}
              >
                {result.risk_category}
              </span>
            </div>

            <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <MetricCard
                label="Skor Risiko"
                value={result.risk_score.toFixed(4)}
              />
              <MetricCard
                label="Kelebihan Biaya"
                value={rupiahFormatter.format(
                  result.estimated_extra_cost,
                )}
              />
              <MetricCard
                label="Keterlambatan"
                value={`${result.estimated_delay_days} hari`}
              />
            </div>

            <div className="mt-7">
              <h3 className="text-base font-bold">
                Faktor Risiko Teratas
              </h3>
              <div className="mt-4 flex flex-col gap-4">
                {result.shap_features.slice(0, 3).map((feature) => {
                  const positive = feature.impact > 0;
                  const width = Math.max(
                    (Math.abs(feature.impact) / maxShapImpact) * 100,
                    3,
                  );

                  return (
                    <div key={feature.feature}>
                      <div className="mb-2 flex items-center justify-between gap-3 text-xs">
                        <span className="font-semibold text-slate-700">
                          {formatFeatureName(feature.feature)}
                        </span>
                        <span className="font-mono text-slate-600">
                          {feature.impact > 0 ? "+" : ""}
                          {feature.impact.toFixed(4)}
                        </span>
                      </div>
                      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full ${
                            positive ? "bg-red-500" : "bg-green-500"
                          }`}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div
              className={`mt-7 rounded-lg border-l-4 p-4 ${
                isHighRisk
                  ? "border-red-500 bg-red-50"
                  : "border-green-500 bg-green-50"
              }`}
            >
              <h3
                className={`font-bold ${
                  isHighRisk ? "text-red-900" : "text-green-900"
                }`}
              >
                Rekomendasi Tindakan
              </h3>
              <p
                className={`mt-2 text-sm leading-6 ${
                  isHighRisk ? "text-red-800" : "text-green-800"
                }`}
              >
                {result.resolution_narrative}
              </p>
              <p
                className={`mt-3 text-sm font-bold ${
                  isHighRisk ? "text-red-900" : "text-green-900"
                }`}
              >
                {result.recommended_action}
              </p>
            </div>

            <button
              className="mt-5 rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              type="button"
              onClick={handleReset}
            >
              Reset
            </button>
          </section>
        )}

        <section className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <button
            aria-expanded={historyOpen}
            className="flex w-full items-center justify-between px-5 py-4 text-left text-sm font-bold hover:bg-slate-50"
            type="button"
            onClick={() => setHistoryOpen((current) => !current)}
          >
            <span>Riwayat ({history.length})</span>
            <span
              aria-hidden="true"
              className={`text-lg transition-transform ${
                historyOpen ? "rotate-180" : ""
              }`}
            >
              ⌄
            </span>
          </button>

          {historyOpen && (
            <div className="border-t border-slate-200">
              {history.length === 0 ? (
                <p className="px-5 py-6 text-center text-sm text-slate-500">
                  Belum ada prediksi pada sesi ini.
                </p>
              ) : (
                <ul className="divide-y divide-slate-100">
                  {history.map((entry) => {
                    const entryHighRisk =
                      entry.result.risk_category === "Risiko Tinggi";
                    const expanded = expandedHistoryId === entry.id;

                    return (
                      <li key={entry.id}>
                        <button
                          aria-expanded={expanded}
                          className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left hover:bg-slate-50"
                          type="button"
                          onClick={() =>
                            setExpandedHistoryId(
                              expanded ? null : entry.id,
                            )
                          }
                        >
                          <span className="min-w-0">
                            <span className="block truncate text-sm font-bold">
                              {entry.origin} → {entry.destination}
                            </span>
                            <span className="mt-1 block text-xs text-slate-500">
                              {dateFormatter.format(
                                new Date(entry.timestamp),
                              )}
                            </span>
                          </span>
                          <span className="text-right">
                            <span
                              className={`block text-xs font-bold ${
                                entryHighRisk
                                  ? "text-red-700"
                                  : "text-green-700"
                              }`}
                            >
                              {entry.result.risk_category}
                            </span>
                            <span className="mt-1 block font-mono text-xs text-slate-500">
                              {entry.result.risk_score.toFixed(4)}
                            </span>
                          </span>
                        </button>

                        {expanded && (
                          <div className="bg-slate-50 px-5 py-4 text-xs leading-5 text-slate-600">
                            <div className="grid grid-cols-2 gap-3">
                              <span>
                                Biaya:{" "}
                                <strong className="text-slate-900">
                                  {rupiahFormatter.format(
                                    entry.result.estimated_extra_cost,
                                  )}
                                </strong>
                              </span>
                              <span>
                                Terlambat:{" "}
                                <strong className="text-slate-900">
                                  {
                                    entry.result
                                      .estimated_delay_days
                                  }{" "}
                                  hari
                                </strong>
                              </span>
                            </div>
                            <p className="mt-3">
                              {entry.result.recommended_action}
                            </p>
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}


type InputFieldProps = {
  label: string;
  name: keyof FormDataState;
  value: string;
  onChange: (
    event: ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) => void;
  type?: "text" | "number";
  step?: string;
  min?: string;
  placeholder: string;
};

function InputField({
  label,
  name,
  value,
  onChange,
  type = "text",
  step,
  min,
  placeholder,
}: InputFieldProps) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-sm font-semibold text-slate-700">
        {label}
      </span>
      <input
        className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm outline-none transition placeholder:text-slate-400 focus:border-slate-700 focus:ring-2 focus:ring-slate-200"
        name={name}
        type={type}
        step={step}
        min={min}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required
      />
    </label>
  );
}

type CitySelectProps = {
  label: string;
  name: "origin_city" | "dest_city";
  value: string;
  onChange: (
    event: ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) => void;
  placeholder: string;
};

function CitySelect({
  label,
  name,
  value,
  onChange,
  placeholder,
}: CitySelectProps) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-sm font-semibold text-slate-700">
        {label}
      </span>
      <select
        className="h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm outline-none transition focus:border-slate-700 focus:ring-2 focus:ring-slate-200"
        name={name}
        value={value}
        onChange={onChange}
        required
      >
        <option value="" disabled>
          {placeholder}
        </option>
        {cityNames.map((city) => (
          <option key={city} value={city}>
            {city}
          </option>
        ))}
      </select>
    </label>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-2 break-words font-mono text-xl font-bold text-slate-950">
        {value}
      </p>
    </div>
  );
}

function formatFeatureName(feature: string) {
  return feature
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
