"use client";

import { FormEvent, useState } from "react";

export default function Dashboard() {
  const [result, setResult] = useState<unknown>(null);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = Object.fromEntries(form.entries());
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, weight: Number(payload.weight), volume: Number(payload.volume), value_of_goods: Number(payload.value_of_goods) }),
    });
    setResult(await response.json());
  }
  return <main><h1>JalurAI</h1><p>Dashboard Prediksi Risiko Logistik</p><form onSubmit={submit}>
    <input name="origin_city" placeholder="Kota asal" required /><input name="dest_city" placeholder="Kota tujuan" required />
    <input name="weight" type="number" step="0.1" placeholder="Berat (kg)" required /><input name="volume" type="number" step="0.1" placeholder="Volume" required />
    <input name="carrier_type" placeholder="Jenis armada" required /><input name="value_of_goods" type="number" placeholder="Nilai barang" required /><button>Prediksi Risiko</button>
  </form>{result && <pre>{JSON.stringify(result, null, 2)}</pre>}</main>;
}
