'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/navBar';

export default function Home() {
  const router = useRouter(); 
  const [formData, setFormData] = useState({
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: ''
  });

  const [result, setResult] = useState<{ crop: string; confidence: number } | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const token = localStorage.getItem("token");

    if (!token) {
      alert("Login required.");
      return;
    }

    const parsedData = {
      N: parseFloat(formData.nitrogen),
      P: parseFloat(formData.phosphorus),
      K: parseFloat(formData.potassium),
      temperature: parseFloat(formData.temperature),
      humidity: parseFloat(formData.humidity),
      ph: parseFloat(formData.ph),
      rainfall: parseFloat(formData.rainfall)
    };

    try {
      const response = await fetch('http://10.49.12.49:1010/predict-simple', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(parsedData)
      });

      const result = await response.json();

      if (result.success) {
        setResult({
          crop: result.predicted_crop,
          confidence: result.confidence
        });
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error al llamar a la API:', error);
      alert('Error al conectar con la API');
    }
  };

  return (
    <div className="min-h-screen flex">
      <Navbar />
      <main className="flex-1 bg-gradient-to-br from-emerald-200/80 via-white to-emerald-50 flex items-center justify-center p-12">
        <div className="bg-white/80 backdrop-blur-md rounded-xl shadow-lg border border-emerald-300 w-full max-w-6xl p-10">
          <h1 className="text-4xl font-bold text-emerald-900 mb-6 text-center">Predición de Cultivo</h1>

          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {[
              { name: 'nitrogen', label: 'Nitrógeno (N)', step: '1', placeholder: '90' },
              { name: 'phosphorus', label: 'Fósforo (P)', step: '1', placeholder: '42' },
              { name: 'potassium', label: 'Potasio (K)', step: '1', placeholder: '43' },
              { name: 'temperature', label: 'Temperatura (°C)', step: '0.1', placeholder: '24.5' },
              { name: 'humidity', label: 'Humedad (%)', step: '0.1', placeholder: '82' },
              { name: 'ph', label: 'Nivel de pH', step: '0.01', placeholder: '6.5' },
              { name: 'rainfall', label: 'Precipitación (mm)', step: '0.1', placeholder: '202' }
            ].map(({ name, label, step, placeholder }) => (
              <div key={name}>
                <label htmlFor={name} className="block text-sm font-medium text-emerald-900 mb-1">
                  {label}
                </label>
                <input
                  id={name}
                  name={name}
                  type="number"
                  step={step}
                  placeholder={placeholder}
                  value={formData[name as keyof typeof formData]}
                  onChange={handleChange}
                  className="w-full p-2 border border-emerald-300 rounded-md shadow-sm focus:ring-2 focus:ring-emerald-500 focus:outline-none bg-white placeholder-gray-400"
                  required
                />
              </div>
            ))}

            <div className="md:col-span-2 text-right">
              <button
                type="submit"
                className="bg-emerald-800/80 hover:bg-emerald-800 text-white font-semibold px-6 py-2 rounded-lg shadow transition"
              >
                Predecir cultivo
              </button>
            </div>
          </form>
          <div className="border border-emerald-300 rounded-lg p-4 bg-white/60 shadow-inner text-center min-h-[80px] flex flex-col items-center justify-center transition-all duration-300">
            {result ? (
              <>
                <p className="text-lg text-emerald-900 font-semibold">
                  Recommended Crop: <span className="font-bold text-emerald-700">{result.crop}</span>
                </p>
                <p className="text-sm text-emerald-800">
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </p>
              </>
            ) : (
              <p className="text-sm text-gray-400 italic">La predicción aparecerá aquí después de enviar el formulario.</p>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
