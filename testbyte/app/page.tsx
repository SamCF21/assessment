'use client';
import { useState } from 'react';

export default function Home() {
  const [formData, setFormData] = useState({
    nitrogen: '',
    phosphorus: '',
    potassium: '',
    temperature: '',
    humidity: '',
    ph: '',
    rainfall: ''
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsedData = {
      nitrogen: parseInt(formData.nitrogen, 10),
      phosphorus: parseInt(formData.phosphorus, 10),
      potassium: parseInt(formData.potassium, 10),
      temperature: parseFloat(formData.temperature),
      humidity: parseFloat(formData.humidity),
      ph: parseFloat(formData.ph),
      rainfall: parseFloat(formData.rainfall)
    };

    const inputArray = [
      parsedData.nitrogen,
      parsedData.phosphorus,
      parsedData.potassium,
      parsedData.temperature,
      parsedData.humidity,
      parsedData.ph,
      parsedData.rainfall
    ];

    console.log('Input:', inputArray)

  };

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-green-900 text-white p-6">
        <h2 className="text-2xl font-bold mb-4">CropAI</h2>
        <nav>
          <ul className="space-y-2">
            <li><a href="/login" className="hover:underline">Login</a></li>
            <li><a href="#" className="hover:underline">Predict</a></li>
          </ul>
        </nav>
      </aside>
      <main className="flex-1 p-10 bg-gray-50">
        <h1 className="text-3xl font-bold mb-6">Enter Soil & Climate Data</h1>
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl">
          {[
            { name: 'nitrogen', label: 'Nitrogen (N)', step: '1' },
            { name: 'phosphorus', label: 'Phosphorus (P)', step: '1' },
            { name: 'potassium', label: 'Potassium (K)', step: '1' },
            { name: 'temperature', label: 'Temperature (°C)', step: '0.1' },
            { name: 'humidity', label: 'Humidity (%)', step: '0.1' },
            { name: 'ph', label: 'pH', step: '0.01' },
            { name: 'rainfall', label: 'Rainfall (mm)', step: '0.1' }
          ].map(({ name, label, step }) => (
            <div key={name}>
              <label htmlFor={name} className="block mb-1 font-medium">{label}</label>
              <input
                id={name}
                name={name}
                type="number"
                step={step}
                value={formData[name as keyof typeof formData]}
                onChange={handleChange}
                className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
          ))}
          <div className="md:col-span-2 text-right">
            <button
              type="submit"
              className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition"
            >
              Predict Crop
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
