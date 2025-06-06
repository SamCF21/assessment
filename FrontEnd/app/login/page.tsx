"use client";


import { useState } from "react";
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });

    const data = await res.json();

    if (!res.ok) {
      setErrorMsg(data.error || 'Login failed');
      setSuccessMsg('');
    } else {
      setSuccessMsg(`Welcome ${data.user.username}`);
      setErrorMsg('');
      localStorage.setItem('token', data.token);
      router.push('/');
    }
  }; 

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-emerald-300/80 via-white to-emerald-100">
      <div className="w-full max-w-md border border-emerald-300 bg-white rounded-2xl shadow-md p-8">
        <h1 className="text-2xl font-bold mb-6 text-center text-black">Iniciar Sesión</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-700">
              Usuario
            </label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none text-black"
              required
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-black">
              Contraseña
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none text-black"
              required
            />
          </div>
          {errorMsg && <p className="text-red-500 text-sm">{errorMsg}</p>}
          {successMsg && <p className="text-green-600 text-sm">{successMsg}</p>}
          <button
            type="submit"
            className="w-full bg-emerald-800/80 text-white py-2 px-4 rounded-md hover:bg-emerald-800 transition"
          >
            Iniciar Sesión
          </button>
        </form>
      </div>
    </div>
  );
}
