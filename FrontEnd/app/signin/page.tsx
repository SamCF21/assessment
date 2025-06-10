"use client";

import { useState } from "react";
import { useRouter } from 'next/navigation';

export default function SigninPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    full_name: ""
  });

  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const res = await fetch("/api/auth/signin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

    const data = await res.json();

    if (!res.ok) {
      setErrorMsg(data.error || "Signup failed");
      setSuccessMsg("");
    } else {
      setSuccessMsg("User registered successfully!");
      setErrorMsg("");
      setFormData({ username: "", email: "", password: "", full_name: "" });
      router.push("/predict");
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-emerald-300/80 via-white to-emerald-100">
      <div className="w-full max-w-md border border-emerald-300 bg-white rounded-2xl shadow-md p-8">
        <h1 className="text-2xl font-bold mb-6 text-center text-black">Crear Cuenta</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {[
            { name: "username", label: "Usuario" },
            { name: "email", label: "Correo Electrónico" },
            { name: "password", label: "Contraseña" },
            { name: "full_name", label: "Nombre Completo" }
          ].map(({ name, label }) => (
            <div key={name}>
              <label htmlFor={name} className="block text-sm font-medium text-gray-700">
                {label}
              </label>
              <input
                type={name === "password" ? "password" : "text"}
                id={name}
                name={name}
                value={formData[name as keyof typeof formData]}
                onChange={handleChange}
                className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none text-black"
                required
              />
            </div>
          ))}

          {errorMsg && <p className="text-sm text-red-500">{errorMsg}</p>}
          {successMsg && <p className="text-sm text-green-600">{successMsg}</p>}

          <button
            type="submit"
            className="w-full bg-emerald-800/80 text-white py-2 px-4 rounded-md hover:bg-emerald-800 transition"
          >
            Crear Cuenta
          </button>
        </form>
      </div>
    </div>
  );
}
