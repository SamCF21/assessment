'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';


export default function Navbar() {
    const router = useRouter();

    const handleSignOut = () => {
        localStorage.removeItem("token");
        router.push('/login');
    };

    const navItems = [
    { href: '/', label: 'Inicio', icon: '/icons/house.svg' },
    { href: '/predict', label: 'Predicción de Cultivo', icon: '/icons/sprout.svg' },
    { href: '/about', label: 'Acerca de Nosotras', icon: '/icons/users.svg' },
    ];

    return (
    <nav className="group w-20 hover:w-64 bg-emerald-800/80 text-white min-h-screen p-4 hover:p-6 rounded-r-2xl transition-all duration-300 shadow-lg backdrop-blur-md border-r border-white/10">
      <h2 className="text-xl font-bold mb-6 text-center group-hover:text-left transition-all">Crop Wise</h2>
      <ul className="space-y-3">
        {navItems.map(({ href, label, icon }) => (
          <li key={label}>
            <Link
              href={href}
              className="flex items-center gap-3 px-2 py-2 rounded-md hover:bg-white/10 hover:ring-1 hover:ring-white/30 transition"
            >
              <img src={icon} alt={label} className="w-5 h-5 invert" />
              <span className="whitespace-nowrap opacity-0 group-hover:opacity-100 group-hover:visible invisible transition duration-300">
                {label}
              </span>
            </Link>
          </li>
        ))}
        <li>
          <button
            onClick={handleSignOut}
            className="flex items-center gap-3 w-full text-left px-2 py-2 rounded-md hover:bg-white/10 hover:ring-1 hover:ring-white/30 transition"
          >
            <img src="/icons/log-out.svg" alt="Sign out" className="w-5 h-5 invert" />
            <span className="whitespace-nowrap opacity-0 group-hover:opacity-100 group-hover:visible invisible transition duration-300">
              Cerrar Sesión
            </span>
          </button>
        </li>
      </ul>
    </nav>
  );
}