'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

const Icon = (src: string, alt: string) => (
  <Image src={src} alt={alt} width={24} height={24} />
);

export default function Page() {
  const [y, setY] = useState(0);
  const [ready, setReady] = useState(false);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    setReady(true);
    const onScroll = () => setY(window.scrollY);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (video.current && ready) video.current.playbackRate = 0.5;
  }, [ready]);

  if (!ready) return null;

  const items = [
    {
      icon: Icon('/icons/brain.svg', 'Brain'),
      title: 'Asesoramiento Avanzado',
      desc: 'Nuestra IA analiza 7 factores clave: temperatura, humedad, precipitación, pH del suelo, nitrógeno, fósforo y potasio para recomendar el cultivo perfecto.',
    },
    {
      icon: Icon('/icons/leaf.svg', 'Leaf'),
      title: '22 Variedades Cubiertas',
      desc: 'Desde arroz y trigo hasta mangos y uvas, nuestro sistema cuenta con algunos de los cultivos más populares e importantes de la industria.',
    },
    {
      icon: Icon('/icons/globe.svg', 'Globe'),
      title: 'Recomendaciones Instantáneas',
      desc: 'Simplemente llena los campos y obtendrás las sugerencias de inmediato, sin esperas ni procesos complicados.',
    },
    {
      icon: Icon('/icons/target.svg', 'Target'),
      title: 'Facilidad de Uso',
      desc: 'Con una interfaz intuitiva y fácil de usar, hacemos que las decisiones inteligentes sean aún más simples.',
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-emerald-100 to-teal-100">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-md shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/home" className="text-2xl font-bold bg-gradient-to-r from-emerald-800 to-emerald-600 bg-clip-text text-transparent"> CROPWISE</Link>
            <div className="flex items-center space-x-4">
              <Link href="/login" className="px-4 py-2 text-emerald-900 hover:text-emerald-800 font-medium transition-colors border-2 border-emerald-800/30 rounded-lg">Inicia Sesión</Link>
              <Link href="/signin" className="px-4 py-2 bg-emerald-800/80 text-white rounded-lg hover:bg-emerald-800 transition-colors font-medium">Regístrate</Link>
            </div>
          </div>
        </div>
      </nav>

      <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-emerald-700/90 to-teal-700/90">
        <div className="max-w-7xl mx-auto">
          <div className="relative h-[70vh] overflow-hidden rounded-3xl shadow-[0_4px_20px_rgba(0,0,0,0.3)] border border-emerald-900/20">
            <div className="absolute inset-0">
              <video ref={video} autoPlay muted loop playsInline className="absolute inset-0 w-full h-full object-cover" style={{ transform: `translateY(${y * 0.4}px) scale(1.1)` }}>
                <source src="/crop-field.mp4" type="video/mp4" />
              </video>
              <div className="absolute inset-0 bg-gradient-to-b from-emerald-800/60 via-emerald-800/40 to-emerald-800/60" />
            </div>
            <div className="relative h-full flex items-center justify-center px-4">
              <div className="max-w-4xl text-center text-white">
                <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="text-5xl md:text-6xl font-bold mb-6">
                  Modelo de Inteligencia Artificial de Recomendación de Cultivos
                </motion.h1>
                <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2 }} className="text-xl md:text-2xl text-white mb-8">
                  Toma decisiones inteligentes sobre tus cultivos analizando 7 factores clave con nuestro modelo de machine learning
                </motion.p>
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4 }} className="flex flex-col sm:flex-row gap-4 justify-center">
                  <Link href="/signin" className="px-8 py-4 bg-emerald-800/80 text-white rounded-xl hover:bg-emerald-800 transition-colors font-semibold text-lg">Pruébalo ahora</Link>
                  <Link href="/about" className="px-8 py-4 bg-white/10 text-white border-2 border-emerald-300/30 rounded-xl hover:bg-white/20 transition-colors font-semibold text-lg">Sobre nosotras</Link>
                </motion.div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto text-center mb-16">
          <h2 className="text-3xl font-bold text-emerald-900 mb-4">¿Por qué utilizar nuestro sistema?</h2>
          <p className="text-xl text-emerald-700 max-w-2xl mx-auto">Combinamos lo último en tecnología con el campo tradicional de la agricultura para ofrecer una solución de la era digital.</p>
        </div>
        <div className="grid md:grid-cols-2 gap-8">
          {items.map((f, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: i * 0.1 }} viewport={{ once: true }} className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-2xl p-8 hover:shadow-lg transition-shadow border border-emerald-200">
                <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 p-3 bg-white rounded-xl shadow-sm text-emerald-600">{f.icon}</div>
                    <div>
                    <h3 className="text-xl font-semibold text-emerald-900 mb-3">{f.title}</h3>
                    <p className="text-emerald-700 leading-relaxed">{f.desc}</p>
                    </div>
                </div>
                </motion.div>
          ))}
        </div>
      </section>

      <section className="py-12 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-emerald-700/90 to-teal-700/90">
        <div className="max-w-4xl mx-auto text-center text-white">
          <h2 className="text-3xl font-bold mb-4">¿Quieres optimizar tus cultivos?</h2>
          <p className="text-xl mb-6">¡Anímate a probar la manera <em>inteligente</em> y descubre qué cultivos son perfectos para tus necesidades!</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/signin" className="px-8 py-4 bg-white text-emerald-900 rounded-xl hover:bg-emerald-50 transition-colors font-semibold text-lg">¡Quiero empezar!</Link>
            <Link href="/login" className="px-8 py-4 bg-white/10 text-white border-2 border-emerald-300/30 rounded-xl hover:bg-white/20 transition-colors font-semibold text-lg">Iniciar Sesión</Link>
          </div>
        </div>
      </section>
    </div>
  );
}
