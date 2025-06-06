import React from 'react';

interface TeamMember {
  name: string;
  role: string;
}

interface Stat {
  number: string;
  label: string;
}

export default function AboutUsPage() {
  const teamMembers: TeamMember[] = [
    {
      name: "Cristina González",
      role: "Developer"
    },
    {
      name: "Andrea San Martín",
      role: "Developer"
    },
    {
      name: "Samantha Covarrubias",
      role: "Developer"
    },
    {
      name: "Valeria Martínez",
      role: "Developer"
    }
  ];

  const stats: Stat[] = [
    { number: "22", label: "Tipos de Cultivos" },
    { number: "24/7", label: "Disponibilidad" },
    { number: "10,000+", label: "Datos de Entrenamiento" }
  ];

  const technologies: string[] = [
    'Next.js', 'TypeScript', 'Tailwind', 'Docker', 'Node.js', 'MySQL', 'Python'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-emerald-100 to-teal-100">
      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-emerald-900 mb-6">
            Acerca de{' '}
            <span className="bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
              Nosotras
            </span>
          </h1>
          <p className="text-xl text-emerald-700 max-w-3xl mx-auto leading-relaxed">
            Somos un grupo de estudiantes de la carrera de Ingeniería en Tecnologías Computacionales de 8vo Semestre, este es nuestro proyecto de fin de carrera para el módulo de Inteligencia Artificial en la clase de Desarrollo de Aplicaiones Avanzadas de Ciencias Computacionales.
          </p>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-emerald-800/80">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center bg-white/20 backdrop-blur-sm rounded-2xl p-8 border border-emerald-300/30">
                <div className="text-4xl md:text-5xl font-bold text-white mb-3">
                  {stat.number}
                </div>
                <div className="text-emerald-100 font-medium text-lg">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-emerald-900 mb-4">
              Nuestro Equipo
            </h2>
            <p className="text-xl text-emerald-700 max-w-2xl mx-auto">
              Desarrolladoras
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers.map((member, index) => (
              <div key={index} className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-2xl p-8 text-center hover:transform hover:scale-105 transition-all duration-300 border border-emerald-200 shadow-lg hover:shadow-xl">
                <div className="w-20 h-20 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full mx-auto mb-6 flex items-center justify-center">
                  <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
                    <div className="w-8 h-8 bg-emerald-500 rounded-full"></div>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-emerald-900 mb-2">
                  {member.name}
                </h3>
                <p className="text-emerald-600 font-semibold">
                  {member.role}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-emerald-800 to-teal-800">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Tecnologías Utilizadas
            </h2>
            <p className="text-emerald-100 text-lg">
              Stack tecnológico
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-6">
            {technologies.map((tech, index) => (
              <div key={index} className="bg-white/10 backdrop-blur-sm rounded-xl p-4 text-center hover:bg-white/20 transition-all duration-300 border border-emerald-300/30">
                <span className="text-emerald-100 font-semibold text-sm md:text-base">{tech}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}