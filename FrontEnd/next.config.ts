import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  
  // Configurar proxy para desarrollo local
  async rewrites() {
    // Solo en desarrollo local
    if (process.env.NODE_ENV === 'development') {
      return [
        {
          source: '/api/auth/:path*',
          destination: 'http://localhost:5001/api/auth/:path*'
        },
        {
          source: '/api/:path*', 
          destination: 'http://localhost:5001/:path*'
        }
      ]
    }
    // En producción, NGINX maneja el proxy
    return []
  }
};

export default nextConfig;