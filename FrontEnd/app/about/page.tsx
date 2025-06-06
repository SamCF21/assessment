import Navbar from '@/components/navBar';

export default function AboutUs() {
  return (
    <div className="min-h-screen flex">
      <Navbar />
      <main className="flex-1 p-10 bg-gray-50">
        <h1 className="text-3xl font-bold mb-6">About us</h1>
        <p className="text-gray-700">
          We are a team of developers focused on improving agricultural productivity
          through AI-based crop recommendations. Our mission is to support farmers with data-driven decisions.
        </p>
      </main>
    </div>
  );
}