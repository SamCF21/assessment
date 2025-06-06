import Navbar from '@/components/navBar';

export default function Home() {
  return (
    <div className="min-h-screen flex">
      <Navbar />
      <main className="flex-1 p-10 bg-gray-50">
        <h1 className="text-3xl font-bold mb-6">Home Page</h1>
        <p className="text-gray-700">
          Placeholder
        </p>
      </main>
    </div>
  );
}