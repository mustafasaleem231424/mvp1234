import Link from 'next/link';
import { Leaf } from 'lucide-react';

export default function NotFound() {
  return (
    <main className="min-h-screen flex items-center justify-center px-6">
      <div className="text-center max-w-md">
        <div className="w-20 h-20 rounded-2xl bg-[#F2F7F0] border border-[rgba(33,160,73,0.2)] flex items-center justify-center mx-auto mb-8">
          <Leaf className="w-10 h-10 text-[#21A049]" />
        </div>
        <h1 className="text-6xl font-extrabold text-[var(--green-dark)] mb-4">404</h1>
        <h2 className="text-2xl font-bold mb-4 text-[var(--text)]">Page Not Found</h2>
        <p className="text-[var(--text-secondary)] mb-8">
          Looks like this crop field doesn&apos;t exist. Let&apos;s get you back to scanning.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link href="/" className="btn btn-secondary">Go Home</Link>
          <Link href="/dashboard" className="btn btn-primary">Start Scanning</Link>
        </div>
      </div>
    </main>
  );
}
