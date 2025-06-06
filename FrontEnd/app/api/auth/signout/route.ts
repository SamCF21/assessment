import { NextResponse } from "next/server";

export async function POST() {
  // Aquí podrías hacer logout, limpiar cookies, etc.
  return NextResponse.json({ message: "Signed out successfully" });
}