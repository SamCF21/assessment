import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  console.log("Received request in /api/auth/login");
  try {
    const body = await request.json();
    const { username, password } = body as {
      username: string;
      password: string;
    };

    if (!username || !password) {
      return NextResponse.json(
        { error: "Username and password are required." },
        { status: 400 }
      );
    }

    const flaskRes = await fetch(` http://10.49.12.49:1010//login `, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const flaskData = await flaskRes.json();

    if (!flaskRes.ok) {
      return NextResponse.json(flaskData, { status: flaskRes.status });
    }

    return NextResponse.json(flaskData, { status: 200 });

  } catch (err: unknown) {
    console.error("Error in /api/auth/signin", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json(
      { error: "Internal error in signin proxy", detail: message },
      { status: 500 }
    );
  }
}
