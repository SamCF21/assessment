import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
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

    const flaskRes = await fetch("http://localhost:5001/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const flaskData = await flaskRes.json();

    if (!flaskRes.ok) {
      return NextResponse.json(flaskData, { status: flaskRes.status });
    }

    return NextResponse.json(flaskData, { status: 200 });

  } catch (err: any) {
    console.error("Login error:", err);
    return NextResponse.json(
      { error: "Internal error in login proxy", detail: err.message },
      { status: 500 }
    );
  }
}
