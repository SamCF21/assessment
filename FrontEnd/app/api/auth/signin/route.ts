import { NextRequest, NextResponse} from "next/server";

export async function POST(request: NextRequest) {
    try{
        const body = await request.json();
        const {
            username,
            email,
            password,
            full_name
        } = body as {
            username: string;
            email: string;
            password: string;
            full_name: string;
        };

        if (
            !username ||
            !email ||
            !password ||
            !full_name 
        ) {
            return NextResponse.json(
                {error: "Missing obligatory fields"},
                {status: 400}
            );
        }

        const flaskRes = await fetch("http://localhost:5001/api/auth/signin", {

            method: 'POST',
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                username,
                email,
                password,
                full_name
            }),
        });

        const flaskData = await flaskRes.json();

        if (!flaskRes.ok) {
            return NextResponse.json(flaskData, {status: flaskRes.status});
        }

        return NextResponse.json(flaskData, {status: 200});

        } catch(err) {
            console.error("Error in /api/auth/signin", err);
            return NextResponse.json(
                {error: "Internal error in signin proxy"},
                {status: 500}
            );
        }
}
