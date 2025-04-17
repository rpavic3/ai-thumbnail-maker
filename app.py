from supabase import create_client, Client
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# 🔐 Supabase credentials
SUPABASE_URL = "https://mjwjxxfnqbaroxwjewms.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# 🔐 OpenAI key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 🔐 Flux (FAL) key
FAL_API_KEY = os.environ.get("FAL_API_KEY")

app = Flask(__name__)
CORS(app)

# Flux image generation function
def generate_flux_image(prompt):
    print("🌀 Sending prompt to Flux:", prompt)
    api_url = "https://fal.run/fal-ai/flux-pro/v1.1-ultra"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "width": 1280,
        "height": 720
    }

    response = requests.post(api_url, headers=headers, json=payload)
    print("✅ Flux responded with:", response.status_code)

    if response.status_code == 200:
        image_url = response.json()["images"][0]["url"]
        print("🖼️ Image URL:", image_url)
        return image_url
    else:
        print("❌ FAL ERROR:", response.text)
        raise Exception("Flux generation failed.")

# Tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_thumbnail_image",
            "description": "Generate a YouTube thumbnail image from a visual idea",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "DALL·E-style visual description of the thumbnail"
                    }
                },
                "required": ["prompt"]
            }
        }
    }
]

@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    data = request.json
    title = data.get("title")
    niche = data.get("niche")

    print("🧠 Prompt generation requested")
    print("📌 Title:", title)
    print("📌 Niche:", niche)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a viral YouTube thumbnail designer who generates visual ideas and images."},
                {"role": "user", "content": f"""Create ONE thumbnail idea for this video.

Video Title: {title}
Channel Niche: {niche}

Return only a DALL·E-style visual prompt (don't call any tools)."""}
            ]
        )

        prompt = response.choices[0].message.content.strip()
        print("✅ Prompt generated:", prompt)

        return jsonify({"prompt": prompt})

    except Exception as e:
        print("❌ GPT prompt generation failed:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    print("📥 /generate endpoint called")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("⛔ No token provided!")
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ")[1]

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY
        }
        user_response = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)

        if user_response.status_code != 200:
            raise Exception("Invalid token")

        user_info = user_response.json()
        user_id = user_info["id"]
        print("✅ Verified user ID:", user_id)

    except Exception as e:
        print("❌ Token verification failed:", e)
        return jsonify({"error": "Invalid token"}), 401

    data = request.json
    title = data.get("title")
    niche = data.get("niche")
    image_prompt = data.get("prompt")

    print("📌 Title:", title)
    print("📌 Niche:", niche)
    print("🎨 Final Prompt to use:", image_prompt)

    try:
        image_url = generate_flux_image(image_prompt)
        print("🖼️ Image URL:", image_url)

        try:
            response = supabase.table("thumbnail").insert({
                "user_id": user_id,
                "title": title,
                "niche": niche,
                "prompt": image_prompt,
                "image_url": image_url
            }).execute()

            print("✅ Saved to Supabase")
        except Exception as e:
            print("❌ Failed to save to Supabase:", e)

        return jsonify({
            "label": "User-approved idea",
            "layout": "Prompt confirmed by user",
            "prompt": image_prompt,
            "image_url": image_url
        })

    except Exception as e:
        print("🔥 GENERATION ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history():
    print("📥 /history endpoint called")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("⛔ No token provided!")
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ")[1]

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY
        }
        user_response = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)

        if user_response.status_code != 200:
            raise Exception("Invalid token")

        user_info = user_response.json()
        user_id = user_info["id"]
        print("✅ Verified user ID:", user_id)

    except Exception as e:
        print("❌ Token verification failed:", e)
        return jsonify({"error": "Invalid token"}), 401

    try:
        print("📦 Fetching thumbnails for user...")
        response = supabase.table("thumbnail") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()

        print("✅ History fetched:", len(response.data), "items")
        return jsonify(response.data)

    except Exception as e:
        print("❌ Error fetching history:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)

