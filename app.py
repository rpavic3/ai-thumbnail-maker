# Keep existing imports
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
CORS(app) # Consider restricting origins in production: CORS(app, origins=["YOUR_VERCEL_DOMAIN"])

# Flux image generation function (Keep as is)
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
        # Try to parse FAL error message if available
        error_message = "Flux generation failed."
        try:
            error_detail = response.json().get('detail', response.text)
            error_message = f"Flux generation failed: {error_detail}"
        except json.JSONDecodeError:
            error_message = f"Flux generation failed: {response.status_code} - {response.text}"
        raise Exception(error_message)


# Tool schema (Keep as is)
tools = [
    # ... (rest of the tools schema)
]

@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    # --- Authentication Check (Optional but Recommended) ---
    # Although the generation itself costs credits, maybe you want only logged-in users
    # to even generate prompts. If so, add token verification here like in /generate.
    # If not, keep it as is.
    # Example:
    # auth_header = request.headers.get("Authorization")
    # try:
    #     user_id = verify_user_token(auth_header) # You'd need to extract token verification logic
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 401
    # print(f"✅ Verified user {user_id} for prompt generation")
    # --- End Optional Auth Check ---

    data = request.json
    title = data.get("title")
    niche = data.get("niche")

    # Keep the rest of the function as is
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

# --- Helper function to verify token and get user ID ---
def get_user_id_from_token(auth_header):
    if not auth_header or not auth_header.startswith("Bearer "):
        print("⛔ No token provided!")
        raise Exception("Unauthorized: No token provided")

    token = auth_header.split(" ")[1]

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY # Use Service Role Key for backend verification
        }
        # Verify the token using Supabase GoTrue endpoint
        user_response = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)

        if user_response.status_code != 200:
            print(f"❌ Supabase token verification failed: {user_response.status_code} - {user_response.text}")
            raise Exception("Invalid token")

        user_info = user_response.json()
        user_id = user_info["id"]
        print(f"✅ Verified user ID via token: {user_id}")
        return user_id

    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        raise Exception(f"Invalid token: {e}")


# --- NEW Endpoint to get user profile/credits ---
@app.route("/get_profile", methods=["GET"])
def get_profile():
    print("📥 /get_profile endpoint called")
    auth_header = request.headers.get("Authorization")
    try:
        user_id = get_user_id_from_token(auth_header)

        # Fetch profile data (specifically credits)
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()

        if profile_response.data:
            credits = profile_response.data.get("credits", 0) # Default to 0 if somehow null
            print(f"✅ Fetched credits for user {user_id}: {credits}")
            return jsonify({"credits": credits})
        else:
            # This case should ideally not happen if the trigger works,
            # but handle it defensively. Maybe create a profile here?
            # For now, return 0 credits or an error.
            print(f"⚠️ Profile not found for user {user_id}. Returning 0 credits.")
            # Optionally create profile:
            # supabase.table("profiles").insert({"id": user_id, "credits": 5}).execute()
            # return jsonify({"credits": 5})
            return jsonify({"credits": 0}) # Or return jsonify({"error": "Profile not found"}), 404

    except Exception as e:
        print(f"❌ Error in /get_profile: {e}")
        # Distinguish between auth errors and other errors
        if "token" in str(e).lower() or "unauthorized" in str(e).lower():
             return jsonify({"error": str(e)}), 401
        else:
             return jsonify({"error": "Failed to fetch profile data."}), 500


@app.route("/generate", methods=["POST"])
def generate():
    print("📥 /generate endpoint called")
    auth_header = request.headers.get("Authorization")

    try:
        user_id = get_user_id_from_token(auth_header)

        # --- Fetch User Credits ---
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()

        if not profile_response.data:
             print(f"❌ Profile not found for user {user_id} during generation.")
             # Maybe create it here as a fallback if the trigger failed?
             # supabase.table("profiles").insert({"id": user_id, "credits": 5}).execute()
             # current_credits = 5
             # Or just deny generation:
             return jsonify({"error": "User profile not found. Cannot generate."}), 404 # Or 500

        current_credits = profile_response.data.get("credits", 0)
        print(f"💰 User {user_id} has {current_credits} credits.")

        # --- Check Credits ---
        if current_credits <= 0:
            print(f"🚫 User {user_id} has insufficient credits (0).")
            return jsonify({"error": "Insufficient credits to generate thumbnail."}), 403 # 403 Forbidden is appropriate

    except Exception as e:
        # Catch auth errors from get_user_id_from_token
        print(f"❌ Authentication or profile fetch error in /generate: {e}")
        return jsonify({"error": str(e)}), 401 # Return 401 for auth issues

    # --- Proceed with Generation if Credits OK ---
    data = request.json
    title = data.get("title")
    niche = data.get("niche")
    image_prompt = data.get("prompt")

    if not image_prompt:
        return jsonify({"error": "Prompt is required."}), 400

    print("📌 Title:", title)
    print("📌 Niche:", niche)
    print("🎨 Final Prompt to use:", image_prompt)

    try:
        # Generate the image
        image_url = generate_flux_image(image_prompt)
        print("🖼️ Image generated successfully:", image_url)

        # --- Deduct Credit ---
        try:
            new_credits = current_credits - 1
            update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
            # Small check for errors during update, though execute() might raise exceptions too
            if not update_response.data and update_response.error:
                 print(f"⚠️ WARNING: Failed to deduct credit for user {user_id} after generation. Error: {update_response.error}")
                 # Decide how critical this is. Maybe still return success but log?
                 # For now, we'll proceed but log the warning.
            else:
                 print(f"✅ Deducted 1 credit from user {user_id}. New balance: {new_credits}")

        except Exception as credit_error:
             print(f"❌ CRITICAL: Failed to deduct credit for user {user_id} after successful generation: {credit_error}")
             # Consider if you should still save to history or return an error
             # For simplicity, we'll log and continue for now

        # --- Save to History ---
        try:
            insert_response = supabase.table("thumbnail").insert({
                "user_id": user_id,
                "title": title,
                "niche": niche,
                "prompt": image_prompt,
                "image_url": image_url
            }).execute()
            if not insert_response.data and insert_response.error:
                 print(f"⚠️ WARNING: Failed to save thumbnail to history for user {user_id}. Error: {insert_response.error}")
            else:
                 print("✅ Saved to Supabase history")
        except Exception as history_error:
             print(f"❌ CRITICAL: Failed to save thumbnail to history for user {user_id}: {history_error}")
             # Log and continue

        # --- Return Success Response with New Credit Count ---
        return jsonify({
            # Keep old fields if needed by frontend logic, or remove
            "label": "User-approved idea",
            "layout": "Prompt confirmed by user",
            "prompt": image_prompt,
            "image_url": image_url,
            "new_credits": new_credits # Send updated credits back
        })

    except Exception as e:
        print(f"🔥 GENERATION or PROCESSING ERROR for user {user_id}: {e}")
        # Check if it was a Flux error specifically
        if "Flux generation failed" in str(e):
             return jsonify({"error": f"Image generation failed: {e}"}), 502 # Bad Gateway or similar if external service fails
        # Check if it was the insufficient credits error we added
        elif "Insufficient credits" in str(e):
             return jsonify({"error": str(e)}), 403
        else:
             return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    print("📥 /history endpoint called")
    auth_header = request.headers.get("Authorization")

    try:
        user_id = get_user_id_from_token(auth_header)

        # Fetch history
        print(f"📦 Fetching thumbnails for user {user_id}...")
        response = supabase.table("thumbnail") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()

        # Handle potential errors during fetch
        if not response.data and response.error:
             print(f"❌ Error fetching history from Supabase: {response.error}")
             raise Exception(f"Database error fetching history: {response.error.message}")

        print("✅ History fetched:", len(response.data), "items")
        return jsonify(response.data)

    except Exception as e:
        print(f"❌ Error in /history: {e}")
        if "token" in str(e).lower() or "unauthorized" in str(e).lower():
             return jsonify({"error": str(e)}), 401
        else:
             return jsonify({"error": f"Failed to fetch history: {e}"}), 500


if __name__ == "__main__":
    import os
    # Use Gunicorn or Waitress in production instead of Flask development server
    port = int(os.environ.get("PORT", 5001)) # Render typically sets PORT env var
    # For Render deployment, host should be '0.0.0.0'
    app.run(host="0.0.0.0", port=port) # Removed debug=True for production/Render