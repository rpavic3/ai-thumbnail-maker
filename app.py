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
SUPABASE_URL = "https://mjwjxxfnqbaroxwjewms.supabase.co" # Your Supabase URL
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Ensure the service role key is loaded
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable not set.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# 🔐 OpenAI key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 🔐 Flux (FAL) key
FAL_API_KEY = os.environ.get("FAL_API_KEY")

app = Flask(__name__)
CORS(app) # Allow requests from your frontend domain

# --- NEW: Helper function for user credits ---
# --- REPLACE your existing function with this one ---
# --- REPLACE your existing function AGAIN with this SAFER version ---
# --- TEMPORARY TEST VERSION ---
def ensure_user_has_credit_row(user_id):
    """
    TEMPORARY TEST FUNCTION:
    Checks if the Supabase connection works by querying the 'thumbnail' table
    instead of the 'users' table. Returns a dummy credit value if successful.
    """
    try:
        print("🧪 TEST: Attempting to query 'thumbnail' table instead of 'users'...")
        # Query the thumbnail table - assumes this table exists and has an 'id' column
        test_thumb_response = supabase.table("thumbnail").select("id").limit(1).execute()

        # Check if the response object itself is None (the core issue we were seeing)
        if test_thumb_response is None:
             print("❌ TEST FAILED: Querying 'thumbnail' table returned None object!")
             # Raise the specific error we were seeing before if it happens here too
             raise ConnectionError("TEST FAILED: thumbnail query returned None")
        else:
             # If we get a valid response object (even with empty data), the connection worked
             print(f"✅ TEST SUCCEEDED: Queried 'thumbnail' table. Response data: {getattr(test_thumb_response, 'data', 'N/A')}")
             # Since this is just a test, we can't return actual credits.
             # Return a default value (like 5) to allow the flow to continue.
             print("🧪 TEST: Returning dummy credits (5) to continue flow.")
             return 5 # Return dummy value

    except Exception as e:
        # Log any error during the test query
        print(f"❌ TEST ERROR during thumbnail query: {type(e).__name__} - {e}")
        raise e # Re-raise the exception so the main endpoint knows something failed
# --- END OF TEMPORARY TEST VERSION ---


# Flux image generation function (Unchanged)
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
        raise Exception(f"Flux generation failed: {response.status_code} {response.text}")


@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    # This endpoint remains unchanged as it doesn't consume credits
    data = request.json
    title = data.get("title")
    niche = data.get("niche")

    print("🧠 Prompt generation requested")
    print("📌 Title:", title)
    print("📌 Niche:", niche)

    if not title or not niche:
         return jsonify({"error": "Missing title or niche"}), 400

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

    # --- 1. Authentication & Get User ID ---
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("⛔ No token provided!")
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ")[1]
    user_id = None # Initialize user_id

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY # Use service key to validate token against auth schema
        }
        # This request validates the token and gets user info
        user_response = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)

        if user_response.status_code != 200:
             print(f"❌ Invalid token response: {user_response.status_code} - {user_response.text}")
             raise Exception("Invalid token")

        user_info = user_response.json()
        user_id = user_info.get("id")
        if not user_id:
             print("❌ User ID not found in token response:", user_info)
             raise Exception("User ID not found in token")

        print(f"✅ Verified user ID: {user_id}")

        # --- 2. Ensure Credits Row & Check Credits ---
        print("💡 Checking user credits...")
        current_credits = ensure_user_has_credit_row(user_id) # Ensure row exists, get current credits

        print(f"🪙 Current credits for user {user_id}: {current_credits}")

        if current_credits <= 0:
            print(f"🚫 User {user_id} is out of credits.")
            return jsonify({"error": "You’ve used all your free credits! Stay tuned — paid plans coming soon 😊"}), 403 # Use 403 Forbidden

        # --- If credits OK, proceed ---
        data = request.json
        title = data.get("title")
        niche = data.get("niche")
        image_prompt = data.get("prompt")

        if not image_prompt:
             return jsonify({"error": "Missing image prompt"}), 400
        # Optionally add checks for title/niche if needed for history

        print("📌 Title:", title)
        print("📌 Niche:", niche)
        print("🎨 Final Prompt to use:", image_prompt)

        # --- 3. Generate Image ---
        image_url = generate_flux_image(image_prompt) # This will raise Exception on failure
        print("🖼️ Image URL generated:", image_url)

        # --- 4. Save to History (Optional but recommended before deducting credits) ---
        try:
            # Use a dictionary for insert data
            insert_data = {
                "user_id": user_id,
                "prompt": image_prompt,
                "image_url": image_url
            }
            # Only include title and niche if they are provided
            if title:
                insert_data["title"] = title
            if niche:
                insert_data["niche"] = niche

            response = supabase.table("thumbnail").insert(insert_data).execute()

            if response.data:
                 print("✅ Saved generation history to Supabase")
            else:
                 print("⚠️ Failed to save generation history to Supabase, but proceeding. Response:", response)
                 # Decide if this failure should prevent credit deduction
                 # For now, we'll still deduct credits

        except Exception as e:
            print(f"❌ Failed to save history to Supabase for user {user_id}: {e}")
            # Decide if this failure should prevent credit deduction
            # For now, we'll still deduct credits

        # --- 5. Deduct Credit ---
        try:
             update_response = supabase.table("users").update({"credits": current_credits - 1}).eq("id", user_id).execute()
             if update_response.data:
                 print(f"💸 Deducted 1 credit for user {user_id}. Remaining: {current_credits - 1}")
             else:
                 # This is more serious - log it prominently
                 print(f"🚨 CRITICAL: Failed to deduct credit for user {user_id} after generation! Response: {update_response}")
        except Exception as e:
             print(f"🚨 CRITICAL: Exception during credit deduction for user {user_id}: {e}")


        # --- 6. Return Success Response ---
        return jsonify({
            "label": "User-approved idea", # Consider removing if not used
            "layout": "Prompt confirmed by user", # Consider removing if not used
            "prompt": image_prompt,
            "image_url": image_url,
            "credits_remaining": current_credits - 1 # Optionally return new count
        })

    except Exception as e:
        # Catch errors from token check, credit check, image gen, or deduction attempts
        print(f"🔥 Overall /generate error for user {user_id or 'UNKNOWN'}: {e}")
        # Avoid exposing detailed internal errors to the client
        # Check if it's the specific out-of-credits message we set
        if "You’ve used all your free credits" in str(e):
             return jsonify({"error": "You’ve used all your free credits! Stay tuned — paid plans coming soon 😊"}), 403
        elif "Invalid token" in str(e):
             return jsonify({"error": "Invalid token"}), 401
        elif "Flux generation failed" in str(e) or "Fal generation failed" in str(e):
             return jsonify({"error": "Image generation failed, please try again."}), 500
        else:
             # Generic error for other unexpected issues
             return jsonify({"error": "An unexpected error occurred during generation."}), 500


@app.route("/history", methods=["GET"])
def get_history():
    # This endpoint remains largely unchanged
    print("📥 /history endpoint called")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("⛔ No token provided!")
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ")[1]
    user_id = None

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_SERVICE_ROLE_KEY
        }
        user_response = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)

        if user_response.status_code != 200:
            raise Exception("Invalid token")

        user_info = user_response.json()
        user_id = user_info.get("id")
        if not user_id:
             raise Exception("User ID not found")

        print(f"✅ Verified user ID for history: {user_id}")

        print("📦 Fetching thumbnails for user...")
        response = supabase.table("thumbnail") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()

        print("✅ History fetched:", len(response.data), "items")
        return jsonify(response.data)

    except Exception as e:
        print(f"❌ Error fetching history for user {user_id or 'UNKNOWN'}: {e}")
        if "Invalid token" in str(e):
            return jsonify({"error": "Invalid token"}), 401
        else:
            return jsonify({"error": "Failed to fetch history"}), 500

# --- NEW: Endpoint to get current credits ---
@app.route("/get_credits", methods=["GET"])
def get_credits():
    print("💰 /get_credits endpoint called")
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401

    token = auth_header.split(" ")[1]
    user_id = None

    try:
        # Verify token and get user_id
        headers = { "Authorization": f"Bearer {token}", "apikey": SUPABASE_SERVICE_ROLE_KEY }
        user_response = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)
        if user_response.status_code != 200: raise Exception("Invalid token")
        user_info = user_response.json()
        user_id = user_info.get("id")
        if not user_id: raise Exception("User ID not found")
        print(f"✅ Verified user ID for credits: {user_id}")

        # Ensure user row exists and get credits
        current_credits = ensure_user_has_credit_row(user_id)

        return jsonify({"credits": current_credits})

    except Exception as e:
        print(f"❌ Error in /get_credits for user {user_id or 'UNKNOWN'}: {e}")
        if "Invalid token" in str(e):
            return jsonify({"error": "Invalid token"}), 401
        else:
            # Don't expose potentially sensitive error details
            return jsonify({"error": "Could not fetch credits"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) # Changed default port slightly just in case
    print(f"🚀 Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)