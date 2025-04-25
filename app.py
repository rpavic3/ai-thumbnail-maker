# Keep existing imports
from supabase import create_client, Client
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai
import json
import requests
import os
from dotenv import load_dotenv
import stripe

load_dotenv()

# 🔐 Supabase credentials
SUPABASE_URL = "https://mjwjxxfnqbaroxwjewms.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# 🔐 OpenAI key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 🔐 Flux (FAL) key
FAL_API_KEY = os.environ.get("FAL_API_KEY")

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY") # <<< ADD THIS (Use your sk_live_...)
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

# --- Map Stripe Price IDs to Credit Amounts ---
# !!! IMPORTANT: Replace with YOUR actual Price IDs from Stripe Live mode !!!
PRICE_ID_TO_CREDITS = {
    "price_1RGPLZFj9LYfI1R0bRYP2IJn": 50,   # Example: Your Price ID for the 50 credit pack
    # Add entries for all the credit packs you created in Stripe
}


app = Flask(__name__)
CORS(app, origins=["https://ai-thumbnail-copilot.vercel.app"]) # Consider restricting origins in production: CORS(app, origins=["YOUR_VERCEL_DOMAIN"])

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
# In app.py

@app.route("/get_profile", methods=["GET"])
def get_profile():
    print("📥 /get_profile endpoint called")
    auth_header = request.headers.get("Authorization")
    user_id = None # Define user_id outside try block to use in logging
    try:
        user_id = get_user_id_from_token(auth_header)

        # Fetch profile data
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()

        # --- MODIFIED CHECK ---
        # First check if maybe_single() found a row (returned something other than None)
        if profile_response:
             # Now check if the response object actually contains data
             # (Should generally be true if profile_response is not None, but good practice)
             if hasattr(profile_response, 'data') and profile_response.data:
                 credits = profile_response.data.get("credits", 5) # Default to 5 if credits column is somehow null
                 print(f"✅ Fetched credits for user {user_id}: {credits}")
                 return jsonify({"credits": credits})
             else:
                 # This case means we got a response object, but it had no data. Unexpected.
                 print(f"⚠️ Profile response received, but no data found for user {user_id}. Response: {profile_response}. Returning 0 credits.")
                 return jsonify({"credits": 0}) # Or handle as an error
        else:
            # profile_response is None, meaning maybe_single() found nothing. THIS IS THE LIKELY CASE.
            print(f"⚠️ Profile not found for user {user_id} in /get_profile (maybe_single returned None). Returning 0 credits.")
            # Optional: You could create the profile here if it's missing,
            # but it might be better to handle that during signup/first action.
            # For now, just return 0.
            return jsonify({"credits": 0})

    except Exception as e:
        # Log error using the user_id if available
        user_id_str = f" for user {user_id}" if user_id else ""
        print(f"❌ Error in /get_profile{user_id_str}: {e}")

        error_message = str(e)
        status_code = 500
        # Make error message more user-friendly if it's the NoneType error
        if isinstance(e, AttributeError) and "'NoneType' object has no attribute 'data'" in error_message:
             error_message = "Failed to retrieve profile data (Profile might be missing)."
             # Keep status_code 500 as it indicates a server-side handling issue before this fix
        elif "token" in error_message.lower() or "unauthorized" in error_message.lower():
             status_code = 401
        # Add more specific error handling if needed
        return jsonify({"error": error_message}), status_code


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
        
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    print("📥 /create-checkout-session called")
    auth_header = request.headers.get("Authorization")
    data = request.json
    price_id = data.get('priceId')

    if not price_id:
        print("❌ Price ID missing in request")
        return jsonify({"error": "Price ID is required."}), 400

    # !!! IMPORTANT: Replace with YOUR Vercel frontend domain !!!
    YOUR_FRONTEND_DOMAIN = "https://https://ai-thumbnail-copilot.vercel.app" # Or your custom domain

    try:
        # 1. Verify user
        user_id = get_user_id_from_token(auth_header)
        print(f"✅ User {user_id} requesting checkout for price {price_id}")

        # 2. Define redirect URLs
        success_url = f"{YOUR_FRONTEND_DOMAIN}?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{YOUR_FRONTEND_DOMAIN}?payment=cancel"

        # 3. Create Stripe Checkout Session
        print(f"Creating Stripe session for Price ID: {price_id}")
        checkout_session = stripe.checkout.Session.create(
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='payment',
            success_url=success_url,
            cancel_url=cancel_url,
            # Pass user_id and price_id for the webhook
            metadata={
                'supabase_user_id': user_id,
                'price_id': price_id
            }
        )
        print(f"✅ Stripe session created: {checkout_session.id}")
        # Return the session URL to the frontend
        return jsonify({'url': checkout_session.url})

    except stripe.error.StripeError as e:
        print(f"❌ Stripe Error creating session: {e}")
        # Provide a user-friendly message if available
        user_message = getattr(e, 'user_message', str(e))
        return jsonify({"error": f"Stripe error: {user_message}"}), 500
    except Exception as e:
        print(f"❌ Error creating checkout session: {e}")
        if "token" in str(e).lower() or "unauthorized" in str(e).lower():
            return jsonify({"error": str(e)}), 401
        else:
            return jsonify({"error": "Failed to create checkout session."}), 500
        
@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    print("🔔 /stripe-webhook received event")
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    event = None

    if not STRIPE_WEBHOOK_SECRET:
        print("❌ Webhook Error: Signing secret not configured on server.")
        return jsonify(error="Webhook secret not configured"), 500

    try:
        # Verify the event signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
        print(f"✅ Webhook signature verified for event: {event['type']}")
    except ValueError as e:
        # Invalid payload
        print(f"❌ Invalid webhook payload: {e}")
        return jsonify(error="Invalid payload"), 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        print(f"❌ Invalid webhook signature: {e}")
        return jsonify(error="Invalid signature"), 400
    except Exception as e:
        print(f"❌ Error constructing webhook event: {e}")
        return jsonify(error="Webhook error"), 500 # Use 500 for generic server errors


    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        print(f"Processing 'checkout.session.completed' for session: {session.get('id')}")

        # Check if payment was successful
        if session.get('payment_status') == 'paid':
            metadata = session.get('metadata')
            if not metadata:
                print("❌ Metadata missing from checkout session.")
                # Acknowledge receipt but indicate issue
                return jsonify(status="Metadata missing"), 200 # Still return 200 OK to Stripe

            user_id = metadata.get('supabase_user_id')
            price_id = metadata.get('price_id')

            if not user_id or not price_id:
                print("❌ User ID or Price ID missing from metadata.")
                return jsonify(status="Missing metadata fields"), 200 # Still return 200 OK

            # Look up credits to add based on the price ID
            credits_to_add = PRICE_ID_TO_CREDITS.get(price_id)
            if credits_to_add is None:
                print(f"⚠️ Unrecognized price_id in webhook metadata: {price_id}")
                return jsonify(status="Unrecognized price ID"), 200 # Still return 200 OK

            print(f"Attempting to grant {credits_to_add} credits to user {user_id} for price {price_id}")

            # --- Securely Update Credits in Supabase ---
            try:
                # 1. Get current credits
                profile_res = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()

                current_credits = 0
                if profile_res and hasattr(profile_res, 'data') and profile_res.data:
                    current_credits = profile_res.data.get("credits", 0)
                else:
                    # Handle missing profile - maybe create one? For now, assume 0.
                    print(f"⚠️ Profile not found for user {user_id} during credit update. Starting credit count at 0.")
                    # Consider inserting profile: supabase.table("profiles").insert({"id": user_id, "credits": 0}).execute()

                # 2. Calculate new total
                new_total_credits = current_credits + credits_to_add
                print(f"   Updating credits for {user_id}: {current_credits} -> {new_total_credits}")

                # 3. Update the database
                update_res = supabase.table("profiles").update({"credits": new_total_credits}).eq("id", user_id).execute()

                # Check for update errors
                if hasattr(update_res, 'error') and update_res.error:
                     print(f"❌ Supabase DB Error updating credits for user {user_id}. Error: {update_res.error}")
                     return jsonify(error="Failed to update credits in DB"), 500 # Return 500 so Stripe retries
                # Also check if data was actually returned (indicates success)
                elif not (hasattr(update_res, 'data') and update_res.data):
                     print(f"❌ Supabase DB Warning: Update command for user {user_id} might not have updated rows (returned no data).")
                     # Depending on library version, this might indicate the user_id didn't match.
                     # Treat as error for retry.
                     return jsonify(error="Failed to update credits (user not found?)"), 500
                else:
                     print(f"✅ Credits successfully updated for user {user_id}.")

            except Exception as db_error:
                print(f"❌ Database exception during credit update for user {user_id}: {db_error}")
                return jsonify(error="Database update failed"), 500 # Return 500 so Stripe retries

        else:
             print(f"Payment status was not 'paid': {session.get('payment_status')}")
             # No credits granted for unpaid sessions

    else:
        print(f"🤷‍♀️ Unhandled event type received: {event['type']}")

    # Acknowledge the webhook event was received successfully
    return jsonify(success=True), 200


if __name__ == "__main__":
    import os
    # Use Gunicorn or Waitress in production instead of Flask development server
    port = int(os.environ.get("PORT", 5001)) # Render typically sets PORT env var
    # For Render deployment, host should be '0.0.0.0'
    app.run(host="0.0.0.0", port=port) # Removed debug=True for production/Render