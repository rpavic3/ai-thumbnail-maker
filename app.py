# Required Imports
import os
import json
import base64 # Needed for decoding/encoding images
import io     # Needed for handling image bytes
import logging
import requests # Needed for Flux, Stability AI, and fetching image URLs
import stripe
from datetime import datetime  # Add this import for datetime functions
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI, APIError # v1.x+ library
from PIL import Image, UnidentifiedImageError # Pillow for image processing
from supabase import create_client, Client
# Optional, but good practice if dealing with filenames later:
# from werkzeug.utils import secure_filename

# --- Load Environment Variables FIRST ---
load_dotenv() # Loads variables from .env file into environment

# --- Setup Logging ---
# Will be reconfigured based on FLASK_DEBUG in __main__
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Supabase Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL: logging.error("❌ Missing Supabase URL environment variable."); raise ValueError("Supabase URL missing.")
if not SUPABASE_SERVICE_ROLE_KEY: logging.error("❌ Missing Supabase Service Role Key environment variable."); raise ValueError("Supabase Key missing.")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    logging.info("✅ Supabase client initialized.")
except Exception as e:
    logging.error(f"❌ Failed Supabase init: {e}", exc_info=True); raise

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: logging.warning("⚠️ OpenAI API Key environment variable missing or empty.")
try:
    # Ensure key is provided even if empty, OpenAI constructor requires it
    client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else "dummy_key_if_missing")
    if OPENAI_API_KEY:
      logging.info("✅ OpenAI client initialized (v1.x+).")
    else:
      logging.warning("⚠️ OpenAI client initialized with dummy key due to missing env var.")
except APIError as e:
    logging.error(f"❌ OpenAI API Error client init: {e}"); raise ValueError("OpenAI client failed init.") from e
except Exception as e:
    logging.error(f"❌ Failed OpenAI client init: {e}"); raise ValueError("OpenAI client failed init.") from e

# --- Flux (FAL) Configuration ---
FAL_API_KEY = os.getenv("FAL_API_KEY")
if not FAL_API_KEY: logging.warning("⚠️ FAL_API_KEY environment variable missing or empty.")

# --- Stripe Configuration ---
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
if not stripe.api_key: logging.warning("⚠️ Stripe Secret Key environment variable missing.")
if not STRIPE_WEBHOOK_SECRET: logging.warning("⚠️ Stripe Webhook Secret environment variable missing.")

# --- Stripe Price ID Mapping ---
PRICE_ID_TO_CREDITS = {}
price_id_env_var_name = "STRIPE_PRICE_ID_50_CREDITS"
credits_env_var_name = "CREDITS_FOR_50_PACK"
default_credits = "50"
price_id_pack1 = os.environ.get(price_id_env_var_name)
credits_pack1_str = os.environ.get(credits_env_var_name, default_credits)
if price_id_pack1:
    try:
        credits_pack1 = int(credits_pack1_str)
        PRICE_ID_TO_CREDITS[price_id_pack1] = credits_pack1
        logging.info(f"✅ Loaded Stripe Price ID: {price_id_pack1} -> {credits_pack1} credits")
    except ValueError:
        logging.error(f"❌ Invalid credit amount '{credits_pack1_str}' for price ID {price_id_pack1}.")
else:
     logging.warning(f"⚠️ {price_id_env_var_name} environment variable not set.")
if not PRICE_ID_TO_CREDITS: logging.warning("⚠️ PRICE_ID_TO_CREDITS mapping is empty.")

# --- Credit Costs ---
CREDIT_COST_PER_THUMBNAIL = 1 # Cost for thumbnail generation (Flux)
CREDIT_COST_PER_ASSET_EXTRACTION = 3 # Cost for asset extraction (OpenAI)
CREDIT_COST_PER_STYLE_ANALYSIS = 5 # Cost for style profile analysis (ChatGPT)

# --- Flask App Initialization ---
app = Flask(__name__)
allowed_origins_str = os.environ.get("FRONTEND_DOMAIN", "http://127.0.0.1:5500")
# Allow flexibility for local dev ports if needed, split by comma
cors_origins = [origin.strip() for origin in allowed_origins_str.split(',')]
CORS(app, origins=cors_origins, supports_credentials=True)
logging.info(f"✅ CORS configured for origins: {cors_origins}")

# --- Constants ---
PREVIEW_WIDTH = 320 # Target width for previews in pixels

# --- Helper Functions ---

def generate_flux_image(prompt: str) -> str:
    """Calls FAL/Flux-Pro and returns the public image URL"""
    if not FAL_API_KEY:
        logging.error("❌ Cannot generate Flux image: FAL_API_KEY is not set.")
        raise ValueError("Flux API Key is missing.")

    logging.info(f"🌀 Sending prompt to Flux: '{prompt}'")
    api_url = "https://fal.run/fal-ai/flux-pro/v1.1-ultra"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt, "num_inference_steps": 30, "guidance_scale": 7,
        "width": 1280, "height": 720
    }
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        response_data = r.json()
        if not response_data or "images" not in response_data or not response_data["images"] or "url" not in response_data["images"][0]:
             logging.error(f"❌ Flux returned unexpected response structure: {response_data}")
             raise ValueError("Flux response missing expected image URL.")
        image_url = response_data["images"][0]["url"]
        logging.info(f"✅ Flux responded successfully. Image URL: {image_url}")
        return image_url
    except requests.exceptions.RequestException as req_err:
         logging.error(f"❌ Network error calling Flux API: {req_err}", exc_info=True)
         raise ConnectionError(f"Network error connecting to Flux: {req_err}") from req_err
    except Exception as e:
         logging.error(f"❌ Failed Flux API call: {e}", exc_info=True)
         error_detail = getattr(e, 'response', None)
         error_text = getattr(error_detail, 'text', str(e))
         raise Exception(f"Flux generation failed: {error_text}") from e

def generate_preview_data_uri(full_data_uri: str, target_width: int = PREVIEW_WIDTH, output_format='JPEG', quality=60) -> str | None:
    """
    Decodes a base64 Data URI, resizes the image, and re-encodes it.
    Can output JPEG (default) or PNG.
    """
    logging.info(f"🖼️ Generating preview (target width: {target_width}px, format: {output_format})...")
    try:
        if not full_data_uri or not full_data_uri.startswith("data:image/") or ";base64," not in full_data_uri:
            logging.warning(f"⚠️ Invalid Data URI for preview generation.")
            return None

        header, encoded_data = full_data_uri.split(',', 1)
        image_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(image_data))

        save_params = {}
        if output_format.upper() == 'PNG':
             if image.mode != 'RGBA':
                 # Ensure alpha channel exists before converting/saving as PNG
                 if 'A' not in image.getbands(): image.putalpha(255) # Add opaque alpha if missing
                 image = image.convert("RGBA")
                 logging.info("Converted image to RGBA for PNG preview.")
             mime_type = 'png'
        elif output_format.upper() == 'JPEG':
              if image.mode == 'RGBA':
                  # Create white background and paste RGBA image onto it
                  bg = Image.new("RGB", image.size, (255, 255, 255))
                  try: bg.paste(image, mask=image.split()[3]) # Use alpha channel as mask
                  except IndexError: bg.paste(image) # Fallback if no alpha
                  image = bg
                  logging.info("Converted RGBA image to RGB for JPEG preview.")
              elif image.mode != 'RGB':
                  # Convert other modes (like P, L) to RGB
                  image = image.convert("RGB")
                  logging.info("Converted image to RGB for JPEG preview.")
              mime_type = 'jpeg'
              save_params['quality'] = quality
        else:
            logging.error(f"Unsupported preview format: {output_format}")
            return None

        # Ensure valid dimensions before proceeding
        width, height = image.size
        if width <= 0 or height <= 0: logging.warning(f"⚠️ Invalid dimensions ({width}x{height})."); return None

        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)

        # Ensure valid target dimensions
        if target_width <= 0 or target_height <= 0: logging.warning(f"⚠️ Invalid target dimensions ({target_width}x{target_height})."); return None


        resample_filter = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.LANCZOS # Pillow 9+ vs older
        preview_image = image.resize((target_width, target_height), resample_filter)
        logging.info(f"✅ Resized preview to {target_width}x{target_height}.")
        buffer = io.BytesIO()
        preview_image.save(buffer, format=output_format.upper(), **save_params)
        buffer.seek(0)
        preview_encoded_data = base64.b64encode(buffer.read()).decode('utf-8')
        preview_data_uri = f"data:image/{mime_type};base64,{preview_encoded_data}"
        logging.info(f"✅ Preview image generated as {output_format.upper()} Data URI.")
        return preview_data_uri
    except base64.binascii.Error as b64_err:
        logging.error(f"❌ Base64 decoding error during preview generation: {b64_err}", exc_info=True)
        return None
    except UnidentifiedImageError:
        logging.error("❌ Pillow UnidentifiedImageError: Could not open image data for preview.", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"❌ Unexpected error generating preview image: {e}", exc_info=True)
        return None

def get_user_id_from_token(auth_header):
    """Validates Supabase JWT and returns user ID or raises Exception."""
    if not auth_header or not auth_header.startswith("Bearer "):
        logging.warning("⛔ Auth header missing or invalid format.")
        raise Exception("Unauthorized: Missing or invalid token.")

    token = auth_header.split(" ")[1]
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user or not user.id:
             logging.warning(f"⛔ Token validation failed or user ID missing. Status: {getattr(user_response, 'status_code', 'N/A')}")
             raise Exception("Invalid token or user not found.")
        user_id = str(user.id)
        logging.info(f"✅ Token validated successfully for user ID: {user_id}")
        return user_id
    except Exception as e:
        logging.error(f"❌ Error during token validation: {e.__class__.__name__}: {e}", exc_info=True)
        error_str = str(e).lower()
        # Check for specific error messages related to token validity
        if "invalid token" in error_str or "jwt" in error_str or "token is invalid" in error_str:
             raise Exception("Invalid token.") from e
        else:
             raise Exception("Token verification failed.") from e

# --- API Routes ---

@app.route("/healthz", methods=["GET", "HEAD"])
def health_check():
    """Basic health check endpoint."""
    return "ok", 200

# --- /generate_prompt (modified) ---
@app.route("/generate_prompt", methods=["POST"])
@app.route("/prompt_suggestion", methods=["POST", "OPTIONS"])
def generate_prompt():
    """Generates a prompt suggestion based on title, niche, and selected options."""
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return "", 204
    
    logging.info("📥 /generate_prompt or /prompt_suggestion called")
    data = request.json
    title = data.get("title")
    niche = data.get("niche")
    selected_options = data.get('options', []) # Default to empty list
    style_profile_json = data.get('style_profile_json') # Get the style profile JSON string
    logging.info(f"🕵️ Received style_profile_json raw: {style_profile_json!r} (Type: {type(style_profile_json)})") # Temp Info log

    if not title or not niche:
        logging.warning("⚠️ /generate_prompt: Missing title or niche.")
        return jsonify({"error": "Title and Niche are required."}), 400

    logging.info(f"🧠 Prompt generation request - Title: '{title}', Niche: '{niche}', Options: {selected_options}")

    if not client or not client.api_key or client.api_key == "dummy_key_if_missing": # Check for dummy key
        logging.error("❌ OpenAI client not configured for /generate_prompt.")
        return jsonify({"error": "AI Service (OpenAI) is not configured."}), 503

    try:
        options_string = ", ".join(selected_options) if selected_options else "None specified"
        # Modified instruction part
        # Start building the user message content
        user_message_content_parts = [
            "Generate ONE compelling, eyecatching, **and highly specific** visual prompt idea for a YouTube thumbnail based on this video information AND the following style/content options.",
            "The prompt MUST describe concrete visual elements suitable for a literal AI image generator (like Flux or DALL-E).",
            "",
            "**IMPORTANT INSTRUCTIONS for the prompt content:**",
            "* **Be Specific:** Describe exact objects, characters, settings, colors, lighting, and composition details.",
            "* **Avoid Vague Terms:** Do NOT use generic or abstract terms like 'analytics', 'data', 'symbols', 'icons', 'graphs' unless you describe *exactly* what they look like (e.g., 'a glowing blue bar chart showing an upward trend', 'a simple red heart icon floating near the top left'). If you can't describe it visually and specifically, don't include it.",
            "* **No Decorative Elements:** Avoid adding unnecessary decorative elements like confetti, sparkles, or generic symbols. Focus on core visual elements.",
            "* **Focus:** Ensure the prompt clearly focuses on the main subject derived from the title and niche.",
            "",
            f"Video Title: {title}",
            f"Channel Niche: {niche}",
            f"Selected Options/Requests: {options_string}",
            "",
            "Incorporate the selected style/content options naturally into the specific visual description."
        ]
        # --- Add Style Profile Instruction if available and valid --- 
        if style_profile_json and isinstance(style_profile_json, str) and style_profile_json.strip() and style_profile_json != '{}':
            try:
                # Attempt to parse to ensure it's valid JSON before sending
                json.loads(style_profile_json) 
                user_message_content_parts.extend([
                    "",
                    "**Style Profile:** Ensure the thumbnail adheres strictly to the rules defined in this JSON style profile:",
                    f"```json\n{style_profile_json}\n```" # Use markdown code block for clarity
                ])
                logging.info(f"✅ Including valid style profile JSON in prompt for GPT.")
            except json.JSONDecodeError:
                logging.warning(f"⚠️ Received style_profile_json that is not valid JSON. It will NOT be included in the prompt. Content: {style_profile_json[:200]}...") # Log truncated invalid JSON
        else:
            logging.info("No valid style profile JSON provided, skipping inclusion in prompt.")
        # ----------------------------------------------------------
        # Add the final instruction
        user_message_content_parts.append("\nReturn ONLY the visual prompt itself, without any extra text, formatting, or explanations.")
        # Join all parts into the final message
        user_message_content = "\n".join(user_message_content_parts)

        logging.info(f"✉️ Sending user message to GPT-4o-mini:\n{user_message_content}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                 {"role": "system", "content": "You are a viral YouTube thumbnail designer who generates visual prompt ideas for AI image generators, carefully considering user requests."},
                 {"role": "user", "content": user_message_content }
             ]
        )

        if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logging.error(f"❌ GPT prompt generation returned unexpected response structure: {response}")
            raise Exception("AI failed to generate a valid prompt structure.")

        prompt = response.choices[0].message.content.strip()
        if not prompt:
            logging.warning("⚠️ GPT returned an empty prompt.")
            raise Exception("AI failed to generate a non-empty prompt.")

        logging.info(f"✅ Prompt generated successfully (incorporating options): '{prompt}'")
        return jsonify({"prompt": prompt})

    except APIError as e:
        logging.error(f"❌ OpenAI API Error during prompt generation: {e}", exc_info=True)
        return jsonify({"error": f"AI prompt generation failed ({e.status_code}): {e.message or 'Unknown API Error'}"}), 502
    except Exception as e:
        logging.error(f"❌ Unexpected error in /generate_prompt: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during prompt generation: {e}"}), 500

# --- /get_profile (unchanged) ---
@app.route("/get_profile", methods=["GET"])
def get_profile():
    """Fetches the user's current credit balance."""
    logging.info("📥 /get_profile called")
    auth_header = request.headers.get("Authorization")
    user_id = None
    try:
        user_id = get_user_id_from_token(auth_header)
        logging.info(f"Fetching profile for user: {user_id}")

        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()

        if hasattr(profile_response, 'error') and profile_response.error:
            logging.error(f"❌ Supabase DB error fetching profile for {user_id}: {profile_response.error}")
            raise Exception(f"Database error fetching profile: {profile_response.error.message}")

        if profile_response and profile_response.data:
            credits = profile_response.data.get("credits", 0)
            logging.info(f"✅ Fetched credits for {user_id}: {credits}")
            return jsonify({"credits": credits})
        else:
            logging.warning(f"⚠️ Profile not found for {user_id}. Returning 0 credits.")
            return jsonify({"credits": 0})

    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""
        logging.error(f"❌ Error in /get_profile{user_id_str}: {e}", exc_info=True)
        error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message:
             status_code = 401; error_message = "Authentication failed. Please log in again."
        elif "Database error" in error_message:
             status_code = 503; error_message = "Failed to retrieve profile data due to a database issue."
        else: error_message = "An unexpected error occurred while fetching your profile."
        return jsonify({"error": error_message}), status_code

# --- /generate (Thumbnail - FAL) (modified) ---
@app.route("/generate", methods=["POST"])
def generate():
    """Generates 2 thumbnail images using Flux, processes (preview), saves, returns results."""
    logging.info("📥 /generate called (using Flux, requesting 2 images)")
    auth_header = request.headers.get("Authorization"); user_id = None; current_credits = 0
    NUM_IMAGES_TO_GENERATE = 2
    TOTAL_CREDIT_COST = CREDIT_COST_PER_THUMBNAIL * NUM_IMAGES_TO_GENERATE

    # 1. --- Authentication and Credit Check ---
    try:
        user_id = get_user_id_from_token(auth_header)
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()
        if hasattr(profile_response, 'error') and profile_response.error: raise Exception(f"DB error checking credits: {profile_response.error.message}")
        if not profile_response or not profile_response.data: logging.error(f"❌ Profile not found for user {user_id}."); return jsonify({"error": "User profile not found."}), 404
        current_credits = profile_response.data.get("credits", 0); logging.info(f"💰 User {user_id} has {current_credits} credits (needs {TOTAL_CREDIT_COST}).")
        if current_credits < TOTAL_CREDIT_COST: logging.warning(f"🚫 Insufficient credits for user {user_id}."); return jsonify({"error": f"Insufficient credits ({TOTAL_CREDIT_COST} needed)."}), 403
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""; logging.error(f"❌ Auth/Profile error in /generate{user_id_str}: {e}", exc_info=True); error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message: status_code = 401; error_message = "Authentication failed."
        elif "DB error" in error_message: status_code = 503; error_message = "Failed to check credits."
        elif "Insufficient credits" in error_message: status_code = 403; error_message = error_message
        else: error_message = "Error before generation."
        return jsonify({"error": error_message}), status_code

    # 2. --- Get Request Data ---
    data = request.json
    title = data.get("title", "")
    niche = data.get("niche", "")
    original_image_prompt = data.get("prompt")
    style_profile_json = data.get("style_profile_json") # This can be None or a dict - used only for record keeping
    if not original_image_prompt: logging.warning(f"⚠️ Missing 'prompt' for user {user_id}."); return jsonify({"error": "Image prompt required."}), 400
    logging.info(f" Rcvd Gen Request - User:{user_id}, Prompt:'{original_image_prompt}'")

    # Use only the original prompt from GPT-4o-mini without any modifications
    combined_prompt = original_image_prompt

    # Log that we're using only the original prompt without modifications
    logging.info(f"🔧 Using only original prompt for Flux: '{combined_prompt}'")

    # 3. --- Generate images with Flux ---
    generated_data_uris = []
    flux_errors = []
    logging.info(f"⏳ Requesting {NUM_IMAGES_TO_GENERATE} images from Flux for user {user_id}...")
    for i in range(NUM_IMAGES_TO_GENERATE):
        logging.info(f"   Generating image {i+1}/{NUM_IMAGES_TO_GENERATE}...")
        try:
            img_url = generate_flux_image(combined_prompt)
            logging.info(f"     Fetching image content from URL: {img_url}")
            img_response = requests.get(img_url, timeout=60); img_response.raise_for_status(); img_bytes = img_response.content
            # Determine mime type based on URL or headers (fallback to jpeg)
            content_type = img_response.headers.get('Content-Type', '').lower()
            if 'png' in content_type or img_url.lower().endswith('.png'): mime = 'png'
            elif 'jpeg' in content_type or 'jpg' in content_type or img_url.lower().endswith('.jpg') or img_url.lower().endswith('.jpeg'): mime = 'jpeg'
            else: mime = 'jpeg'; logging.warning(f"Unknown content type '{content_type}', assuming jpeg.")

            b64_encoded_data = base64.b64encode(img_bytes).decode('utf-8'); data_uri = f"data:image/{mime};base64,{b64_encoded_data}"
            generated_data_uris.append(data_uri); logging.info(f"   ✅ Image {i+1} fetched and encoded as {mime}.")
        except Exception as e: logging.error(f"❌ Flux/fetch failed image {i+1}: {e}", exc_info=True); flux_errors.append(f"Img {i+1}: {e}"); generated_data_uris.append(None)

    if not any(generated_data_uris): logging.error(f"❌ Flux generation failed entirely."); return jsonify({"error": f"Image generation failed: {'; '.join(flux_errors)}"}), 502
    logging.info(f"✅ Successfully generated {sum(1 for u in generated_data_uris if u)}/{NUM_IMAGES_TO_GENERATE} images.")

    # 5. --- Process Each Generated Image ---
    processed_results_urls = []
    db_save_tasks = []
    for index, original_uri in enumerate(generated_data_uris):
        if original_uri:
            logging.info(f"⚙️ Processing image {index + 1}...")
            # Generate JPEG preview for thumbnails for smaller size
            preview_uri = generate_preview_data_uri(original_uri, target_width=160, output_format='JPEG', quality=50) # Source can be JPEG preview
            processed_results_urls.append(original_uri) # Keep original full-res URI for display/download
            db_save_tasks.append({"user_id": user_id, "title": title, "niche": niche, "prompt": original_image_prompt, "image_url": original_uri, "preview_image_url": preview_uri})
        else:
            logging.warning(f"⚠️ Skipping processing for failed image {index + 1}.")

    # 6. --- Deduct Credits ---
    new_credits = max(0, current_credits - TOTAL_CREDIT_COST)
    try:
        logging.info(f"Attempting to deduct {TOTAL_CREDIT_COST} credits from user {user_id} (new balance: {new_credits}).")
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        if hasattr(update_response, 'error') and update_response.error: logging.error(f"⚠️ DB WARN: Failed to deduct credits for {user_id}: {update_response.error}")
        elif not (hasattr(update_response, 'data') and update_response.data): logging.warning(f"⚠️ DB WARN: Credit deduction no data for {user_id}.")
        else: logging.info(f"✅ Deducted {TOTAL_CREDIT_COST} credit(s) from {user_id}. New balance: {new_credits}")
    except Exception as credit_error:
        logging.error(f"❌ DB CRITICAL: Exception deducting credit for {user_id}: {credit_error}", exc_info=True)
        # Decide if we should still return images even if credit deduction fails? For now, yes.

    # 7. --- Save to History ---
    if db_save_tasks:
        try:
            logging.info(f"Attempting to save {len(db_save_tasks)} thumbnail records to history...")
            insert_response = supabase.table("thumbnail").insert(db_save_tasks).execute()
            if hasattr(insert_response, 'error') and insert_response.error: logging.error(f"⚠️ DB WARN: Failed to save batch history: {insert_response.error}")
            elif not (hasattr(insert_response, 'data') and insert_response.data): logging.warning(f"⚠️ DB WARN: Batch history insert returned no data.")
            else: logging.info(f"✅ Saved {len(insert_response.data)} thumbnails to history.")
        except Exception as history_error:
            logging.error(f"❌ DB CRITICAL: Exception saving batch history: {history_error}", exc_info=True)
            # History saving failure shouldn't prevent user from getting images
    else:
        logging.warning(f"No valid images to save to history.")

    # 8. --- Return Success Response ---
    if not processed_results_urls: logging.error(f"❌ No valid URLs to return."); return jsonify({"error": "Processing failed."}), 500
    logging.info(f"✅ Returning {len(processed_results_urls)} image URLs."); return jsonify({ "prompt": original_image_prompt, "image_urls": processed_results_urls, "new_credits": new_credits })

# --- /history (Thumbnail History) (unchanged) ---
@app.route("/history", methods=["GET"])
def get_history():
    """Fetches paginated thumbnail generation history for the user."""
    logging.info("📥 /history called")
    auth_header = request.headers.get("Authorization"); user_id = None
    try:
        user_id = get_user_id_from_token(auth_header)
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        if page < 1: page = 1
        limit = max(1, min(limit, 50)) # Allow up to 50, default 10
        offset = (page - 1) * limit
        logging.info(f"📦 Fetching history page {page} (limit {limit}) for user {user_id}...")
        # Select all needed fields including preview
        response = supabase.table("thumbnail").select(
            "id, created_at, title, niche, prompt, image_url, preview_image_url",
             count='exact' # Get total count for pagination
        ).eq("user_id", user_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()

        if hasattr(response, 'error') and response.error: raise Exception(f"DB error: {response.error.message}")
        history_data = response.data if hasattr(response, 'data') else []; total_count = response.count if hasattr(response, 'count') else 0
        logging.info(f"✅ History fetched page {page}: {len(history_data)} items (Total: {total_count})")
        return jsonify({ "items": history_data, "total": total_count, "page": page, "limit": limit })
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""; logging.error(f"❌ Error in /history{user_id_str}: {e}", exc_info=True); error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message: status_code = 401; error_message = "Authentication failed."
        elif "DB error" in error_message: status_code = 503; error_message = "Failed to get history."
        else: error_message = "Unexpected error getting history."
        return jsonify({"error": error_message}), status_code

# --- /extract_asset (OpenAI Image Edit) (updated) ---
@app.route('/extract_asset', methods=['POST'])
def extract_asset():
    """
    Receives an image and a text description, calls OpenAI Image Edit API
    to extract the described asset, generates previews, saves to history,
    and returns the result.
    """
    logging.info("📥 /extract_asset called")
    auth_header = request.headers.get("Authorization")
    user_id = None
    current_credits = 0
    cost = CREDIT_COST_PER_ASSET_EXTRACTION

    # 1. --- Authentication and Credit Check ---
    try:
        user_id = get_user_id_from_token(auth_header)
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()
        if hasattr(profile_response, 'error') and profile_response.error: raise Exception(f"DB error: {profile_response.error.message}")
        if not profile_response or not profile_response.data: logging.error(f"❌ Profile not found for user {user_id}."); return jsonify({"error": "User profile not found."}), 404
        current_credits = profile_response.data.get("credits", 0); logging.info(f"💰 User {user_id} has {current_credits} credits (needs {cost}).")
        if current_credits < cost: logging.warning(f"🚫 Insufficient credits for user {user_id}."); return jsonify({"error": f"Insufficient credits ({cost} needed)."}), 403
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""; logging.error(f"❌ Auth/Profile error in /extract_asset{user_id_str}: {e}", exc_info=True); error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message: status_code = 401; error_message = "Authentication failed."
        elif "DB error" in error_message: status_code = 503; error_message = "Failed to check credits."
        elif "Insufficient credits" in error_message: status_code = 403; error_message = error_message
        else: error_message = "Error before extraction."
        return jsonify({"error": error_message}), status_code

    # 2. --- Get Request Data & Prepare Image ---
    source_image_bytes = None # Initialize
    try:
        data = request.json
        source_image_data_uri = data.get("image_data_uri")
        asset_description = data.get("asset_description")
        if not source_image_data_uri or not asset_description: return jsonify({"error": "Missing image or description."}), 400
        if not source_image_data_uri.startswith('data:image'): return jsonify({"error": "Invalid image data format."}), 400

        header, encoded = source_image_data_uri.split(",", 1)
        logging.info(f"Received image data URI with header: {header}")
        image_format = header.split('/')[1].split(';')[0].lower()

        logging.info(f" Rcvd Asset Extract Request - User:{user_id}, Asset:'{asset_description}', Format:'{image_format}'")

        source_image_bytes_initial = base64.b64decode(encoded)

        # Ensure image is PNG for OpenAI edit endpoint
        if image_format != 'png':
            logging.warning(f"Input image was not PNG ('{image_format}'), attempting conversion...")
            try:
                img = Image.open(io.BytesIO(source_image_bytes_initial))
                logging.info(f"Opened image for conversion. Original format: {img.format}, Mode: {img.mode}")
                # Ensure RGBA mode for PNG output with transparency
                if img.mode != 'RGBA':
                     if 'A' not in img.getbands(): img.putalpha(255) # Add alpha if missing
                     img = img.convert("RGBA")
                     logging.info(f"Converting image mode {img.mode} to RGBA for PNG.")

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                source_image_bytes = buffer.getvalue()
                logging.info(f"Successfully converted input image to PNG bytes (Size: {len(source_image_bytes)}).")
                img.close()
            except UnidentifiedImageError:
                logging.error(f"❌ Pillow couldn't identify the image format '{image_format}' during conversion attempt.", exc_info=True)
                return jsonify({"error": "Could not process the uploaded image file. Please try a different image."}), 400
            except Exception as convert_err:
                logging.error(f"❌ Failed to convert input image to PNG: {convert_err}", exc_info=True)
                return jsonify({"error": "Input image must be PNG, or conversion failed."}), 400
        else:
             source_image_bytes = source_image_bytes_initial
             logging.info("Input image is already PNG.")

        # --- Verification Step ---
        if not source_image_bytes:
             logging.error("❌ source_image_bytes is empty or None before verification!")
             return jsonify({"error": "Internal error preparing image."}), 500
        logging.info(f"Verifying prepared image bytes (Size: {len(source_image_bytes)})...")
        try:
            with Image.open(io.BytesIO(source_image_bytes)) as verify_img:
                if verify_img.format != 'PNG':
                    logging.error(f"CRITICAL: Prepared image bytes are not recognized as PNG by Pillow! Format: {verify_img.format}")
                    raise ValueError("Internal error: Failed to prepare image data correctly.")
                logging.info(f"✅ Verified image bytes are readable as PNG (Mode: {verify_img.mode}) before sending to OpenAI.")
        except Exception as verify_err:
            logging.error(f"CRITICAL: Failed to verify prepared image bytes with Pillow: {verify_err}", exc_info=True)
            raise ValueError("Internal error: Image data verification failed.")

    except base64.binascii.Error as b64_err:
        logging.error(f"❌ Base64 decoding error: {b64_err}", exc_info=True)
        return jsonify({"error": "Invalid image data encoding."}), 400
    except Exception as e:
        logging.error(f"❌ Error parsing/processing request data: {e}", exc_info=True)
        if isinstance(e, ValueError) and "Internal error" in str(e): return jsonify({"error": str(e)}), 500
        return jsonify({"error": "Invalid request data or image processing failed."}), 400

    # --- 3. AI Call using OpenAI Image Edit API ---
    if not client or not client.api_key or client.api_key == "dummy_key_if_missing":
        logging.error("❌ OpenAI client not configured for /extract_asset.")
        return jsonify({"error": "AI Service (OpenAI) is not configured."}), 503

    extracted_asset_png_uri = None # Initialize here
    try:
        logging.info(f"🤖 Calling OpenAI Images Edit API for '{asset_description}'...")
        # Updated prompt for better transparency instruction
        extraction_prompt = f"Isolate and extract only '{asset_description}' object from the image onto a fully transparent background. Ensure ONLY the described object remains, and the background is entirely transparent (alpha=0)."
        logging.info(f"   Using prompt: '{extraction_prompt}'")

        # Prepare image data as a file tuple for the API
        image_tuple_for_api = ("image.png", source_image_bytes, "image/png")

        response = client.images.edit(
            model="gpt-image-1", # Use GPT-4 Vision model for better extraction
            image=image_tuple_for_api,
            prompt=extraction_prompt,
            n=1,
            size="1024x1024" # Ensure size is supported
        )

        # Handle the response format from gpt-image-1 model
        if not response or not response.data:
             logging.error(f"❌ OpenAI Edit API returned unexpected response: {response}")
             # Log details if possible
             if response: logging.error(f"Response received: {response}")
             raise ValueError("AI model did not return valid image data.")
        
        # Check if response has url or b64_json
        image_data = response.data[0]
        if hasattr(image_data, 'b64_json') and image_data.b64_json:
            # If b64_json is available
            b64_json_data = image_data.b64_json
            extracted_asset_png_uri = f"data:image/png;base64,{b64_json_data}"
            logging.info(f"✅ OpenAI Edit API successful for user {user_id}. Received b64_json.")
        elif hasattr(image_data, 'url') and image_data.url:
            # If URL is returned instead, fetch the image
            logging.info(f"OpenAI returned image URL instead of b64_json. Fetching from: {image_data.url}")
            try:
                img_response = requests.get(image_data.url, timeout=30)
                img_response.raise_for_status()
                img_bytes = img_response.content
                b64_data = base64.b64encode(img_bytes).decode('utf-8')
                extracted_asset_png_uri = f"data:image/png;base64,{b64_data}"
                logging.info(f"✅ Successfully fetched image from URL for user {user_id}.")
            except Exception as url_err:
                logging.error(f"❌ Failed to fetch image from URL: {url_err}", exc_info=True)
                raise ValueError(f"Failed to fetch result image: {str(url_err)}")
        else:
            logging.error(f"❌ OpenAI response missing both b64_json and url: {image_data}")
            raise ValueError("AI model returned data in an unexpected format.")

    except APIError as e:
        logging.error(f"❌ OpenAI API Error during asset extraction: {e}", exc_info=True)
        error_msg = f"AI extraction failed ({getattr(e, 'status_code', 'N/A')})"
        status_code = getattr(e, 'status_code', 502)
        # Try to get more specific error message from the response body
        api_message = getattr(e, 'message', None)
        if not api_message and hasattr(e, 'body') and isinstance(e.body, dict):
            api_message = e.body.get('error', {}).get('message', 'Unknown API Error')
        error_msg = f"{error_msg}: {api_message or 'Unknown API Error'}"

        # Handle specific common errors
        if status_code == 400:
            if 'Unsupported image format' in str(api_message): error_msg = "Extraction failed: Invalid image format sent to AI (must be PNG)."
            elif 'content policy' in str(api_message).lower(): error_msg = "Extraction failed due to content policy violation."; status_code = 400
            elif 'mask' in str(api_message).lower(): error_msg = "Extraction failed: OpenAI API might require a mask for this operation type (internal configuration issue)." # Should not happen with prompt only

        return jsonify({"error": error_msg}), status_code
    except Exception as ai_error:
        logging.error(f"❌ Error during OpenAI call or processing: {ai_error}", exc_info=True)
        if isinstance(ai_error, ValueError) and "expected b64_json" in str(ai_error):
             return jsonify({"error": "AI response format was unexpected (Missing b64_json)."}), 500
        elif isinstance(ai_error, ValueError) and "Internal error" in str(ai_error):
             return jsonify({"error": str(ai_error)}), 500
        return jsonify({"error": f"Failed to extract asset via AI: {str(ai_error)}"}), 500

    # --- Ensure extracted_asset_png_uri was set ---
    if not extracted_asset_png_uri:
        logging.error("❌ Failed to obtain Data URI from API response after processing.")
        return jsonify({"error": "Internal error processing final image."}), 500

    # --- 4. Generate Previews ---
    logging.info("🖼️ Generating previews for asset history...")
    # Create final source URI from verified bytes
    final_source_data_uri = f"data:image/png;base64,{base64.b64encode(source_image_bytes).decode('utf-8')}"
    source_preview_uri = generate_preview_data_uri(final_source_data_uri, target_width=160, output_format='JPEG', quality=50) # Source can be JPEG preview
    extracted_preview_uri = generate_preview_data_uri(extracted_asset_png_uri, target_width=160, output_format='PNG') # Extracted should be PNG preview

    # --- 5. Deduct Credits ---
    new_credits = max(0, current_credits - cost)
    try:
        logging.info(f"Attempting to deduct {cost} credits (asset extraction) from user {user_id} (new balance: {new_credits}).")
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        if hasattr(update_response, 'error') and update_response.error: logging.error(f"⚠️ DB WARN: Failed to deduct credits: {update_response.error}")
        elif not (hasattr(update_response, 'data') and update_response.data): logging.warning(f"⚠️ DB WARN: Credit deduction no data returned.")
        else: logging.info(f"✅ Deducted {cost} credit(s). New balance: {new_credits}")
    except Exception as credit_error:
        logging.error(f"❌ DB CRITICAL: Exception deducting credit: {credit_error}", exc_info=True)

    # --- 6. Save to Asset History Table ---
    history_entry = {
         "user_id": user_id, "asset_description": asset_description,
         "source_image_preview_url": source_preview_uri, # Store JPEG preview of source
         "extracted_asset_url": extracted_asset_png_uri, # Store the full PNG Data URI result
         "extracted_asset_preview_url": extracted_preview_uri # Store PNG preview of result
    }
    try:
        insert_response = supabase.table("asset_history").insert(history_entry).execute()
        if hasattr(insert_response, 'error') and insert_response.error: logging.error(f"⚠️ DB WARN: Failed to save asset history: {insert_response.error}")
        else: logging.info(f"✅ Saved asset extraction to history.")
    except Exception as history_error:
        logging.error(f"❌ DB CRITICAL: Exception saving asset history: {history_error}", exc_info=True)

    # --- 7. Return Success Response ---
    return jsonify({
         "result_png_uri": extracted_asset_png_uri, # Return the final Data URI
         "new_credits": new_credits
    })

# --- /asset_history (unchanged) ---
@app.route("/asset_history", methods=["GET"])
def get_asset_history():
    """Fetches paginated asset extraction history for the user."""
    logging.info("📥 /asset_history called")
    auth_header = request.headers.get("Authorization"); user_id = None
    try:
        user_id = get_user_id_from_token(auth_header)
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 5, type=int) # Default 5 per page
        if page < 1: page = 1
        limit = max(1, min(limit, 25)) # Allow up to 25 per page
        offset = (page - 1) * limit
        logging.info(f"📦 Fetching asset history page {page} (limit {limit}) for user {user_id}...")
        response = supabase.table("asset_history").select(
            "id, created_at, asset_description, source_image_preview_url, extracted_asset_url, extracted_asset_preview_url",
             count='exact' # Get total count
        ).eq("user_id", user_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()

        if hasattr(response, 'error') and response.error: raise Exception(f"DB error: {response.error.message}")
        history_data = response.data if hasattr(response, 'data') else []; total_count = response.count if hasattr(response, 'count') else 0
        logging.info(f"✅ Asset history fetched page {page}: {len(history_data)} items (Total: {total_count})")
        return jsonify({ "items": history_data, "total": total_count, "page": page, "limit": limit })
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""; logging.error(f"❌ Error in /asset_history{user_id_str}: {e}", exc_info=True); error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message: status_code = 401; error_message = "Authentication failed."
        elif "DB error" in error_message: status_code = 503; error_message = "Failed to get asset history."
        else: error_message = "Unexpected error getting asset history."
        return jsonify({"error": error_message}), status_code

# --- /create-checkout-session (unchanged) ---
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    """Creates a Stripe Checkout session for purchasing credits."""
    logging.info("📥 /create-checkout-session called")
    auth_header = request.headers.get("Authorization"); data = request.json; user_id = None
    price_id = data.get('priceId')
    if not price_id: logging.warning("❌ Price ID missing."); return jsonify({"error": "Price ID is required."}), 400
    # Use FRONTEND_DOMAIN env var for redirects
    YOUR_FRONTEND_DOMAIN = allowed_origins_str.split(',')[0].strip() # Use the first allowed origin
    logging.info(f"Using frontend domain for Stripe redirects: {YOUR_FRONTEND_DOMAIN}")

    if not stripe or not stripe.api_key: logging.error("❌ Stripe not configured."); return jsonify({"error": "Payment system not configured."}), 503

    try:
        user_id = get_user_id_from_token(auth_header); logging.info(f"✅ User {user_id} checkout for price ID: {price_id}")
        if price_id not in PRICE_ID_TO_CREDITS: logging.error(f"❌ Unrecognized Price ID: {price_id}"); return jsonify({"error": "Invalid purchase option."}), 400

        success_url=f"{YOUR_FRONTEND_DOMAIN}/app/?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{YOUR_FRONTEND_DOMAIN}/app/?payment=cancel"
        checkout_session = stripe.checkout.Session.create(
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='payment',
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                'supabase_user_id': str(user_id),
                'price_id': price_id
            }
        )
        logging.info(f"✅ Stripe session created: {checkout_session.id}"); return jsonify({'url': checkout_session.url})
    except stripe.error.StripeError as e: logging.error(f"❌ Stripe Error: {e}", exc_info=True); return jsonify({"error": f"Stripe error: {getattr(e, 'user_message', 'Payment error.')}"}), 500
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""; logging.error(f"❌ Error creating checkout{user_id_str}: {e}", exc_info=True); error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message: status_code = 401; error_message = "Authentication failed."
        else: error_message = "Failed to create checkout session."
        return jsonify({"error": error_message}), status_code

# --- /stripe-webhook (unchanged) ---
@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    """Handles incoming Stripe webhook events, specifically checkout.session.completed."""
    logging.info("🔔 /stripe-webhook received event")
    payload = request.data; sig_header = request.headers.get('Stripe-Signature'); event = None
    webhook_secret = STRIPE_WEBHOOK_SECRET
    if not webhook_secret: logging.error("❌ Webhook Secret missing."); return jsonify(status="Webhook config error on server"), 200
    if not stripe or not stripe.api_key: logging.error("❌ Stripe not configured for webhook."); return jsonify(status="Stripe config error on server"), 200

    try: event = stripe.Webhook.construct_event( payload, sig_header, webhook_secret )
    except ValueError as e: logging.error(f"❌ Invalid payload: {e}"); return jsonify(error="Invalid payload"), 400
    except stripe.error.SignatureVerificationError as e: logging.error(f"❌ Invalid signature: {e}"); return jsonify(error="Invalid signature"), 400
    except Exception as e: logging.error(f"❌ Error constructing event: {e}"); return jsonify(error="Webhook processing error"), 500
    logging.info(f"✅ Webhook verified. Type: {event.get('type')}")

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']; session_id = session.get('id'); logging.info(f"Processing completed session: {session_id}")
        if session.get('payment_status') == 'paid':
            metadata = session.get('metadata')
            if not metadata: logging.error(f"❌ Metadata missing: {session_id}"); return jsonify(status="Metadata missing"), 200
            user_id = metadata.get('supabase_user_id'); price_id = metadata.get('price_id')
            if not user_id or not price_id: logging.error(f"❌ Missing metadata fields: {session_id}"); return jsonify(status="Required metadata missing"), 200
            credits_to_add = PRICE_ID_TO_CREDITS.get(price_id)
            if credits_to_add is None: logging.error(f"⚠️ Unrecognized price_id '{price_id}': {session_id}"); return jsonify(status="Unrecognized price ID"), 200
            logging.info(f"Attempting grant {credits_to_add} credits to user {user_id}...")
            try: # Upsert credits - fetch current first
                profile_res = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()
                current_credits = 0
                if hasattr(profile_res, 'error') and profile_res.error:
                    logging.error(f"❌ DB Error fetching current credits for webhook: {profile_res.error}")
                    # Optionally decide whether to proceed or return error
                    # For now, we'll assume 0 if fetch fails but user exists
                elif profile_res and profile_res.data:
                    current_credits = profile_res.data.get("credits", 0)
                else:
                    logging.warning(f"⚠️ Profile not found for {user_id} during webhook. Will insert/upsert with new credits.")

                new_total_credits = current_credits + credits_to_add
                # Use upsert to handle both new and existing profiles
                upsert_response = supabase.table("profiles").upsert({ "id": user_id, "credits": new_total_credits }).execute()
                if hasattr(upsert_response, 'error') and upsert_response.error:
                    logging.error(f"❌ DB Error upserting credits: {upsert_response.error}")
                    # Consider retrying or alerting
                else:
                     logging.info(f"✅ Credits updated for {user_id} to {new_total_credits}")
            except Exception as db_error: logging.error(f"❌ DB Exception during credit update: {db_error}", exc_info=True)
            # Important: Still return 200 to Stripe even if DB update fails, to avoid repeated webhook calls for this issue. Log thoroughly.
        else: logging.info(f"Session {session_id} status: '{session.get('payment_status')}' - no credits added.")
    else: logging.info(f"🤷\u200D♀️ Unhandled event type: {event.get('type')}")
    return jsonify(success=True), 200


# --- Style Profiles API Routes ---
@app.route("/style_profiles", methods=["GET"])
def get_style_profiles():
    """Get all style profiles for the current user."""
    # Get JWT token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication required'}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        # Verify the JWT token with Supabase
        user = supabase.auth.get_user(token)
        user_id = user.user.id
        
        # Query style profiles for the current user
        response = supabase.table('style_profiles').select('*').eq('user_id', user_id).execute()
        
        # Return the profiles
        return jsonify({'profiles': response.data})
    except Exception as e:
        logging.error(f"Error getting style profiles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/style_profiles", methods=["POST"])
def create_style_profile():
    """Create a new style profile."""
    # Get JWT token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication required'}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        # Verify the JWT token with Supabase
        user = supabase.auth.get_user(token)
        user_id = user.user.id
        
        # Get request data
        data = request.json
        if not data or not data.get('name') or not data.get('profile'):
            return jsonify({'error': 'Name and profile are required'}), 400
        
        # Create new style profile
        profile_data = {
            'user_id': user_id,
            'name': data['name'],
            'profile': data['profile'],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        response = supabase.table('style_profiles').insert(profile_data).execute()
        
        # Return the created profile
        return jsonify({'profile': response.data[0]})
    except Exception as e:
        logging.error(f"Error creating style profile: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/style_profiles/<profile_id>", methods=["PUT"])
def update_style_profile(profile_id):
    """Update an existing style profile."""
    # Get JWT token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication required'}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        # Verify the JWT token with Supabase
        user = supabase.auth.get_user(token)
        user_id = user.user.id
        
        # Get request data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare update data
        update_data = {}
        if 'name' in data:
            update_data['name'] = data['name']
        if 'profile' in data:
            update_data['profile'] = data['profile']
        
        update_data['updated_at'] = datetime.now().isoformat()
        
        # Update the profile
        response = supabase.table('style_profiles').update(update_data).eq('id', profile_id).eq('user_id', user_id).execute()
        
        if not response.data:
            return jsonify({'error': 'Profile not found or not authorized'}), 404
        
        # Return the updated profile
        return jsonify({'profile': response.data[0]})
    except Exception as e:
        logging.error(f"Error updating style profile: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/style_profiles/<profile_id>", methods=["DELETE"])
def delete_style_profile(profile_id):
    """Delete a style profile."""
    # Get JWT token from Authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Authentication required'}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        # Verify the JWT token with Supabase
        user = supabase.auth.get_user(token)
        user_id = user.user.id
        
        # Delete the profile
        response = supabase.table('style_profiles').delete().eq('id', profile_id).eq('user_id', user_id).execute()
        
        if not response.data:
            return jsonify({'error': 'Profile not found or not authorized'}), 404
        
        # Return success
        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Error deleting style profile: {e}")
        return jsonify({'error': str(e)}), 500

# --- Style Profile Analysis Route ---
@app.route("/analyze_style", methods=["POST"])
def analyze_style():
    """Analyzes uploaded thumbnails and generates a style profile using ChatGPT."""
    logging.info("📥 /analyze_style called")
    auth_header = request.headers.get("Authorization")
    user_id = None
    current_credits = 0
    cost = CREDIT_COST_PER_STYLE_ANALYSIS

    # --- 1. Authentication & Credit Check ---
    try:
        user_id = get_user_id_from_token(auth_header)
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()
        if hasattr(profile_response, 'error') and profile_response.error: raise Exception(f"DB error checking credits: {profile_response.error.message}")
        if not profile_response or not profile_response.data: logging.error(f"❌ Profile not found for user {user_id}."); return jsonify({"error": "User profile not found."}), 404
        current_credits = profile_response.data.get("credits", 0); logging.info(f"💰 User {user_id} has {current_credits} credits (needs {cost}).")
        if current_credits < cost: logging.warning(f"🚫 Insufficient credits for user {user_id}."); return jsonify({"error": f"Insufficient credits ({cost} needed)."}), 403
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""; logging.error(f"❌ Auth/Profile error in /analyze_style{user_id_str}: {e}", exc_info=True); error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message: status_code = 401; error_message = "Authentication failed."
        elif "DB error" in error_message: status_code = 503; error_message = "Failed to check credits."
        elif "Insufficient credits" in error_message: status_code = 403; error_message = error_message
        else: error_message = "Error before style analysis."
        return jsonify({"error": error_message}), status_code

    # --- 2. Get Input Data (multipart/form-data) ---
    try:
        thumbnail_files = []
        for key in request.files:
            if key.startswith('thumbnail_'):
                thumbnail_files.append(request.files[key])

        if not thumbnail_files:
            logging.warning(f"⚠️ No thumbnail files uploaded for style analysis (user {user_id}).")
            return jsonify({"error": "Please upload at least one thumbnail image."}), 400

        # Basic validation of uploaded files
        for idx, thumb_file in enumerate(thumbnail_files):
            if not thumb_file.mimetype or not thumb_file.mimetype.startswith('image/'):
                logging.warning(f"⚠️ Invalid mimetype '{thumb_file.mimetype}' for thumbnail {idx} (user {user_id}).")
                return jsonify({"error": f"Invalid file type uploaded for thumbnail {idx+1}. Please upload images only."}), 400

        logging.info(f"Received Style Analysis Request - User:{user_id}, Thumbnails:{len(thumbnail_files)}")

        # Convert images to base64 for OpenAI
        thumbnail_base64s = []
        for thumb_file in thumbnail_files:
            image_bytes = thumb_file.read()
            if not image_bytes:
                logging.error(f"❌ Failed to read bytes from uploaded thumbnail file for user {user_id}.")
                continue
            b64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            thumbnail_base64s.append(b64_encoded)

        if not thumbnail_base64s:
            logging.error(f"❌ Failed to process any thumbnail images for user {user_id}.")
            return jsonify({"error": "Could not process uploaded images."}), 400

    except Exception as e:
        logging.error(f"❌ Error processing style analysis request data for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to process request data."}), 400

    # --- 3. Check OpenAI Configuration ---
    if not OPENAI_API_KEY or OPENAI_API_KEY == "dummy_key_if_missing":
        logging.error("❌ OpenAI API Key not configured.")
        return jsonify({"error": "Style analysis service is not configured."}), 503

    # --- 4. Call OpenAI Vision API ---
    try:
        logging.info(f"🤖 Calling OpenAI Vision API for style analysis for user {user_id}...")
        
        # Prepare the message with images
        messages = [
            {
                "role": "system",
                "content": "You are a thumbnail style analyzer. Your task is to analyze the provided thumbnails and create a JSON profile that captures their style characteristics. Focus on visual elements, colors, composition, typography, and overall aesthetic. The JSON should be structured to help AI replicate similar thumbnails."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
Analyze these thumbnails and create a comprehensive JSON style profile with extraordinary detail. Include:

- Color analysis: Identify exact colors (hex codes), color relationships, contrast levels, saturation choices, color psychology, and how colors are distributed across the thumbnails
                
- Visual composition: Describe the precise layout structure, focal points, rule of thirds usage, foreground/background separation, depth effects, negative space usage, and how elements are balanced

- Typography details: Font styles, sizing ratios between headers and subtext, text positioning patterns, emphasis techniques, text effects (shadows, outlines, etc.), and how text interacts with imagery
                
- Technical elements: Image quality, sharpness levels, filter usage, special effects, editing techniques, blending modes, texture applications
                
- Subject treatment: How people/objects are positioned, expressions if people are present, cropping patterns, scaling techniques, and subject emphasis methods
                
- Emotional impact: The intended psychological effect (urgency, curiosity, excitement), emotional tone, how the thumbnail creates viewer engagement
                
- Distinctive patterns: Any recurring motifs, signature techniques, or unique stylistic choices that define this thumbnail style

Provide extensive detail on ALL of these aspects - be comprehensive and specific so an AI image generator could recreate the exact style.
"""}
                ]
            }
        ]
        
        # Add each image to the user message content
        for b64_img in thumbnail_base64s:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for vision capabilities
            messages=messages,
            max_tokens=2000
        )
        
        # Extract the JSON profile from the response
        style_profile = response.choices[0].message.content
        
        # Try to clean up the response if it contains markdown code blocks
        if "```json" in style_profile:
            style_profile = style_profile.split("```json")[1].split("```")[0].strip()
        elif "```" in style_profile:
            style_profile = style_profile.split("```")[1].split("```")[0].strip()
            
        # Validate that it's proper JSON
        try:
            json.loads(style_profile)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract just the JSON part
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, style_profile)
            if json_match:
                style_profile = json_match.group(0)
                # Validate again
                json.loads(style_profile)  # This will raise an exception if still invalid
            else:
                raise ValueError("Could not extract valid JSON from the response")
        
        logging.info(f"✅ OpenAI style analysis successful for user {user_id}.")

    except Exception as e:
        logging.error(f"❌ Error during OpenAI API call for style analysis for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": f"Style analysis failed: {str(e)}"}), 500

    # --- 5. Deduct Credits ---
    new_credits = max(0, current_credits - cost)
    try:
        logging.info(f"Attempting to deduct {cost} credits (style analysis) from user {user_id} (new balance: {new_credits}).")
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        if hasattr(update_response, 'error') and update_response.error: logging.error(f"⚠️ DB WARN: Failed to deduct credits: {update_response.error}")
        elif not (hasattr(update_response, 'data') and update_response.data): logging.warning(f"⚠️ DB WARN: Credit deduction no data returned.")
        else: logging.info(f"✅ Deducted {cost} credit(s). New balance: {new_credits}")
    except Exception as credit_error:
        logging.error(f"❌ DB CRITICAL: Exception deducting credit: {credit_error}", exc_info=True)

    # --- 6. Return Success Response ---
    logging.info(f"✅ Returning style profile for user {user_id}.")
    return jsonify({
        "style_profile": style_profile,
        "new_credits": new_credits
    })

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) # Default 5000
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    debug_mode_str = os.environ.get("FLASK_DEBUG", "True") # Default True for dev
    debug_mode = debug_mode_str.lower() in ["true", "1", "t", "yes"]

    log_level = logging.DEBUG if debug_mode else logging.INFO
    # Force=True is important if basicConfig was called implicitly before (e.g., by a library)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    app.logger.setLevel(log_level) # Ensure Flask's logger uses the same level

    logging.info(f"Effective root logging level: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    logging.info(f"Flask app logger level: {logging.getLevelName(app.logger.getEffectiveLevel())}")

    if debug_mode: logging.warning("⚠️ Flask running in DEBUG mode!")
    else: logging.info("Flask running in PRODUCTION mode (Debug=False).")

    logging.info(f"🚀 Starting Flask server on http://{host}:{port} (Debug: {debug_mode})")
    # Turn off reloader in production explicitly if needed, debug=debug_mode usually handles this
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)