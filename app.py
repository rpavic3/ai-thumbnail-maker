# Required Imports
from supabase import create_client, Client
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI, APIError # v1.x+ library (Still needed for prompt generation)
import json
import requests # Needed for Flux and fetching image URL
import os
from dotenv import load_dotenv
import stripe
import logging
import base64 # Needed for decoding/encoding Flux image
import io     # Needed for handling image bytes
from PIL import Image, UnidentifiedImageError # Pillow for image processing (preview)

# --- Load Environment Variables FIRST ---
load_dotenv() # Loads variables from .env file into environment

# --- Setup Logging ---
# Default to INFO level. Will be adjusted in __main__ if FLASK_DEBUG is set.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Supabase Configuration ---
# Get values directly from environment (loaded from .env)
SUPABASE_URL = "https://mjwjxxfnqbaroxwjewms.supabase.co" # Keep your actual URL
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# Check if variables were loaded successfully
if not SUPABASE_URL: logging.error("❌ Missing Supabase URL environment variable."); raise ValueError("Supabase URL missing.")
if not SUPABASE_SERVICE_ROLE_KEY: logging.error("❌ Missing Supabase Service Role Key environment variable."); raise ValueError("Supabase Key missing.")
try: supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY); logging.info("✅ Supabase client initialized.")
except Exception as e: logging.error(f"❌ Failed Supabase init: {e}", exc_info=True); raise

# --- OpenAI Configuration (v1.x+) - Still needed for prompt generation ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: logging.warning("⚠️ OpenAI API Key environment variable missing or empty.")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("✅ OpenAI client initialized (v1.x+).")
except APIError as e: logging.error(f"❌ OpenAI API Error client init: {e}"); raise ValueError("OpenAI client failed init.") from e
except Exception as e: logging.error(f"❌ Failed OpenAI client init: {e}"); raise ValueError("OpenAI client failed init.") from e

# --- Flux (FAL) Configuration ---
FAL_API_KEY = os.getenv("FAL_API_KEY") # ⬅️ Added FAL API Key loading
if not FAL_API_KEY: logging.warning("⚠️ FAL_API_KEY environment variable missing or empty.")

# --- Stripe Configuration ---
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
if not stripe.api_key: logging.warning("⚠️ Stripe Secret Key environment variable missing.")
if not STRIPE_WEBHOOK_SECRET: logging.warning("⚠️ Stripe Webhook Secret environment variable missing.")

# --- Stripe Price ID Mapping ---
# Load Price ID and corresponding credits from environment variables
PRICE_ID_TO_CREDITS = {}
# Example for one price package:
# Ensure these env var names match your .env file exactly
price_id_env_var_name = "STRIPE_PRICE_ID_50_CREDITS"
credits_env_var_name = "CREDITS_FOR_50_PACK"
default_credits = "50" # Default credits if env var for credits is missing

price_id_pack1 = os.environ.get("STRIPE_PRICE_ID_50_CREDITS")
credits_pack1_str = os.environ.get(credits_env_var_name, default_credits)

if price_id_pack1:
    try:
        credits_pack1 = int(credits_pack1_str)
        PRICE_ID_TO_CREDITS[price_id_pack1] = credits_pack1
        logging.info(f"✅ Loaded Stripe Price ID: {price_id_pack1} -> {credits_pack1} credits")
    except ValueError:
        logging.error(f"❌ Invalid credit amount '{credits_pack1_str}' (from env var {credits_env_var_name}) for price ID {price_id_pack1}. Must be an integer.")
else:
     logging.warning(f"⚠️ {price_id_env_var_name} environment variable not set.")

# Add more blocks here if you have multiple price packages defined in .env

if not PRICE_ID_TO_CREDITS: logging.warning("⚠️ PRICE_ID_TO_CREDITS mapping is empty. Stripe purchases might fail.")


# --- Flask App Initialization ---
app = Flask(__name__)
# Configure CORS properly for production
# Get allowed frontend origin(s) from environment variable
allowed_origins = os.environ.get("FRONTEND_DOMAIN", "http://127.0.0.1:5500") # Default to local dev
# Split if multiple origins are provided, separated by commas
cors_origins = [origin.strip() for origin in allowed_origins.split(',')]
CORS(app, origins=cors_origins, supports_credentials=True) # Allow specific origins
logging.info(f"✅ CORS configured for origins: {cors_origins}")


# --- Constants for Image Processing ---
PREVIEW_WIDTH = 320 # Target width for previews in pixels
# TOP_BOTTOM_CROP_PIXELS = 62 # Pixels to crop from top and bottom (No longer used with Flux)
# EXPECTED_WIDTH = 1536 # Expected width from OpenAI (No longer used with Flux)
# EXPECTED_HEIGHT = 1024 # Expected height from OpenAI (No longer used with Flux)
CREDIT_COST_PER_IMAGE = 1 # Cost for EACH image generated

# --- Helper Function: Generate Flux Image (Unchanged from previous version) ---
def generate_flux_image(prompt: str) -> str:
    """Calls FAL/Flux-Pro and returns the public image URL"""
    if not FAL_API_KEY:
        logging.error("❌ Cannot generate Flux image: FAL_API_KEY is not set.")
        raise ValueError("Flux API Key is missing.") # Raise error if key is missing

    # Log the *full* prompt being sent now
    logging.info(f"🌀 Sending prompt to Flux: '{prompt}'")
    api_url = "https://fal.run/fal-ai/flux-pro/v1.1-ultra"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    # Flux payload expects 1280x720
    payload = {
        "prompt": prompt, # Send the full combined prompt
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "width": 1280,
        "height": 720
    }
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=120) # Add timeout
        r.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
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
    except Exception as e: # Catch other errors like JSONDecodeError or raise_for_status() errors
         logging.error(f"❌ Failed Flux API call: {e}", exc_info=True)
         # Try to include response text if available
         error_detail = getattr(e, 'response', None)
         error_text = getattr(error_detail, 'text', str(e))
         raise Exception(f"Flux generation failed: {error_text}") from e


# --- Helper Function: Generate Preview Image Data URI (Unchanged) ---
def generate_preview_data_uri(full_data_uri: str, target_width: int = PREVIEW_WIDTH) -> str | None:
    """
    Decodes a base64 Data URI, resizes the image to a target width
    (maintaining aspect ratio), and re-encodes it as a new base64 Data URI (JPEG).
    Returns None if resizing fails.
    """
    logging.info(f"🖼️ Generating preview (target width: {target_width}px)...")
    try:
        # Basic check for Data URI format
        if not full_data_uri or not full_data_uri.startswith("data:image/") or ";base64," not in full_data_uri:
            logging.warning(f"⚠️ Invalid or missing Data URI for preview generation. Cannot generate preview.")
            return None

        header, encoded_data = full_data_uri.split(',', 1)

        # image_format = header.split('/')[1].split(';')[0] # Format not strictly needed here
        image_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(image_data))

        # Ensure image is in RGB mode for JPEG saving
        if image.mode != 'RGB':
             image = image.convert("RGB")
             logging.info("Converted image to RGB for JPEG preview.")

        width, height = image.size
        if width <= 0 or height <= 0: logging.warning(f"⚠️ Image has invalid dimensions ({width}x{height}). Cannot generate preview."); return None

        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)

        if target_width <= 0 or target_height <= 0:
            logging.warning(f"⚠️ Invalid target dimensions ({target_width}x{target_height}). Cannot generate preview.")
            return None

        # Use recommended resampling filter
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.ANTIALIAS
        preview_image = image.resize((target_width, target_height), resample_filter)
        logging.info(f"✅ Resized preview to {target_width}x{target_height}.")

        buffer = io.BytesIO()
        # Lower JPEG quality for smaller preview file size
        preview_image.save(buffer, format='JPEG', quality=60)
        buffer.seek(0)

        preview_encoded_data = base64.b64encode(buffer.read()).decode('utf-8')
        preview_data_uri = f"data:image/jpeg;base64,{preview_encoded_data}" # Use jpeg type

        logging.info("✅ Preview image generated and encoded as JPEG Data URI.")
        return preview_data_uri
    except base64.binascii.Error as b64_err:
         logging.error(f"❌ Base64 decoding error during preview generation: {b64_err}", exc_info=True)
         return None
    except UnidentifiedImageError: # Catch Pillow specific error for bad image data
         logging.error("❌ Pillow UnidentifiedImageError: Could not open image data for preview.", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"❌ Unexpected error generating preview image: {e}", exc_info=True)
        return None

# --- Helper Function: Crop Image Data URI (No longer used by default, kept for potential future use) ---
# def crop_image_data_uri(data_uri: str) -> str | None:
#     [...] # Code remains commented out


@app.route("/", methods=["GET", "HEAD"])
def health_root():
    """Basic health check endpoint."""
    return "ok", 200


# --- API Route: Generate Prompt (MODIFIED to accept options) ---
@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    """Generates a prompt suggestion based on title, niche, and selected options."""
    logging.info("📥 /generate_prompt called")
    data = request.json
    title = data.get("title")
    niche = data.get("niche")
    # --- Get the new 'options' parameter from the request ---
    # Default to an empty list if not provided
    selected_options = data.get("options", [])

    if not title or not niche:
        logging.warning("⚠️ /generate_prompt: Missing title or niche.")
        return jsonify({"error": "Title and Niche are required."}), 400

    # Log the received options
    logging.info(f"🧠 Prompt generation request - Title: '{title}', Niche: '{niche}', Options: {selected_options}")

    if not client or not client.api_key:
        logging.error("❌ OpenAI client not configured for /generate_prompt.")
        return jsonify({"error": "AI Service is not configured."}), 503 # Service Unavailable

    try:
        # --- Construct the user message for GPT-4o-mini, including options ---
        options_string = ", ".join(selected_options) if selected_options else "None specified"

        user_message_content = f"""Generate ONE compelling, eyecatching visual prompt idea for a YouTube thumbnail based on this video information AND the following style/content options. The prompt should be suitable for an AI image generator like Flux or DALL-E. Focus on creating a visually engaging scene.

Video Title: {title}
Channel Niche: {niche}
Selected Options/Requests: {options_string}

Incorporate the selected options naturally into the visual description.

Return ONLY the visual prompt itself, without any extra text, formatting, or explanations."""
        # --- End of modified user message ---

        logging.debug(f"✉️ Sending user message to GPT-4o-mini:\n{user_message_content}") # Log the full message in debug

        # Using gpt-4o-mini for prompt generation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7, # Balance creativity and coherence
            messages=[
                 {"role": "system", "content": "You are a viral YouTube thumbnail designer who generates visual prompt ideas for AI image generators, carefully considering user requests."}, # Slightly updated system prompt
                 {"role": "user", "content": user_message_content } # Use the new message with options
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
        return jsonify({"error": f"AI prompt generation failed ({e.status_code}): {e.message or 'Unknown API Error'}"}), 502 # Bad Gateway
    except Exception as e:
        logging.error(f"❌ Unexpected error in /generate_prompt: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during prompt generation: {e}"}), 500


# --- Helper: Verify Supabase Token (Unchanged) ---
def get_user_id_from_token(auth_header):
    """Validates Supabase JWT and returns user ID or raises Exception."""
    if not auth_header or not auth_header.startswith("Bearer "):
        logging.warning("⛔ Auth header missing or invalid format.")
        raise Exception("Unauthorized: Missing or invalid token.")

    token = auth_header.split(" ")[1]
    try:
        # Use the Supabase client's built-in method (requires supabase-py >= 2.0)
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user or not user.id:
             logging.warning(f"⛔ Token validation failed or user ID missing. Response status: {getattr(user_response, 'status_code', 'N/A')}")
             raise Exception("Invalid token or user not found.")

        user_id = str(user.id) # Ensure it's a string if needed later
        logging.info(f"✅ Token validated successfully for user ID: {user_id}")
        return user_id

    except Exception as e:
        logging.error(f"❌ Error during token validation: {e.__class__.__name__}: {e}", exc_info=True)
        error_str = str(e).lower()
        if "invalid token" in error_str or "jwt" in error_str or "token is invalid" in error_str:
              raise Exception("Invalid token.") from e
        else:
              raise Exception("Token verification failed.") from e


# --- API Route: Get User Profile/Credits (Unchanged) ---
@app.route("/get_profile", methods=["GET"])
def get_profile():
    """Fetches the user's current credit balance."""
    logging.info("📥 /get_profile called")
    auth_header = request.headers.get("Authorization")
    user_id = None
    try:
        user_id = get_user_id_from_token(auth_header)
        logging.info(f"Fetching profile for user: {user_id}")

        # Fetch credits from the 'profiles' table
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
        if "Unauthorized" in error_message or "Invalid token" in error_message or "Token verification failed" in error_message:
              status_code = 401; error_message = "Authentication failed. Please log in again."
        elif "Database error" in error_message:
              status_code = 503; error_message = "Failed to retrieve profile data due to a database issue."
        else: error_message = "An unexpected error occurred while fetching your profile."
        return jsonify({"error": error_message}), status_code


# --- API Route: Generate Image (Unchanged from previous version with prefix) ---
@app.route("/generate", methods=["POST"])
def generate():
    """Generates 2 thumbnail images using Flux, processes (preview), saves, returns results."""
    logging.info("📥 /generate called (using Flux, requesting 2 images)")
    auth_header = request.headers.get("Authorization"); user_id = None; current_credits = 0
    NUM_IMAGES_TO_GENERATE = 2 # Still generate 2 images
    TOTAL_CREDIT_COST = CREDIT_COST_PER_IMAGE * NUM_IMAGES_TO_GENERATE # Calculate total cost

    # 1. --- Authentication and Credit Check (Unchanged) ---
    try:
        user_id = get_user_id_from_token(auth_header)
        profile_response = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()

        if hasattr(profile_response, 'error') and profile_response.error:
            raise Exception(f"DB error checking credits: {profile_response.error.message}")
        if not profile_response or not profile_response.data:
            logging.error(f"❌ Profile not found for user {user_id} during generation.")
            return jsonify({"error": "User profile not found."}), 404 # Not Found

        current_credits = profile_response.data.get("credits", 0)
        logging.info(f"💰 User {user_id} attempting generation with {current_credits} credits (needs {TOTAL_CREDIT_COST}).")

        if current_credits < TOTAL_CREDIT_COST:
            logging.warning(f"🚫 Insufficient credits for user {user_id} ({current_credits}/{TOTAL_CREDIT_COST}).")
            return jsonify({"error": f"Insufficient credits. This action requires {TOTAL_CREDIT_COST} credits."}), 403 # Forbidden

    except Exception as e: # Handle Auth/DB errors before generation attempt
        user_id_str = f" for user {user_id}" if user_id else ""
        logging.error(f"❌ Auth/Profile fetch error in /generate{user_id_str}: {e}", exc_info=True)
        error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message or "Token verification failed" in error_message:
              status_code = 401; error_message = "Authentication failed. Please log in again."
        elif "DB error" in error_message:
              status_code = 503; error_message = "Failed to check credits due to a database issue."
        elif "Insufficient credits" in error_message:
              status_code = 403; error_message = error_message
        else: error_message = "An error occurred before generation could start."
        return jsonify({"error": error_message}), status_code

    # 2. --- Get Request Data ---
    data = request.json
    title = data.get("title", "") # Default to empty string if missing
    niche = data.get("niche", "") # Default to empty string if missing
    original_image_prompt = data.get("prompt") # Get the prompt from the frontend

    if not original_image_prompt:
        logging.warning(f"⚠️ Missing 'prompt' in request body for user {user_id}.")
        return jsonify({"error": "Image prompt is required."}), 400 # Bad Request

    logging.info(f" Rcvd Gen Request - User:{user_id}, Title:'{title}', Niche:'{niche}', Original Prompt:'{original_image_prompt}'")

    # --- Define the Thumbnail Prompt Prefix ---
    thumbnail_prefix = (
        "Create a highly clickable, viral-quality YouTube thumbnail. "
        "The image should be bold, colorful, and extremely eye-catching even at small sizes. "
        "Focus on clear, dramatic compositions with strong emotions, simple backgrounds, "
        "and exaggerated action or expressions. "
        "Prioritize visual storytelling that instantly grabs attention and makes viewers want to click. "
        "Scene details: "
    )

    # --- Combine the prefix with the original prompt ---
    combined_prompt = f"{thumbnail_prefix}[{original_image_prompt}]"
    logging.info(f"🔧 Combined prompt for Flux: '{combined_prompt}'")


    # --- 3. Generate images with Flux (Using Combined Prompt) ---
    generated_data_uris = []
    flux_errors = [] # Keep track of errors during generation loop

    logging.info(f"⏳ Requesting {NUM_IMAGES_TO_GENERATE} images from Flux for user {user_id}...")
    for i in range(NUM_IMAGES_TO_GENERATE):
        logging.info(f"  Generating image {i+1}/{NUM_IMAGES_TO_GENERATE}...")
        try:
            # Call the Flux helper function WITH THE COMBINED PROMPT
            img_url = generate_flux_image(combined_prompt) # <--- Use combined_prompt here

            # Fetch the image content from the URL
            logging.info(f"    Fetching image content from URL: {img_url}")
            img_response = requests.get(img_url, timeout=60) # Add timeout for fetch
            img_response.raise_for_status() # Check if fetch was successful
            img_bytes = img_response.content

            # Determine MIME type based on URL extension (simple check)
            mime = "jpeg" if img_url.lower().endswith(('.jpg', '.jpeg')) else "png"
            logging.info(f"    Determined MIME type: image/{mime}")

            # Encode the bytes as base64
            b64_encoded_data = base64.b64encode(img_bytes).decode('utf-8')

            # Create the Data URI
            data_uri = f"data:image/{mime};base64,{b64_encoded_data}"
            generated_data_uris.append(data_uri)
            logging.info(f"  ✅ Image {i+1} generated and converted to Data URI.")

        except Exception as e:
            # Log the error for this specific image generation attempt
            logging.error(f"❌ Flux or fetch failed on image {i+1}: {e}", exc_info=True)
            flux_errors.append(f"Image {i+1}: {e}")
            generated_data_uris.append(None) # Add None to keep list length consistent

    # Check if *any* images were generated successfully
    successful_count = sum(1 for uri in generated_data_uris if uri is not None)
    if successful_count == 0:
        logging.error(f"❌ Flux generation failed for all {NUM_IMAGES_TO_GENERATE} images for user {user_id}.")
        error_summary = "; ".join(flux_errors)
        # Return a 502 Bad Gateway as the upstream service (Flux) failed
        return jsonify({"error": f"Image generation failed for all attempts. Errors: {error_summary}"}), 502

    logging.info(f"✅ Successfully generated {successful_count}/{NUM_IMAGES_TO_GENERATE} images from Flux.")
    # --- End of Image Generation Block ---


    # 5. --- Process Each Generated Image (Preview & Crop - Crop Disabled) ---
    processed_results_urls = [] # Store final URLs for response
    db_save_tasks = []      # Store data dictionaries for DB insertion

    for index, original_uri in enumerate(generated_data_uris):
        # Skip processing if generation failed for this image
        if original_uri is None:
            logging.warning(f"⚠️ Skipping processing for image {index + 1} due to generation failure.")
            continue # Move to the next image

        logging.info(f"⚙️ Processing image {index + 1}/{NUM_IMAGES_TO_GENERATE} for user {user_id}...")
        preview_uri = None # Initialize here
        # cropped_uri = None # Crop is disabled

        # Start with the original URI as the final URI (since cropping is disabled)
        final_uri_for_db_and_response = original_uri

        try:
            # --- CROP DISABLED as per instructions ---
            logging.info(f"  Skipping crop for image {index + 1} (using Flux direct output).")

            # --- Generate preview from the final version (original Flux URI) ---
            preview_uri = generate_preview_data_uri(final_uri_for_db_and_response)
            if preview_uri:
                 logging.info(f"  ✅ Preview generated from FINAL URI for image {index + 1}.")
            else:
                 logging.warning(f"  ⚠️ Failed to generate preview from FINAL URI for image {index + 1}.")

        except Exception as processing_err:
            # Log error but continue processing other images if possible
            logging.error(f"❌ Error processing image {index + 1} for user {user_id}: {processing_err}", exc_info=True)
            # Ensure preview is None, final URL remains the original URI
            preview_uri = None
            final_uri_for_db_and_response = original_uri # Keep original even if preview failed

        # Add the final URL (original Flux URI) to the list for the response
        processed_results_urls.append(final_uri_for_db_and_response)

        # Prepare data for database insertion for this image
        # Store the ORIGINAL prompt from the user/GPT, not the combined one, for history clarity
        db_save_tasks.append({
            "user_id": user_id,
            "title": title,
            "niche": niche,
            "prompt": original_image_prompt, # Store the original prompt
            "image_url": final_uri_for_db_and_response, # Original Flux URI
            "preview_image_url": preview_uri # Preview URI (from final) or None
        })

    # 6. --- Deduct Credits (Unchanged) ---
    new_credits = max(0, current_credits - TOTAL_CREDIT_COST)
    try:
        logging.info(f"Attempting to deduct {TOTAL_CREDIT_COST} credits from user {user_id} (new balance: {new_credits}).")
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        if hasattr(update_response, 'error') and update_response.error:
             logging.error(f"⚠️ DB WARN: Failed to deduct {TOTAL_CREDIT_COST} credits for {user_id} after generation: {update_response.error}")
        elif not (hasattr(update_response, 'data') and update_response.data):
             logging.warning(f"⚠️ DB WARN: Credit deduction query returned no data for {user_id} (user might be deleted?).")
        else:
             logging.info(f"✅ Deducted {TOTAL_CREDIT_COST} credit(s) from {user_id}. New balance: {new_credits}")
    except Exception as credit_error:
        logging.error(f"❌ DB CRITICAL: Exception deducting credit for {user_id}: {credit_error}", exc_info=True)

    # 7. --- Save All Processed Images to History (Unchanged) ---
    saved_count = 0
    if db_save_tasks: # Only attempt insert if there are valid tasks
        try:
            logging.info(f"Attempting to save {len(db_save_tasks)} thumbnail records to history for user {user_id}...")
            insert_response = supabase.table("thumbnail").insert(db_save_tasks).execute()

            if hasattr(insert_response, 'error') and insert_response.error:
                 logging.error(f"⚠️ DB WARN: Failed to save batch history for {user_id}: {insert_response.error}")
            elif not (hasattr(insert_response, 'data') and insert_response.data):
                 logging.warning(f"⚠️ DB WARN: Batch history insert query returned no data for {user_id}.")
            else:
                 saved_count = len(insert_response.data)
                 logging.info(f"✅ Saved {saved_count}/{len(db_save_tasks)} generated thumbnails to history for {user_id}")
                 if saved_count < len(db_save_tasks):
                      logging.warning(f"DB WARN: Mismatch in expected vs saved history items for {user_id}.")

        except Exception as history_error:
            logging.error(f"❌ DB CRITICAL: Exception saving batch history for {user_id}: {history_error}", exc_info=True)
    else:
        logging.warning(f"No valid images to save to history for user {user_id}.")

    # 8. --- Return Success Response ---
    if not processed_results_urls:
        logging.error(f"❌ No valid image URLs available to return for user {user_id} after processing.")
        return jsonify({"error": "Image processing failed to produce any usable results."}), 500

    logging.info(f"✅ Returning {len(processed_results_urls)} processed image URLs to user {user_id}.")
    return jsonify({
        "prompt": original_image_prompt, # Return ORIGINAL prompt for context in UI
        "image_urls": processed_results_urls, # Return LIST of final (Flux) URLs
        "new_credits": new_credits # Return updated credit balance
    })


# --- API Route: Get History (Unchanged) ---
@app.route("/history", methods=["GET"])
def get_history():
    """Fetches paginated thumbnail generation history for the user."""
    logging.info("📥 /history called")
    auth_header = request.headers.get("Authorization"); user_id = None
    try:
        user_id = get_user_id_from_token(auth_header)
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int) # Default limit to 10
        if page < 1: page = 1
        if limit < 1: limit = 10 # Ensure limit is positive
        if limit > 50: limit = 50 # Optional: Set a max limit
        offset = (page - 1) * limit
        logging.info(f"📦 Fetching history page {page} (limit {limit}, offset {offset}) for user {user_id}...")

        # Fetch history including the preview_image_url, with range and count
        response = supabase.table("thumbnail").select(
            "id, created_at, title, niche, prompt, image_url, preview_image_url", # Select specific columns
            count='exact' # Get total count for pagination
        ).eq("user_id", user_id).order(
            "created_at", desc=True # Order by most recent
        ).range(
            offset, offset + limit - 1 # Apply pagination range
        ).execute()

        if hasattr(response, 'error') and response.error:
            logging.error(f"❌ Supabase DB error fetching history for {user_id}: {response.error}")
            raise Exception(f"DB error fetching history: {response.error.message or 'Unknown DB Error'}")

        history_data = response.data if hasattr(response, 'data') else []
        total_count = response.count if hasattr(response, 'count') else 0

        logging.info(f"✅ History fetched page {page} for {user_id}: {len(history_data)} items (Total: {total_count})")

        # Return the data, total count, and pagination info
        return jsonify({
            "items": history_data, # Each item contains image_url and preview_image_url
            "total": total_count,
            "page": page,
            "limit": limit
        })

    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""
        logging.error(f"❌ Error in /history{user_id_str}: {e}", exc_info=True)
        error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message or "Token verification failed" in error_message:
            status_code = 401; error_message = "Authentication failed. Please log in again."
        elif "DB error" in error_message:
            status_code = 503; error_message = "Failed to retrieve history data due to a database issue."
        else: error_message = "An unexpected error occurred while fetching your history."
        return jsonify({"error": error_message}), status_code


# --- API Route: Create Stripe Checkout Session (Unchanged) ---
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    """Creates a Stripe Checkout session for purchasing credits."""
    logging.info("📥 /create-checkout-session called")
    auth_header = request.headers.get("Authorization"); data = request.json; user_id = None
    price_id = data.get('priceId')

    if not price_id:
        logging.warning("❌ Price ID missing in request.")
        return jsonify({"error": "Price ID is required."}), 400

    # Get frontend domain from environment variable for redirect URLs
    YOUR_FRONTEND_DOMAIN = os.environ.get("FRONTEND_DOMAIN", "http://127.0.0.1:5500") # Default to local dev
    if YOUR_FRONTEND_DOMAIN == "http://127.0.0.1:5500":
         logging.warning("⚠️ FRONTEND_DOMAIN is using default localhost. Ensure this is correct for deployment.")
    logging.info(f"Using frontend domain for Stripe redirects: {YOUR_FRONTEND_DOMAIN}")

    try:
        user_id = get_user_id_from_token(auth_header)
        logging.info(f"✅ User {user_id} initiating checkout for price ID: {price_id}")

        # Validate the price ID against configured mapping
        if price_id not in PRICE_ID_TO_CREDITS:
            logging.error(f"❌ Unrecognized Stripe Price ID requested: {price_id}")
            return jsonify({"error": "Invalid purchase option selected."}), 400

        # Define Stripe success and cancel URLs
        success_url=f"{YOUR_FRONTEND_DOMAIN}?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{YOUR_FRONTEND_DOMAIN}?payment=cancel"

        logging.info(f"Creating Stripe Checkout session for price: {price_id}")
        checkout_session = stripe.checkout.Session.create(
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='payment',
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={ 'supabase_user_id': user_id, 'price_id': price_id }
        )

        logging.info(f"✅ Stripe session created: {checkout_session.id} for user {user_id}")
        return jsonify({'url': checkout_session.url})

    except stripe.error.StripeError as e:
        logging.error(f"❌ Stripe Error creating session for {user_id or 'Unknown User'}: {e}", exc_info=True)
        return jsonify({"error": f"Stripe error: {getattr(e, 'user_message', 'Payment processing error.')}"}), 500
    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""
        logging.error(f"❌ Error creating checkout session{user_id_str}: {e}", exc_info=True)
        error_message = str(e); status_code = 500
        if "Unauthorized" in error_message or "Invalid token" in error_message or "Token verification failed" in error_message:
              status_code = 401; error_message = "Authentication failed. Please log in again."
        else: error_message = "Failed to create checkout session due to an unexpected error."
        return jsonify({"error": error_message}), status_code


# --- API Route: Stripe Webhook Handler (Unchanged) ---
@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    """Handles incoming Stripe webhook events, specifically checkout.session.completed."""
    logging.info("🔔 /stripe-webhook received event")
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    event = None

    if not STRIPE_WEBHOOK_SECRET:
        logging.error("❌ Stripe Webhook Secret is not configured.")
        return jsonify(error="Webhook configuration error on server"), 500

    try:
        event = stripe.Webhook.construct_event( payload, sig_header, STRIPE_WEBHOOK_SECRET )
    except ValueError as e: # Invalid payload
        logging.error(f"❌ Invalid webhook payload: {e}")
        return jsonify(error="Invalid payload"), 400
    except stripe.error.SignatureVerificationError as e: # Invalid signature
        logging.error(f"❌ Invalid webhook signature: {e}")
        return jsonify(error="Invalid signature"), 400
    except Exception as e:
        logging.error(f"❌ Error constructing webhook event: {e}", exc_info=True)
        return jsonify(error="Webhook processing error"), 500

    logging.info(f"✅ Webhook signature verified. Event ID: {event.get('id', 'N/A')}, Type: {event.get('type')}")

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        session_id = session.get('id')
        logging.info(f"Processing 'checkout.session.completed' event for session: {session_id}")

        if session.get('payment_status') == 'paid':
            metadata = session.get('metadata')
            if not metadata:
                logging.error(f"❌ Metadata missing in successful payment event: {session_id}")
                return jsonify(status="Metadata missing"), 200

            user_id = metadata.get('supabase_user_id')
            price_id = metadata.get('price_id')

            if not user_id or not price_id:
                logging.error(f"❌ Missing supabase_user_id or price_id in metadata for session: {session_id}")
                return jsonify(status="Required metadata fields missing"), 200

            credits_to_add = PRICE_ID_TO_CREDITS.get(price_id)
            if credits_to_add is None:
                logging.error(f"⚠️ Unrecognized price_id '{price_id}' found in webhook metadata for session: {session_id}")
                return jsonify(status="Unrecognized price ID"), 200

            logging.info(f"Attempting to grant {credits_to_add} credits to user {user_id} for price {price_id} (Session: {session_id})")

            try:
                profile_res = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()
                current_credits = 0; profile_exists = False
                if hasattr(profile_res, 'error') and profile_res.error:
                     logging.error(f"❌ DB Error fetching profile for credit update (user {user_id}, session {session_id}): {profile_res.error}")
                elif profile_res and profile_res.data:
                     current_credits = profile_res.data.get("credits", 0); profile_exists = True
                     logging.info(f"   Profile exists for {user_id}. Current credits: {current_credits}")
                else:
                     logging.warning(f"⚠️ Profile not found for {user_id} during webhook credit update (Session: {session_id}). Will create/update starting from 0 credits.")

                new_total_credits = current_credits + credits_to_add
                logging.info(f"   Updating credits for {user_id}: {current_credits} -> {new_total_credits}")

                update_res = supabase.table("profiles").upsert({ "id": user_id, "credits": new_total_credits }).execute()

                if hasattr(update_res, 'error') and update_res.error:
                    logging.error(f"❌ DB Error updating/upserting credits for {user_id} (Session: {session_id}): {update_res.error}")
                    return jsonify(error="Database update failed during credit grant"), 500
                elif not (hasattr(update_res, 'data') and update_res.data):
                    logging.error(f"❌ DB Error: Upsert credit query returned no data for {user_id} (Session: {session_id}).")
                    return jsonify(error="Database update failed (no data returned)"), 500
                else:
                    logging.info(f"✅ Credits successfully updated for {user_id} (Session: {session_id}). New balance: {new_total_credits}")

            except Exception as db_error:
                logging.error(f"❌ DB Exception during credit update for {user_id} (Session: {session_id}): {db_error}", exc_info=True)
                return jsonify(error="Database exception during credit grant"), 500
        else:
            logging.info(f"Checkout session {session_id} completed but payment status was '{session.get('payment_status')}'. No credits granted.")
    else:
        logging.info(f"🤷‍♀️ Received unhandled Stripe event type: {event.get('type')}")

    return jsonify(success=True), 200


# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) # Default port 5001
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0") # Default host
    debug_mode_str = os.environ.get("FLASK_DEBUG", "False")
    debug_mode = debug_mode_str.lower() in ["true", "1", "t", "yes"]

    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    app.logger.setLevel(log_level)

    logging.info(f"Effective root logging level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    logging.info(f"Flask app logger level set to: {logging.getLevelName(app.logger.getEffectiveLevel())}")

    if debug_mode:
        logging.warning("⚠️ Flask is running in DEBUG mode! Ensure this is False in production.")
    else:
        logging.info("Flask is running in PRODUCTION mode (Debug=False).")

    logging.info(f"🚀 Starting Flask server on http://{host}:{port} (Debug: {debug_mode})")
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
