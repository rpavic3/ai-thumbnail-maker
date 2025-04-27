# Required Imports
from supabase import create_client, Client
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI, APIError # v1.x+ library
import json
import requests
import os
from dotenv import load_dotenv
import stripe
import logging
import base64 # Needed for decoding/encoding
import io     # Needed for handling image bytes
from PIL import Image, UnidentifiedImageError # <<< ENSURE Pillow IS IMPORTED & UnidentifiedImageError

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

# --- OpenAI Configuration (v1.x+) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY: logging.warning("⚠️ OpenAI API Key environment variable missing or empty.")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("✅ OpenAI client initialized (v1.x+).")
except APIError as e: logging.error(f"❌ OpenAI API Error client init: {e}"); raise ValueError("OpenAI client failed init.") from e
except Exception as e: logging.error(f"❌ Failed OpenAI client init: {e}"); raise ValueError("OpenAI client failed init.") from e

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

price_id_pack1 = os.environ.get(price_id_env_var_name)
credits_pack1_str = os.environ.get(credits_env_var_name, default_credits)

# --- ADD DEBUG LINE 1 ---
# --- END DEBUG LINE 1 ---

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

# --- ADD DEBUG LINE 2 ---
# --- END DEBUG LINE 2 ---


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
TOP_BOTTOM_CROP_PIXELS = 62 # Pixels to crop from top and bottom
EXPECTED_WIDTH = 1536 # Expected width from OpenAI
EXPECTED_HEIGHT = 1024 # Expected height from OpenAI
CREDIT_COST_PER_IMAGE = 1 # Cost for EACH image generated


# --- Helper Function: Generate Preview Image Data URI ---
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

        image_format = header.split('/')[1].split(';')[0]
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
        # --- MODIFICATION: Lower JPEG quality for smaller preview file size ---
        preview_image.save(buffer, format='JPEG', quality=60) # Lowered quality from 80 to 75
        # --- END MODIFICATION ---
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

# --- Helper Function: Crop Image Data URI ---
def crop_image_data_uri(data_uri: str) -> str | None:
    """
    Decodes a base64 Data URI, crops TOP_BOTTOM_CROP_PIXELS from top and bottom,
    and re-encodes it as a base64 Data URI. Preserves original format if possible.
    Returns None if cropping fails.
    """
    logging.info("✂️ Attempting to crop image...")
    try:
        # Basic check for Data URI format
        if not data_uri or not data_uri.startswith("data:image/") or ";base64," not in data_uri:
            logging.warning(f"⚠️ Invalid or missing Data URI for cropping. Cannot crop.")
            return None

        header, encoded_data = data_uri.split(',', 1)
        image_format_from_header = header.split('/')[1].split(';')[0].lower() # Get format from header

        image_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(image_data))

        # Determine the format to save in (prefer original detected format)
        original_image_format = image.format if image.format else image_format_from_header.upper()
        if not original_image_format:
             logging.warning("⚠️ Could not determine original image format for saving cropped image. Defaulting to PNG.")
             original_image_format = "PNG" # Safe default

        width, height = image.size
        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            logging.warning(f"⚠️ Image dimensions ({width}x{height}) differ from expected ({EXPECTED_WIDTH}x{EXPECTED_HEIGHT}) for cropping. Cropping may be inaccurate or fail.")
            # Decide if you want to return None or proceed anyway. Proceeding for now.

        # Calculate crop box
        left = 0
        upper = TOP_BOTTOM_CROP_PIXELS
        right = width
        lower = height - TOP_BOTTOM_CROP_PIXELS

        # Validate crop box dimensions before cropping
        if upper < 0 or lower > height or left < 0 or right > width or upper >= lower or left >= right:
            logging.error(f"❌ Invalid crop dimensions calculated: ({left}, {upper}, {right}, {lower}) from image size ({width}x{height}). Cannot crop.")
            return None # Return None if crop box is invalid

        crop_box = (left, upper, right, lower)
        logging.info(f"Calculated crop box: {crop_box}")

        cropped_image = image.crop(crop_box)
        logging.info(f"✅ Image cropped successfully to {cropped_image.size[0]}x{cropped_image.size[1]}.")

        # Save the cropped image back to a buffer
        buffer = io.BytesIO()
        save_params = {'format': original_image_format}
        # Add quality param only if it's relevant (like JPEG), otherwise Pillow handles defaults
        if original_image_format.upper() == 'JPEG':
            save_params['quality'] = 95 # Keep high quality for main image if JPEG

        cropped_image.save(buffer, **save_params)
        buffer.seek(0)

        # Re-encode the cropped image
        cropped_encoded_data = base64.b64encode(buffer.read()).decode('utf-8')
        # Use the determined original format in the new Data URI header
        cropped_data_uri = f"data:image/{original_image_format.lower()};base64,{cropped_encoded_data}"
        logging.info(f"✅ Cropped image re-encoded to Data URI (Format: {original_image_format}).")
        return cropped_data_uri

    except base64.binascii.Error as b64_err:
         logging.error(f"❌ Base64 decoding error during cropping: {b64_err}", exc_info=True)
         return None
    except UnidentifiedImageError: # Catch Pillow specific error for bad image data
         logging.error("❌ Pillow UnidentifiedImageError: Could not open image data for cropping.", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"❌ Unexpected error during image cropping: {e}", exc_info=True)
        return None # Return None if cropping fails
    

@app.route("/", methods=["GET", "HEAD"])
def health_root():
    return "ok", 200



# --- API Route: Generate Prompt ---
@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    """Generates a DALL-E style prompt suggestion based on title and niche."""
    logging.info("📥 /generate_prompt called")
    data = request.json
    title = data.get("title")
    niche = data.get("niche")

    if not title or not niche:
        logging.warning("⚠️ /generate_prompt: Missing title or niche.")
        return jsonify({"error": "Title and Niche are required."}), 400

    logging.info(f"🧠 Prompt generation request - Title: '{title}', Niche: '{niche}'")

    if not client or not client.api_key:
        logging.error("❌ OpenAI client not configured for /generate_prompt.")
        return jsonify({"error": "AI Service is not configured."}), 503 # Service Unavailable

    try:
        # Using gpt-4o for prompt generation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7, # Balance creativity and coherence
            messages=[
                 {"role": "system", "content": "You are a viral YouTube thumbnail designer who generates visual ideas and images."},
                 {"role": "user", "content": f"""Generate ONE compelling, eyecatching visual prompt idea for a YouTube thumbnail based on this video information. The prompt should be suitable for an AI image generator like 4o image generation. Focus on creating a visually engaging scene.

Video Title: {title}
Channel Niche: {niche}

Return ONLY the 4o image generation style visual prompt itself, without any extra text, formatting, or explanations."""}
             ]
        )

        if not response or not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            logging.error(f"❌ GPT prompt generation returned unexpected response structure: {response}")
            raise Exception("AI failed to generate a valid prompt structure.")

        prompt = response.choices[0].message.content.strip()
        if not prompt:
            logging.warning("⚠️ GPT returned an empty prompt.")
            raise Exception("AI failed to generate a non-empty prompt.")

        logging.info(f"✅ Prompt generated successfully: '{prompt}'")
        return jsonify({"prompt": prompt})

    except APIError as e:
        logging.error(f"❌ OpenAI API Error during prompt generation: {e}", exc_info=True)
        return jsonify({"error": f"AI prompt generation failed ({e.status_code}): {e.message or 'Unknown API Error'}"}), 502 # Bad Gateway
    except Exception as e:
        logging.error(f"❌ Unexpected error in /generate_prompt: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during prompt generation: {e}"}), 500


# --- Helper: Verify Supabase Token ---
def get_user_id_from_token(auth_header):
    """Validates Supabase JWT and returns user ID or raises Exception."""
    if not auth_header or not auth_header.startswith("Bearer "):
        logging.warning("⛔ Auth header missing or invalid format.")
        raise Exception("Unauthorized: Missing or invalid token.")

    token = auth_header.split(" ")[1]
    try:
        # Use the Supabase client's built-in method (requires supabase-py >= 2.0)
        # Note: This uses the SERVICE_ROLE_KEY implicitly if the client was initialized with it.
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user or not user.id:
             # Log the actual response if available and doesn't contain sensitive info
             logging.warning(f"⛔ Token validation failed or user ID missing. Response status: {getattr(user_response, 'status_code', 'N/A')}")
             raise Exception("Invalid token or user not found.")

        user_id = str(user.id) # Ensure it's a string if needed later
        logging.info(f"✅ Token validated successfully for user ID: {user_id}")
        return user_id

    except Exception as e:
        # Catch potential API errors from supabase.auth.get_user or other issues
        logging.error(f"❌ Error during token validation: {e.__class__.__name__}: {e}", exc_info=True)
        # Check for specific error types if supabase-py provides them, otherwise generalize
        # Convert error to string for checking content
        error_str = str(e).lower()
        if "invalid token" in error_str or "jwt" in error_str or "token is invalid" in error_str:
             raise Exception("Invalid token.") from e
        else:
             # Generalize for other potential issues (network, Supabase internal)
             raise Exception("Token verification failed.") from e


# --- API Route: Get User Profile/Credits ---
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

        # Handle potential Supabase errors
        if hasattr(profile_response, 'error') and profile_response.error:
            logging.error(f"❌ Supabase DB error fetching profile for {user_id}: {profile_response.error}")
            raise Exception(f"Database error fetching profile: {profile_response.error.message}")

        # Check if profile exists and return credits or 0
        if profile_response and profile_response.data:
            credits = profile_response.data.get("credits", 0)
            logging.info(f"✅ Fetched credits for {user_id}: {credits}")
            return jsonify({"credits": credits})
        else:
            # Profile might not exist yet (e.g., new signup) - return 0 credits
            logging.warning(f"⚠️ Profile not found for {user_id}. Returning 0 credits.")
            return jsonify({"credits": 0})

    except Exception as e:
        user_id_str = f" for user {user_id}" if user_id else ""
        logging.error(f"❌ Error in /get_profile{user_id_str}: {e}", exc_info=True)
        error_message = str(e)
        status_code = 500
        # Determine status code based on error type
        if "Unauthorized" in error_message or "Invalid token" in error_message or "Token verification failed" in error_message:
             status_code = 401
             error_message = "Authentication failed. Please log in again."
        elif "Database error" in error_message:
             status_code = 503 # Service Unavailable (DB issue)
             error_message = "Failed to retrieve profile data due to a database issue."
        else:
             error_message = "An unexpected error occurred while fetching your profile."
        return jsonify({"error": error_message}), status_code


# --- API Route: Generate Image (MODIFIED for 2 Images) ---
@app.route("/generate", methods=["POST"])
def generate():
    """Generates 2 thumbnail images based on the prompt, crops, saves, returns results."""
    logging.info("📥 /generate called (requesting 2 images)")
    auth_header = request.headers.get("Authorization"); user_id = None; current_credits = 0
    NUM_IMAGES_TO_GENERATE = 2 # Define how many images we want
    TOTAL_CREDIT_COST = CREDIT_COST_PER_IMAGE * NUM_IMAGES_TO_GENERATE # Calculate total cost

    # 1. --- Authentication and Credit Check ---
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

        # Check if user has enough credits
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
        elif "Insufficient credits" in error_message: # Should be caught above, but as fallback
             status_code = 403; error_message = error_message # Pass specific message
        else:
             error_message = "An error occurred before generation could start."
        return jsonify({"error": error_message}), status_code

    # 2. --- Get Request Data ---
    data = request.json
    title = data.get("title", "") # Default to empty string if missing
    niche = data.get("niche", "") # Default to empty string if missing
    image_prompt = data.get("prompt")

    if not image_prompt:
        logging.warning(f"⚠️ Missing 'prompt' in request body for user {user_id}.")
        return jsonify({"error": "Image prompt is required."}), 400 # Bad Request

    logging.info(f" Rcvd Gen Request - User:{user_id}, Title:'{title}', Niche:'{niche}', Original Prompt:'{image_prompt}'")

    # 3. --- Prepare Prompt for AI (with aspect ratio instructions) ---
    # --- DO NOT CHANGE THIS PROMPT (User Request) ---
    image_prompt_with_bars = f"A digital image featuring [{image_prompt}], ensuring there are solid black bars exactly 62 pixels tall added to the very top and very bottom edges of the entire image frame. PLEASE, THIS IS VERY IMPORTANT!! THE BLACK BARS ARE AT THE TOP, AND AT THE BOTTOM, NEVER ON THE SIDE. The main visual content based on the prompt should be clearly visible and composed naturally between these two black bars."
    logging.info(f"🎨 Using modified prompt for aspect ratio: '{image_prompt_with_bars}'")

    # 4. --- Generate Images from OpenAI ---
    generated_data_uris = [] # To store raw base64 results from OpenAI
    try:
        logging.info(f"⏳ Requesting {NUM_IMAGES_TO_GENERATE} images from OpenAI for user {user_id}...")
        response = client.images.generate(
            model="gpt-image-1",
            prompt=image_prompt_with_bars,
            n=NUM_IMAGES_TO_GENERATE, # Request the desired number
            size="1536x1024",
            quality="high" # Use high quality
            # REMOVED response_format="b64_json" - library handles this
        )

        # Validate response structure and content
        if not response or not response.data or len(response.data) < NUM_IMAGES_TO_GENERATE:
             actual_received = len(response.data) if response and response.data else 0
             logging.error(f"❌ OpenAI did not return the expected number of images ({actual_received}/{NUM_IMAGES_TO_GENERATE}). Response: {response}")
             raise Exception(f"AI failed to generate the requested number of images ({NUM_IMAGES_TO_GENERATE}).")

        # Extract the base64 data for each generated image
        for i, item in enumerate(response.data):
             # The library should provide b64_json when n > 1 or size is large
             if item and item.b64_json:
                 b64_string = item.b64_json
                 # Assume PNG format from OpenAI when b64_json is returned
                 uri = f"data:image/png;base64,{b64_string}"
                 generated_data_uris.append(uri)
                 logging.info(f"✅ Received image {i+1}/{NUM_IMAGES_TO_GENERATE} (base64) from OpenAI.")
             else:
                 logging.warning(f"⚠️ Missing b64_json for image {i+1} in OpenAI response.")
                 generated_data_uris.append(None) # Append None to indicate failure for this specific image

        # Check if we got all images successfully
        successful_count = sum(1 for uri in generated_data_uris if uri is not None)
        if successful_count < NUM_IMAGES_TO_GENERATE:
             logging.error(f"❌ Failed to retrieve all ({NUM_IMAGES_TO_GENERATE}) image data URIs from OpenAI. Got {successful_count}.")
             # Decide on behavior: raise error, or proceed with successful ones?
             # For now, raise an error if *any* failed, as user expects N images.
             raise Exception(f"AI generation incomplete. Received {successful_count}/{NUM_IMAGES_TO_GENERATE} images.")

        logging.info(f"✅ Successfully received {len(generated_data_uris)} image URIs from OpenAI.")

    except APIError as e: # Handle specific OpenAI API errors
        logging.error(f"❌ OpenAI API Error during multi-image generation for user {user_id}: {e}", exc_info=True)
        user_error_message = f"OpenAI image generation failed ({e.status_code}): {e.message or e.code or e.type or 'Unknown OpenAI Error'}"
        # Add specific error checks if needed (billing, rate limit, content policy etc.)
        if e.code == 'billing_not_active': user_error_message = "Image generation failed: OpenAI account billing issue."
        if e.code == 'rate_limit_exceeded': user_error_message = "Image generation failed: Rate limit exceeded. Please try again later."
        if 'content_policy_violation' in str(e.message or '').lower(): user_error_message = "Image generation failed due to content policy. Please modify your prompt."
        return jsonify({"error": user_error_message}), 502 # Bad Gateway for upstream errors
    except Exception as e: # Handle other unexpected errors during generation
        logging.error(f"❌ Unexpected Error during OpenAI multi-image generation for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": f"Image generation failed unexpectedly: {e}"}), 500

    # 5. --- Process Each Generated Image (Preview & Crop) ---
    processed_results_urls = [] # Store final URLs for response
    db_save_tasks = []      # Store data dictionaries for DB insertion

    for index, original_uri in enumerate(generated_data_uris):
        logging.info(f"⚙️ Processing image {index + 1}/{NUM_IMAGES_TO_GENERATE} for user {user_id}...")
        preview_uri = None
        cropped_uri = None
        # Start with the original URI as the fallback
        final_uri_for_db_and_response = original_uri

        if original_uri and original_uri.startswith("data:image/"):
            try:
                # Attempt to generate preview
                preview_uri = generate_preview_data_uri(original_uri)
                if preview_uri: logging.info(f"✅ Preview generated for image {index + 1}.")
                else: logging.warning(f"⚠️ Failed to generate preview for image {index + 1}.")

                # Attempt to crop image
                cropped_uri = crop_image_data_uri(original_uri)
                if cropped_uri:
                    logging.info(f"✅ Image {index + 1} cropped.")
                    final_uri_for_db_and_response = cropped_uri # Use cropped version if successful
                else:
                    logging.warning(f"⚠️ Failed to crop image {index + 1}. Using original (with bars).")
                    # final_uri_for_db_and_response remains original_uri

            except Exception as processing_err:
                # Log error but continue processing other images if possible
                logging.error(f"❌ Error processing image {index + 1} for user {user_id}: {processing_err}", exc_info=True)
                # Ensure preview is None and final URL is the original (potentially broken) one
                preview_uri = None
                final_uri_for_db_and_response = original_uri
        else:
            # This case should ideally not happen if OpenAI call succeeded, but handle defensively
            logging.warning(f"⚠️ Skipping processing for image {index + 1} due to invalid/missing original URI.")
            final_uri_for_db_and_response = None # Mark as None if original was invalid
            preview_uri = None

        # Add the final URL (cropped or original) to the list for the response
        if final_uri_for_db_and_response:
            processed_results_urls.append(final_uri_for_db_and_response)

        # Prepare data for database insertion for this image (even if processing failed, save original)
        # Only add if we have *some* image url (original or cropped)
        if final_uri_for_db_and_response:
            db_save_tasks.append({
                "user_id": user_id,
                "title": title,
                "niche": niche,
                "prompt": image_prompt, # Store the original user prompt for context
                "image_url": final_uri_for_db_and_response, # Cropped or original URI
                "preview_image_url": preview_uri # Preview URI or None
            })
        else:
             logging.error(f"❌ Image {index+1} resulted in no usable URL, skipping DB save for this item.")


    # 6. --- Deduct Credits (Only if generation was initially successful) ---
    # Credits are deducted *after* generation attempt, regardless of processing success,
    # as the OpenAI call was made and charged.
    new_credits = max(0, current_credits - TOTAL_CREDIT_COST)
    try:
        logging.info(f"Attempting to deduct {TOTAL_CREDIT_COST} credits from user {user_id} (new balance: {new_credits}).")
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        # Check for errors during update
        if hasattr(update_response, 'error') and update_response.error:
             # Log error but don't fail the request - user got the images (or tried)
             logging.error(f"⚠️ DB WARN: Failed to deduct {TOTAL_CREDIT_COST} credits for {user_id} after generation: {update_response.error}")
        elif not (hasattr(update_response, 'data') and update_response.data):
             logging.warning(f"⚠️ DB WARN: Credit deduction query returned no data for {user_id} (user might be deleted?).")
        else:
             logging.info(f"✅ Deducted {TOTAL_CREDIT_COST} credit(s) from {user_id}. New balance: {new_credits}")
    except Exception as credit_error:
        # Log critical error if the update query itself fails
        logging.error(f"❌ DB CRITICAL: Exception deducting credit for {user_id}: {credit_error}", exc_info=True)
        # Continue to return images if they were generated


    # 7. --- Save All Processed Images to History ---
    saved_count = 0
    if db_save_tasks: # Only attempt insert if there are valid tasks
        try:
            logging.info(f"Attempting to save {len(db_save_tasks)} thumbnail records to history for user {user_id}...")
            # Perform batch insert
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
            # Images were generated, credits deducted, but history failed. Still return images.
    else:
        logging.warning(f"No valid images to save to history for user {user_id}.")

    # 8. --- Return Success Response ---
    # Ensure we only return valid URLs
    if not processed_results_urls:
        logging.error(f"❌ No valid image URLs available to return for user {user_id} after processing.")
        # Return error even if credits were deducted and history attempted, as user gets nothing.
        return jsonify({"error": "Image processing failed to produce any usable results."}), 500

    logging.info(f"✅ Returning {len(processed_results_urls)} processed image URLs to user {user_id}.")
    return jsonify({
        "prompt": image_prompt, # Return original prompt for context
        "image_urls": processed_results_urls, # Return LIST of final (cropped/fallback) URLs
        "new_credits": new_credits # Return updated credit balance
    })


# --- API Route: Get History ---
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


# --- API Route: Create Stripe Checkout Session ---
@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    """Creates a Stripe Checkout session for purchasing credits."""
    logging.info("📥 /create-checkout-session called")
    auth_header = request.headers.get("Authorization"); data = request.json; user_id = None
    price_id = data.get('priceId')

    # --- ADD DEBUG LINE 3 ---
    # --- END DEBUG LINE 3 ---

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

        # --- ADD DEBUG LINE 4 ---
        # --- END DEBUG LINE 4 ---

        # Validate the price ID against configured mapping
        if price_id not in PRICE_ID_TO_CREDITS: # <--- THIS CHECK IS FAILING
            logging.error(f"❌ Unrecognized Stripe Price ID requested: {price_id}")
            # This is the error message being returned:
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
            # Include user ID and price ID in metadata for webhook processing
            metadata={
                'supabase_user_id': user_id,
                'price_id': price_id
            }
            # Consider adding customer_email: user.email if you fetch user email
        )

        logging.info(f"✅ Stripe session created: {checkout_session.id} for user {user_id}")
        # Return the Stripe Checkout session URL to the frontend for redirection
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


# --- API Route: Stripe Webhook Handler ---
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

    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
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

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        session_id = session.get('id')
        logging.info(f"Processing 'checkout.session.completed' event for session: {session_id}")

        # Check if the payment was successful
        if session.get('payment_status') == 'paid':
            metadata = session.get('metadata')
            if not metadata:
                logging.error(f"❌ Metadata missing in successful payment event: {session_id}")
                return jsonify(status="Metadata missing"), 200 # Acknowledge receipt, but log error

            user_id = metadata.get('supabase_user_id')
            price_id = metadata.get('price_id')

            if not user_id or not price_id:
                logging.error(f"❌ Missing supabase_user_id or price_id in metadata for session: {session_id}")
                return jsonify(status="Required metadata fields missing"), 200 # Acknowledge receipt

            # Get credits to add based on the price ID from metadata
            credits_to_add = PRICE_ID_TO_CREDITS.get(price_id)
            if credits_to_add is None:
                logging.error(f"⚠️ Unrecognized price_id '{price_id}' found in webhook metadata for session: {session_id}")
                return jsonify(status="Unrecognized price ID"), 200 # Acknowledge receipt

            logging.info(f"Attempting to grant {credits_to_add} credits to user {user_id} for price {price_id} (Session: {session_id})")

            # --- Update User Credits in Supabase ---
            # This is a critical section. Using RPC might be safer for atomicity at scale.
            try:
                # 1. Fetch current credits
                profile_res = supabase.table("profiles").select("credits").eq("id", user_id).maybe_single().execute()
                current_credits = 0
                profile_exists = False

                if hasattr(profile_res, 'error') and profile_res.error:
                     # Log error but proceed assuming 0 credits if fetch fails, otherwise payment might not grant credits
                     logging.error(f"❌ DB Error fetching profile for credit update (user {user_id}, session {session_id}): {profile_res.error}")
                elif profile_res and profile_res.data:
                    current_credits = profile_res.data.get("credits", 0)
                    profile_exists = True
                    logging.info(f"   Profile exists for {user_id}. Current credits: {current_credits}")
                else:
                     logging.warning(f"⚠️ Profile not found for {user_id} during webhook credit update (Session: {session_id}). Will create/update starting from 0 credits.")

                # 2. Calculate new total
                new_total_credits = current_credits + credits_to_add
                logging.info(f"   Updating credits for {user_id}: {current_credits} -> {new_total_credits}")

                # 3. Update credits (using upsert might be good if profile might not exist)
                update_res = supabase.table("profiles").upsert({
                    "id": user_id, # Ensure the user ID is included for upsert
                    "credits": new_total_credits
                }).execute()

                # Check update/upsert result
                if hasattr(update_res, 'error') and update_res.error:
                    logging.error(f"❌ DB Error updating/upserting credits for {user_id} (Session: {session_id}): {update_res.error}")
                    # Return 500 to signal Stripe to retry the webhook later
                    return jsonify(error="Database update failed during credit grant"), 500
                elif not (hasattr(update_res, 'data') and update_res.data):
                     # This might happen if the upsert somehow failed without an error object
                    logging.error(f"❌ DB Error: Upsert credit query returned no data for {user_id} (Session: {session_id}).")
                    return jsonify(error="Database update failed (no data returned)"), 500
                else:
                    logging.info(f"✅ Credits successfully updated for {user_id} (Session: {session_id}). New balance: {new_total_credits}")

            except Exception as db_error:
                logging.error(f"❌ DB Exception during credit update for {user_id} (Session: {session_id}): {db_error}", exc_info=True)
                # Return 500 to signal Stripe to retry
                return jsonify(error="Database exception during credit grant"), 500
        else:
            # Payment status was not 'paid' (e.g., 'unpaid', 'no_payment_required')
            logging.info(f"Checkout session {session_id} completed but payment status was '{session.get('payment_status')}'. No credits granted.")
    else:
        # Log unhandled event types
        logging.info(f"🤷‍♀️ Received unhandled Stripe event type: {event.get('type')}")

    # Acknowledge receipt of the event to Stripe
    return jsonify(success=True), 200


# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) # Default port 5001
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0") # Default host
    # Read FLASK_DEBUG environment variable (commonly 'true' or '1' for debug)
    debug_mode_str = os.environ.get("FLASK_DEBUG", "False")
    debug_mode = debug_mode_str.lower() in ["true", "1", "t", "yes"]

    # Configure root logger level based on debug mode BEFORE Flask runs its own setup
    log_level = logging.DEBUG if debug_mode else logging.INFO
    # Use force=True to override any potential default Flask logging config
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

    # Explicitly set Flask's app logger level
    app.logger.setLevel(log_level)

    # Log effective levels to confirm
    logging.info(f"Effective root logging level set to: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    logging.info(f"Flask app logger level set to: {logging.getLevelName(app.logger.getEffectiveLevel())}")

    if debug_mode:
        logging.warning("⚠️ Flask is running in DEBUG mode! Ensure this is False in production.")
    else:
        logging.info("Flask is running in PRODUCTION mode (Debug=False).")

    logging.info(f"🚀 Starting Flask server on http://{host}:{port} (Debug: {debug_mode})")
    # use_reloader=debug_mode ensures the reloader is only active in debug mode
    app.run(host=host, port=port, debug=debug_mode, use_reloader=debug_mode)
