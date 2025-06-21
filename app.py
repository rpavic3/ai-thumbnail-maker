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
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont # Pillow for image processing
from supabase import create_client, Client
# Optional, but good practice if dealing with filenames later:
# from werkzeug.utils import secure_filename

# --- Load Environment Variables FIRST ---
load_dotenv()

# Credit Costs
CREDIT_COST_PER_PRECISION_PREVIEW_SET = 1 # Cost for generating a set of 6 preview images (ensure this is the correct/intended value)
CREDIT_COST_HQ_THUMBNAIL = 2 # Cost for generating one HQ thumbnail # Loads variables from .env file into environment

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
CREDIT_COST_PER_PRECISION_PREVIEW_SET = 1 # Cost for generating 6 precision mode previews

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
        "width": 1280, "height": 720,
        "enable_safety_checker": False

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

def generate_flux_hq_image(prompt: str) -> str:
    """Calls FAL/Flux-11-Ultra and returns the public image URL"""
    if not FAL_API_KEY:
        logging.error("❌ Cannot generate Flux HQ image: FAL_API_KEY is not set.")
        raise ValueError("Flux API Key is missing.")

    logging.info(f"🌀 Sending prompt to Flux-11-Ultra: '{prompt}'")
    api_url = "https://fal.run/fal-ai/flux-pro/v1.1-ultra" # Corrected API endpoint
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
    try:
        # Increased timeout as flux-11-ultra might be more resource-intensive
        r = requests.post(api_url, headers=headers, json=payload, timeout=180) 
        r.raise_for_status()
        response_data = r.json()
        if not response_data or "images" not in response_data or not response_data["images"] or "url" not in response_data["images"][0]:
             logging.error(f"❌ Flux-11-Ultra returned unexpected response structure: {response_data}")
             raise ValueError("Flux-11-Ultra response missing expected image URL.")
        image_url = response_data["images"][0]["url"]
        logging.info(f"✅ Flux-11-Ultra responded successfully. Image URL: {image_url}")
        return image_url
    except requests.exceptions.RequestException as req_err:
         logging.error(f"❌ Network error calling Flux-11-Ultra API: {req_err}", exc_info=True)
         raise ConnectionError(f"Network error connecting to Flux-11-Ultra: {req_err}") from req_err
    except Exception as e:
         logging.error(f"❌ Failed Flux-11-Ultra API call: {e}", exc_info=True)
         error_detail = getattr(e, 'response', None)
         error_text = getattr(error_detail, 'text', str(e))
         raise Exception(f"Flux-11-Ultra generation failed: {error_text}") from e

def upscale_image_with_clarity(image_url: str, upscale_factor: float = 2.0) -> str:
    """
    Upscales an image using the fal-ai/clarity-upscaler model
    
    Args:
        image_url: URL of the image to upscale
        upscale_factor: Factor by which to upscale the image (default: 2.0)
        
    Returns:
        URL of the upscaled image
    """
    if not FAL_API_KEY:
        logging.error("❌ Cannot upscale image: FAL_API_KEY is not set.")
        raise ValueError("Flux API Key is missing.")

    logging.info(f"🔍 Upscaling image with clarity-upscaler (factor: {upscale_factor})")
    api_url = "https://fal.run/fal-ai/clarity-upscaler"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "image_url": image_url,
        "upscale_factor": upscale_factor,
        "prompt": "masterpiece, best quality, highres, 4k, detailed",
        "negative_prompt": "(worst quality, low quality, normal quality:2)",
        "creativity": 0.2,  # Lower value to stay closer to original
        "resemblance": 0.8,  # Higher value to maintain resemblance to original
        "guidance_scale": 4.0,
        "num_inference_steps": 20
    }
    
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        response_data = r.json()
        
        if not response_data or "image" not in response_data or "url" not in response_data["image"]:
            logging.error(f"❌ Clarity Upscaler returned unexpected response structure: {response_data}")
            raise ValueError("Clarity Upscaler response missing expected image URL.")
            
        upscaled_image_url = response_data["image"]["url"]
        logging.info(f"✅ Clarity Upscaler responded successfully. Image URL: {upscaled_image_url}")
        return upscaled_image_url
        
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Network error calling Clarity Upscaler API: {req_err}", exc_info=True)
        raise ConnectionError(f"Network error connecting to Clarity Upscaler: {req_err}") from req_err
    except Exception as e:
        logging.error(f"❌ Failed Clarity Upscaler API call: {e}", exc_info=True)
        error_detail = getattr(e, 'response', None)
        error_text = getattr(error_detail, 'text', str(e))
        raise Exception(f"Clarity Upscaler failed: {error_text}") from e

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


def generate_hidream_preview_image(prompt: str) -> str:
    """Calls FAL/hidream-i1-fast and returns the public image URL"""
    if not FAL_API_KEY:
        logging.error("❌ Cannot generate HiDream image: FAL_API_KEY is not set.")
        raise ValueError("Fal API Key is missing.")

    logging.info(f"🎨 Sending prompt to HiDream: '{prompt}'")
    api_url = "https://fal.run/fal-ai/hidream-i1-fast"
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "image_size": "landscape_16_9",
        "num_inference_steps": 16 
    }
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=90) # 90 seconds timeout
        r.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        response_data = r.json()
        
        if not response_data or "images" not in response_data or not isinstance(response_data["images"], list) or len(response_data["images"]) == 0 or "url" not in response_data["images"][0]:
            logging.error(f"❌ HiDream returned unexpected response structure: {response_data}")
            raise ValueError("HiDream response missing expected image URL.")
        
        image_url = response_data["images"][0]["url"]
        logging.info(f"✅ HiDream responded successfully. Image URL: {image_url}")
        return image_url
    except requests.exceptions.Timeout:
        logging.error(f"❌ Timeout error calling HiDream API for prompt: '{prompt}'", exc_info=True)
        raise ConnectionError(f"Timeout connecting to HiDream for prompt: '{prompt}'")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Network error calling HiDream API: {req_err}", exc_info=True)
        raise ConnectionError(f"Network error connecting to HiDream: {req_err}") from req_err
    except Exception as e: # Catch-all for other unexpected errors like JSON parsing
        logging.error(f"❌ Unexpected error generating HiDream image: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate HiDream image: {e}") from e

def get_visual_prompts_from_gpt(video_topic: str) -> list[str]:
    """
    Sends the video topic to GPT-4o-mini to generate six visual thumbnail prompts. 
    Returns a list of 6 prompt strings.
    """
    if not client or not client.api_key or client.api_key == "dummy_key_if_missing": # Check for dummy key
        logging.error("❌ OpenAI client not configured for generating visual prompts.")
        raise ValueError("AI Service (OpenAI) is not configured.")

    system_prompt = """You are a professional YouTube-thumbnail ideator. The video topics/descriptions you receive will not always be perfect, they can be messy and not always have the right keywords. You will need to refine them to make them more specific and visual. The best course of action would be for you to extract the key things from the description, and think like a professional thumbnail creator. When you receive the description, you should basically create 6 different creative video titles for that topic, and then create 6 different visual prompts for that topic. It is important that every prompt is LONG, because that gives us more details and better images.

GOAL
Generate SIX compelling, eyecatching, distinct (like different styles) **and highly specific** visual prompt idea for a YouTube thumbnail based on this video information. The prompts MUST describe concrete visual elements suitable for a literal AI image generator (like Flux or DALL-E). **IMPORTANT INSTRUCTIONS for the prompt content:**
            * **Be Specific:** Describe exact objects, characters, settings, colors, lighting, and composition details.
            * **Avoid Vague Terms:** Do NOT use generic or abstract terms like 'analytics', 'data', 'symbols', 'icons', 'graphs' unless you describe *exactly* what they look like (e.g., 'a glowing blue bar chart showing an upward trend', 'a simple red heart icon floating near the top left'). If you can't describe it visually and specifically, don't include it.
            * **No Decorative Elements:** Avoid adding unnecessary decorative elements like confetti, sparkles, or generic symbols. Focus on core visual elements.
            * **Focus:** Ensure the prompt clearly focuses on the main subject derived from the title and niche.
            * **DETAIL:** Ensure the prompt is highly detailed and specific, with concrete visual elements that an AI image generator can understand.

• Each prompt must be a fully-self-contained visual description that an AI image generator can understand.

RULES
1. No mention of text, captions, logos, watermarks or brand names.
2. Vary them—different angles, colour palettes, or story beats.
3. Use vivid, but concrete language (describe lighting, setting, subject pose).
4. Output ONLY a JSON array of 6 strings. No commentary, headings or markdown."""

    user_message = f"Video topic: {video_topic}\nReturn the six candidate prompts now."

    logging.info(f"🤖 Sending to GPT-4o-mini for visual prompts. Topic: '{video_topic}'")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"} 
        )
        
        response_content = completion.choices[0].message.content
        logging.info(f"✅ GPT-4o-mini raw response for visual prompts: {response_content}")

        prompts_list = []
        try:
            parsed_json = json.loads(response_content)
            if isinstance(parsed_json, list):
                prompts_list = parsed_json
            elif isinstance(parsed_json, dict):
                if "prompts" in parsed_json and isinstance(parsed_json["prompts"], list):
                    prompts_list = parsed_json["prompts"]
                elif "result" in parsed_json and isinstance(parsed_json["result"], list):
                    prompts_list = parsed_json["result"]
                elif parsed_json: 
                    first_key = next(iter(parsed_json))
                    if isinstance(parsed_json[first_key], list):
                        prompts_list = parsed_json[first_key]

            if not prompts_list or not all(isinstance(p, str) for p in prompts_list) or len(prompts_list) != 6:
                logging.error(f"❌ GPT-4o-mini response format error. Expected JSON array of 6 strings, got: {response_content}")
                raise ValueError("GPT-4o-mini response format error: Expected JSON array of 6 strings.")

        except json.JSONDecodeError as e:
            logging.error(f"❌ GPT-4o-mini response was not valid JSON: {response_content}. Error: {e}")
            raise ValueError("GPT-4o-mini response format error: Invalid JSON.")

        logging.info(f"✅ GPT-4o-mini generated {len(prompts_list)} visual prompts successfully.")
        return prompts_list

    except APIError as e:
        logging.error(f"❌ OpenAI API Error during visual prompt generation: {e}", exc_info=True)
        raise ConnectionError(f"AI service (OpenAI) error: {e}") from e
    except Exception as e:
        logging.error(f"❌ Unexpected error during visual prompt generation: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate visual prompts: {e}") from e

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
        
        # Check if email is confirmed
        if user.email_confirmed_at is None:
            logging.warning(f"⛔ User {user.id} has not confirmed their email.")
            raise Exception("Email not verified. Please check your inbox for a verification link.")
            
        user_id = str(user.id)
        logging.info(f"✅ Token validated successfully for user ID: {user_id}")
        return user_id
    except Exception as e:
        logging.error(f"❌ Error during token validation: {e.__class__.__name__}: {e}", exc_info=True)
        error_str = str(e).lower()
        # Check for specific error messages related to token validity
        if "invalid token" in error_str or "jwt" in error_str or "token is invalid" in error_str:
             raise Exception("Invalid token.") from e
        elif "email not verified" in error_str:
             raise Exception("Email not verified. Please check your inbox for a verification link.") from e
        else:
             raise Exception("Token verification failed.") from e

# --- API Routes ---

@app.route('/api/precision/generate-previews', methods=['POST', 'OPTIONS'])
def precision_generate_previews():
    if request.method == "OPTIONS": # Handle CORS preflight
        return "", 204

    logging.info("🎯 /api/precision/generate-previews called")
    
    # Add call stack info to help debug
    import traceback
    stack = traceback.format_stack()
    logging.info(f"📚 Call stack for /api/precision/generate-previews: {stack[-3:]}")
    
    auth_header = request.headers.get("Authorization")
    try:
        user_id = get_user_id_from_token(auth_header)
    except Exception as e:
        logging.warning(f"⚠️ Auth failed for /api/precision/generate-previews: {e}")
        return jsonify({"error": str(e)}), 401

    data = request.json
    video_topic = data.get("video_topic")
    is_refined_prompt = data.get("is_refined_prompt", False)  # New flag to indicate refined prompts
    
    # Log the request details for debugging
    logging.info(f"📝 Request details - video_topic: '{video_topic}', is_refined_prompt: {is_refined_prompt}")
    
    # Check if this is a duplicate request (a simple concatenated prompt)
    # Look for common edit instructions patterns
    common_edit_patterns = [" make it ", " change to ", " add ", " remove "]
    is_likely_concatenated = False
    
    for pattern in common_edit_patterns:
        if pattern in video_topic.lower() and not is_refined_prompt:
            parts = video_topic.split(pattern, 1)
            if len(parts) > 1 and len(parts[0].split()) > 10:  # Original prompt is substantial
                is_likely_concatenated = True
                break
    
    if is_likely_concatenated:
        logging.warning(f"⚠️ Detected concatenated prompt without is_refined_prompt flag. Skipping: '{video_topic}'")
        # Return a success response with empty data to prevent errors in the frontend
        return jsonify({
            "images": [], 
            "prompts": [],
            "credits_remaining": 0,  # Frontend will ignore this as it's an empty response
            "skipped_duplicate": True
        }), 200

    if not video_topic:
        logging.warning("⚠️ /api/precision/generate-previews: Missing video_topic.")
        return jsonify({"error": "Video topic is required."}), 400

    cost = CREDIT_COST_PER_PRECISION_PREVIEW_SET
    try:
        user_profile_response = supabase.table("profiles").select("credits").eq("id", user_id).single().execute()
        
        if not user_profile_response.data:
            logging.error(f"❌ User profile not found for user_id {user_id} during credit check.")
            return jsonify({"error": "User profile not found for credit check."}), 404

        current_credits = user_profile_response.data.get("credits", 0)

        if current_credits < cost:
            logging.warning(f"⚠️ User {user_id} has insufficient credits ({current_credits}) for precision previews (cost: {cost}).")
            return jsonify({"error": "Insufficient credits."}), 402

        new_credits = current_credits - cost
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        
        if hasattr(update_response, 'error') and update_response.error:
            logging.error(f"❌ Failed to deduct credits for user {user_id}: {update_response.error}")
            return jsonify({"error": "Credit deduction failed."}), 500
        
        logging.info(f"🪙 Credits deducted for user {user_id}. Old: {current_credits}, New: {new_credits} (Cost: {cost})")
    
    except Exception as e: 
        logging.error(f"❌ Error during credit check/deduction for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Credit processing error."}), 500

    try:
        # Check if this is a refined prompt from the edit flow
        # We can detect this by checking if is_refined_prompt flag is set
        # or by checking if the prompt is longer/more detailed than typical video topics
        if is_refined_prompt:  # Simple heuristic: refined prompts are usually longer
            logging.info(f"🎨 Using REFINED prompt for all 6 images (from edit flow): '{video_topic}'")
            
            image_urls = [None] * 6 
            generated_prompts_for_fe = [video_topic] * 6  # Same prompt for all 6 slots

            # Generate 6 images with the same refined prompt
            for i in range(6):
                try:
                    img_url = generate_hidream_preview_image(video_topic)
                    image_urls[i] = img_url
                    logging.info(f"✅ Generated preview image {i+1}/6 for user {user_id}")
                except Exception as img_gen_err:
                    logging.error(f"❌ Failed to generate preview image {i+1} for refined prompt (user {user_id}): {img_gen_err}", exc_info=True)
        else:
            # For regular video topics, use the original approach with GPT generating 6 different prompts
            logging.info(f"🧠 Generating 6 different visual prompts for video topic: '{video_topic}'")
            
            visual_prompts = get_visual_prompts_from_gpt(video_topic)
            if not visual_prompts or len(visual_prompts) != 6:
                logging.error(f"❌ Did not receive exactly 6 prompts from GPT for user {user_id}. Received: {visual_prompts}")
                return jsonify({"error": "Failed to generate the required number of thumbnail ideas."}), 500

            image_urls = [None] * 6 
            generated_prompts_for_fe = [None] * 6 

            for i in range(6):
                p_prompt = visual_prompts[i]
                generated_prompts_for_fe[i] = p_prompt 
                if p_prompt: 
                    try:
                        img_url = generate_hidream_preview_image(p_prompt)
                        image_urls[i] = img_url
                        logging.info(f"✅ Generated preview image {i+1}/6 for user {user_id}")
                    except Exception as img_gen_err:
                        logging.error(f"❌ Failed to generate preview image {i+1} for prompt '{p_prompt}' (user {user_id}): {img_gen_err}", exc_info=True)
                else:
                    logging.warning(f"⚠️ Skipping image generation for empty prompt at index {i} for user {user_id}.")

        successful_image_count = sum(1 for url in image_urls if url is not None)
        if successful_image_count < 6:
             logging.warning(f"⚠️ Only {successful_image_count}/6 images generated successfully for user {user_id}.")

        return jsonify({
            "images": image_urls, 
            "prompts": generated_prompts_for_fe, 
            "credits_remaining": new_credits
        }), 200

    except ValueError as ve: 
        logging.error(f"❌ Configuration error for /api/precision/generate-previews (user {user_id}): {ve}", exc_info=True)
        return jsonify({"error": f"Service configuration error: {ve}"}), 503
    except ConnectionError as ce:
        logging.error(f"❌ Connection error during /api/precision/generate-previews (user {user_id}): {ce}", exc_info=True)
        return jsonify({"error": f"External service connection error: {ce}"}), 502
    except RuntimeError as rte: 
        logging.error(f"❌ Runtime error during /api/precision/generate-previews (user {user_id}): {rte}", exc_info=True)
        return jsonify({"error": f"Processing error: {rte}"}), 500
    except Exception as e: 
        logging.error(f"❌ Unexpected critical error in /api/precision/generate-previews (user {user_id}): {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/api/precision/generate-hq-thumbnail', methods=['POST', 'OPTIONS'])
def precision_generate_hq_thumbnail():
    if request.method == "OPTIONS": # Handle CORS preflight
        return "", 204

    logging.info("🎯 /api/precision/generate-hq-thumbnail called")
    auth_header = request.headers.get("Authorization")
    try:
        user_id = get_user_id_from_token(auth_header)
    except Exception as e:
        logging.warning(f"⚠️ Auth failed for /api/precision/generate-hq-thumbnail: {e}")
        return jsonify({"error": str(e)}), 401

    data = request.json
    base_prompt = data.get("prompt")
    text_overlays = data.get("text_overlays") # Expects an array of text overlay objects

    if not base_prompt:
        logging.warning("⚠️ /api/precision/generate-hq-thumbnail: Missing base_prompt.")
        return jsonify({"error": "Base prompt is required."}), 400
    
    # Validate text_overlays if present
    if text_overlays is not None: # Allow empty list, but if key exists, it must be a list
        if not isinstance(text_overlays, list):
            logging.warning("⚠️ /api/precision/generate-hq-thumbnail: text_overlays is not a valid list.")
            return jsonify({"error": "Invalid text_overlays format. Expected a list."}), 400
        for i, overlay in enumerate(text_overlays):
            if not isinstance(overlay, dict):
                logging.warning(f"⚠️ /api/precision/generate-hq-thumbnail: item {{i}} in text_overlays is not a valid object.")
                return jsonify({"error": f"Invalid item at index {{i}} in text_overlays. Expected an object."}), 400
            # Each overlay should ideally have text, but we'll allow empty text strings;
            # Pillow won't draw them, and frontend should filter if necessary.
            # if not overlay.get("text") or not str(overlay.get("text")).strip():
            #     logging.warning(f"⚠️ /api/precision/generate-hq-thumbnail: item {{i}} in text_overlays is missing 'text' property or text is empty.")
            #     return jsonify({"error": f"Item at index {{i}} in text_overlays requires non-empty 'text' property."}), 400

    cost = CREDIT_COST_HQ_THUMBNAIL
    new_credits = 0
    try:
        user_profile_response = supabase.table("profiles").select("credits").eq("id", user_id).single().execute()
        if not user_profile_response.data:
            logging.error(f"❌ User profile not found for user_id {user_id} during credit check for HQ thumbnail.")
            return jsonify({"error": "User profile not found for credit check."}), 404
        current_credits = user_profile_response.data.get("credits", 0)
        if current_credits < cost:
            logging.warning(f"⚠️ User {user_id} has insufficient credits ({current_credits}) for HQ thumbnail (cost: {cost}).")
            return jsonify({"error": "Insufficient credits."}), 402
        new_credits = current_credits - cost
        update_response = supabase.table("profiles").update({"credits": new_credits}).eq("id", user_id).execute()
        if hasattr(update_response, 'error') and update_response.error:
            logging.error(f"❌ Failed to deduct credits for HQ thumbnail for user {user_id}: {update_response.error}")
            return jsonify({"error": "Credit deduction failed."}), 500
        logging.info(f"🪙 Credits deducted for HQ thumbnail for user {user_id}. Old: {current_credits}, New: {new_credits} (Cost: {cost})")
    except Exception as e:
        logging.error(f"❌ Error during credit check/deduction for HQ thumbnail for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Credit processing error."}), 500

    # Image generation prompt does NOT include text overlay instructions anymore
    logging.info(f"ℹ️ Using base prompt for Fal image generation for user {user_id}: '{base_prompt}'")

    try:
        # Check if original_image_url is provided for upscaling
        if data.get("original_image_url"):
            original_image_url = data.get("original_image_url")
            logging.info(f"🔍 Using clarity-upscaler on original image for user {user_id}")
            base_image_url = upscale_image_with_clarity(original_image_url, 2.0)
            logging.info(f"✅ Image upscaled with clarity-upscaler for user {user_id}. URL: {base_image_url}")
        else:
            # No original image URL provided, generate a new image with Flux
            base_image_url = generate_flux_hq_image(base_prompt)
            logging.info(f"🖼️ Base HQ image generated via Fal for user {user_id}. URL: {base_image_url}")

        # 2. Fetch the generated image
        image_response = requests.get(base_image_url)
        image_response.raise_for_status() # Raise an exception for bad status codes
        img_pil = Image.open(io.BytesIO(image_response.content)).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)

        final_image_data_url = base_image_url # Default to base if no text details

        # 3. If text_overlays are provided and valid, draw each text using Pillow
        if text_overlays and isinstance(text_overlays, list) and len(text_overlays) > 0:
            logging.info(f"✍️ {len(text_overlays)} text overlay(s) provided by user {user_id}. Rendering text with Pillow.")
            valid_overlays_processed = 0
            for i, td in enumerate(text_overlays): # td for text_details of current overlay
                text_content = td.get("text", "")
                if not text_content.strip(): # Skip if text is empty or only whitespace
                    logging.info(f"Skipping text overlay {i+1} due to empty/whitespace text content.")
                    continue

                logging.info(f"Processing text overlay {i+1}/{len(text_overlays)}: '{text_content[:30]}...' ({td.get('font_filename', td.get('font_name', 'DefaultFont'))} @ {td.get('font_size', 30)}pt)")
                
                font_name = td.get("font_name", "Arial").strip() # Default font name if not specified
                font_filename_from_frontend = td.get("font_filename", "").strip() # Actual .ttf filename from frontend
                font_size = int(td.get("font_size", 30))
                text_x = float(td.get("x", 0))
                text_y = float(td.get("y", 0))
                canvas_w = float(td.get("canvas_width", img_pil.width)) # original canvas width from frontend
                canvas_h = float(td.get("canvas_height", img_pil.height)) # original canvas height from frontend
                
                fill_color = td.get("color", "#FFFFFF")
                stroke_color = td.get("stroke_color", "#000000") 
                stroke_width = int(td.get("stroke_width", 0))

                # Font loading (fonts directory is at project root, relative to app.py's location)
                font_file_to_load = ""
                font_path = ""
                try:
                    if font_filename_from_frontend and font_filename_from_frontend.lower().endswith(".ttf"):
                        font_file_to_load = font_filename_from_frontend
                    elif font_name:
                        font_file_to_load = font_name.lower().replace(" ", "") + ".ttf"
                    else:
                        logging.warning(f"⚠️ No valid font_name or font_filename for overlay {i+1}. Attempting default.")
                        raise IOError("Missing font name/filename for overlay.")

                    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
                    if not os.path.exists(font_dir):
                        try:
                            os.makedirs(font_dir)
                            logging.info(f"✅ Created missing fonts directory: {font_dir}")
                        except OSError as e_mkdir:
                            logging.error(f"❌ Failed to create fonts directory {font_dir}: {e_mkdir}. Font loading for overlay {i+1} might fail.")
                    
                    font_path = os.path.join(font_dir, font_file_to_load)
                    logging.info(f"Attempting to load font for overlay {i+1}: {font_path} (size: {font_size})")
                    image_font = ImageFont.truetype(font_path, font_size)
                except IOError as font_io_error:
                    logging.warning(f"⚠️ Font '{font_path if font_path else font_file_to_load}' not found/loadable for overlay {i+1}: {font_io_error}. Falling back.")
                    try:
                        image_font = ImageFont.truetype("arial.ttf", font_size) 
                        logging.info(f"Successfully fell back to arial.ttf for overlay {i+1}.")
                    except IOError:
                        logging.warning(f"⚠️ Arial.ttf also not found for overlay {i+1}. Using Pillow's load_default().")
                        image_font = ImageFont.load_default()
                except Exception as e_font_load:
                    logging.error(f"❌ Unexpected error loading font for overlay {i+1} (Name: '{font_name}', File: '{font_file_to_load}'): {e_font_load}")
                    image_font = ImageFont.load_default()

                # Coordinate scaling
                img_w, img_h = img_pil.size
                scaled_x = (text_x / canvas_w) * img_w if canvas_w > 0 else 0
                scaled_y = (text_y / canvas_h) * img_h if canvas_h > 0 else 0

                # Draw text
                if stroke_width > 0:
                    draw.text((scaled_x, scaled_y), text_content, font=image_font, fill=fill_color, 
                              stroke_width=stroke_width, stroke_fill=stroke_color, anchor="la")
                else:
                    draw.text((scaled_x, scaled_y), text_content, font=image_font, fill=fill_color, anchor="la")
                valid_overlays_processed += 1
            
            if valid_overlays_processed > 0:
                buffered = io.BytesIO()
                img_pil.save(buffered, format="PNG")
                img_str_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                final_image_data_url = f"data:image/png;base64,{img_str_base64}"
                logging.info(f"✅ {valid_overlays_processed} text overlay(s) rendered on image for user {user_id}. Returning base64 PNG.")
            else: # No valid overlays were actually drawn (e.g., all had empty text)
                logging.info(f"ℹ️ No valid text overlays to render for user {user_id} after filtering. Returning base Fal image as base64 PNG.")
                buffered = io.BytesIO()
                img_pil.save(buffered, format="PNG")
                img_str_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                final_image_data_url = f"data:image/png;base64,{img_str_base64}"

        else: # No text_overlays provided, or list is empty initially
            logging.info(f"ℹ️ No text overlays provided or list is empty for user {user_id}. Returning base Fal image as base64 PNG.")
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            final_image_data_url = f"data:image/png;base64,{img_str_base64}"

        return jsonify({
            "image_url": final_image_data_url, # This is now a data URL
            "credits_remaining": new_credits
        }), 200

    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Failed to fetch base image from URL for Pillow processing (user {user_id}): {req_err}", exc_info=True)
        return jsonify({"error": f"Failed to load base image for text rendering: {req_err}"}), 502
    except UnidentifiedImageError:
        logging.error(f"❌ Could not identify image from URL (user {user_id}). Content might not be a valid image.")
        return jsonify({"error": "Invalid image format received from image generator."}), 502
    except ValueError as ve:
        logging.error(f"❌ Configuration or API response error for HQ thumbnail (user {user_id}): {ve}", exc_info=True)
        return jsonify({"error": f"Image generation configuration error: {ve}"}), 503
    except ConnectionError as ce:
        logging.error(f"❌ Connection error during HQ thumbnail generation (user {user_id}): {ce}", exc_info=True)
        return jsonify({"error": f"External service connection error: {ce}"}), 502
    except Exception as e:
        logging.error(f"❌ Unexpected critical error in HQ thumbnail generation (user {user_id}): {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred during image generation."}), 500



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
    generated_image_urls = []
    flux_errors = []
    logging.info(f"⏳ Requesting {NUM_IMAGES_TO_GENERATE} images from Flux for user {user_id}...")
    for i in range(NUM_IMAGES_TO_GENERATE):
        logging.info(f"   Generating image {i+1}/{NUM_IMAGES_TO_GENERATE}...")
        try:
            img_url = generate_flux_image(combined_prompt)
            logging.info(f"     Got image URL from Flux: {img_url}")
            generated_image_urls.append(img_url); logging.info(f"   ✅ Image {i+1} URL received.")
        except Exception as e: logging.error(f"❌ Flux failed to generate image {i+1}: {e}", exc_info=True); flux_errors.append(f"Img {i+1}: {e}"); generated_image_urls.append(None)

    if not any(generated_image_urls): logging.error(f"❌ Flux generation failed entirely."); return jsonify({"error": f"Image generation failed: {'; '.join(flux_errors)}"}), 502
    logging.info(f"✅ Successfully generated {sum(1 for u in generated_image_urls if u)}/{NUM_IMAGES_TO_GENERATE} images.")

    # 5. --- Process Each Generated Image ---
    processed_results_urls = []
    db_save_tasks = []
    for index, img_url in enumerate(generated_image_urls):
        if img_url:
            logging.info(f"⚙️ Processing image {index + 1}...")
            # No need to generate preview - we'll use the same URL for both
            # Client can resize image for preview display
            processed_results_urls.append(img_url) # Keep original URL
            db_save_tasks.append({"user_id": user_id, "title": title, "niche": niche, "prompt": original_image_prompt, "image_url": img_url, "preview_image_url": img_url})
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

@app.route('/api/precision/refine-prompt', methods=['POST', 'OPTIONS'])
def precision_refine_prompt():
    if request.method == "OPTIONS": # Handle CORS preflight
        return "", 204

    logging.info("🎯 /api/precision/refine-prompt called")
    auth_header = request.headers.get("Authorization")
    try:
        user_id = get_user_id_from_token(auth_header)
    except Exception as e:
        logging.warning(f"⚠️ Auth failed for /api/precision/refine-prompt: {e}")
        return jsonify({"error": str(e)}), 401

    data = request.json
    original_prompt = data.get("original_prompt")
    refinement_instructions = data.get("refinement_instructions")
    image_url = data.get("image_url")  # Optional, for reference

    logging.info(f"📝 Received refinement request - User:{user_id}, Original Prompt: '{original_prompt}'")
    logging.info(f"📝 Refinement Instructions: '{refinement_instructions}'")
    if image_url:
        logging.info(f"🖼️ Reference Image URL provided: '{image_url}'")

    if not original_prompt or not refinement_instructions:
        logging.warning("⚠️ /api/precision/refine-prompt: Missing original_prompt or refinement_instructions.")
        return jsonify({"error": "Original prompt and refinement instructions are required."}), 400

    # No credit check for prompt refinement - it's part of the edit flow which already costs credits

    if not client or not client.api_key or client.api_key == "dummy_key_if_missing": # Check for dummy key
        logging.error("❌ OpenAI client not configured for /api/precision/refine-prompt.")
        return jsonify({"error": "AI Service (OpenAI) is not configured."}), 503

    try:
        system_prompt = """You are an expert at refining image generation prompts. Given an original prompt and refinement instructions, create a new coherent prompt that incorporates the requested changes while maintaining the overall style and intent of the original.

Your task is to seamlessly integrate the refinement instructions into the original prompt, creating a single cohesive prompt that an AI image generator can understand. The refined prompt should read naturally as a complete instruction, not as two separate parts.

DO NOT include phrases like "based on the original prompt" or references to the refinement process itself.
DO NOT output anything except the refined prompt - no explanations or additional text.
"""

        logging.info(f"🤖 Sending to GPT-4o-mini for prompt refinement.")
        logging.info(f"🤖 System Prompt: {system_prompt}")
        
        user_message = f"""Original prompt: {original_prompt}
Refinement instructions: {refinement_instructions}

Return only the refined prompt with no explanations or additional text."""

        logging.info(f"🤖 User Message: {user_message}")

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        refined_prompt = completion.choices[0].message.content.strip()
        logging.info(f"✅ GPT-4o-mini refined prompt: '{refined_prompt}'")
        
        # DEV: Log the comparison between original and refined prompts
        logging.info("=== PROMPT REFINEMENT COMPARISON ===")
        logging.info(f"ORIGINAL: {original_prompt}")
        logging.info(f"INSTRUCTIONS: {refinement_instructions}")
        logging.info(f"REFINED: {refined_prompt}")
        logging.info("===================================")

        # Log token usage for monitoring
        if hasattr(completion, 'usage') and completion.usage:
            prompt_tokens = getattr(completion.usage, 'prompt_tokens', 'N/A')
            completion_tokens = getattr(completion.usage, 'completion_tokens', 'N/A')
            total_tokens = getattr(completion.usage, 'total_tokens', 'N/A')
            logging.info(f"📊 Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        return jsonify({
            "refined_prompt": refined_prompt,
            "original_prompt": original_prompt,  # Return original for comparison
            "refinement_instructions": refinement_instructions  # Return instructions for reference
        }), 200

    except APIError as e:
        logging.error(f"❌ OpenAI API Error during prompt refinement: {e}", exc_info=True)
        return jsonify({"error": f"AI service (OpenAI) error: {e}"}), 502
    except Exception as e:
        logging.error(f"❌ Unexpected error during prompt refinement: {e}", exc_info=True)
        return jsonify({"error": f"Failed to refine prompt: {e}"}), 500

@app.route("/api/auth/check-email", methods=["POST", "OPTIONS"])
def check_email_exists():
    """Check if an email exists in the system without revealing too much information."""
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return "", 204
        
    data = request.json
    email = data.get("email")
    
    if not email:
        return jsonify({"error": "Email is required"}), 400
        
    try:
        # For security reasons, we don't want to directly reveal if an email exists
        # Instead, we'll check internally but return a generic message
        
        # Get user by email - this is for internal use only
        res = supabase.auth.admin.list_users()
        users = res.users if hasattr(res, 'users') else []
        
        email_exists = any(user.email == email for user in users)
        logging.info(f"Email check for {email}: {'Exists' if email_exists else 'Not found'}")
        
        # Always return the same message regardless of result
        return jsonify({
            "message": "If this email is registered, you can log in or reset your password."
        }), 200
    except Exception as e:
        logging.error(f"❌ Error checking email existence: {e}", exc_info=True)
        # For security, don't reveal specific errors
        return jsonify({
            "message": "If this email is registered, you can log in or reset your password."
        }), 200

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