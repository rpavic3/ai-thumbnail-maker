/**
 * This file contains a list of console.log statements to remove from the application
 * 
 * These logs are primarily used during development and are not needed in production.
 * 
 * Common patterns:
 * 1. DEBUG logs - marked with [DEBUG]
 * 2. UI element logs - "Attaching listener to...", "Setting up click listener..."
 * 3. Mouse position tracking - "Mouse position:", "Dragging - Mouse position"
 * 4. Value dumps - showing JSON, arrays, or object values
 * 5. DOM events - "clicked", "mousedown", etc.
 */

const debugLogsToRemove = [
    // UI setup logs
    "console.log('Attempting to attach click listener to precisionPreviewsGrid:",
    "console.log('precisionPreviewsGrid clicked. Event target:",
    "console.log('Attempted to find .precision-use-button. Result:",
    "console.log('[DEBUG] Checking addTextObjectButton listener. Has data-listener-attached:",
    "console.log('[DEBUG] Attaching click listener to addTextObjectButton NOW.");
    "console.log('Setting up click listener for upscale button');",
    
    // Debug logs
    "console.log('[DEBUG]",
    
    // Mouse tracking
    "console.log('Canvas mousedown event triggered');",
    "console.log('Mouse position:",
    "console.log('Starting resize operation');",
    "console.log('Clicked object index:",
    "console.log('Selected text object:",
    "console.log('Starting drag operation');",
    "console.log('Resizing - Mouse position:",
    "console.log('Dragging - Mouse position:",
    "console.log('Moving text from",
    "console.log('Stopping resize operation');",
    "console.log('Stopping drag operation');",
    "console.log('Mouse is over text after drag:",
    "console.log('Hovering over text object:",
    "console.log('Mouse left canvas during drag or resize operation');",
    
    // Upscaling logs
    "console.log('Upscale button clicked');",
    "console.log('--- Upscaling Image ---');",
    "console.log('Current canvas captured as data URL for upscaling');",
    "console.log('Canvas dimensions:", 
    "console.log('Data URL starts with:",
    "console.log('Development mode: Using local upscaling simulation for placeholder image');",
    "console.log('Current prompt:",
    "console.log('Current image URL:",
    "console.log('Image is data URL:",
    
    // Auth and user data logs
    "console.log('Google Sign-In initiated",
    "console.log(`Updating view. Logged in:",
    "console.log(\"Fetching user credits...\");",
    "console.log(\"Signup Data Object:",
    "console.log(\"--- Logout function called ---\");",
    "console.log(\"--- Supabase signOut successful (callback pending) ---\");",
    
    // Asset history logs
    "console.log(\"FETCH ASSET HISTORY",
    "console.log(\"isLoadingAssetHistory:",
    "console.log(\"Exiting: Already loading more history.\");",
    "console.log(\"Proceeding to fetch/toggle history display.\");",
    "console.log(\"Toggling history OFF\");",
    "console.log(\"Toggling history ON - preparing for fetch.\");",
    "console.log(\"Loading more history...\");",
    "console.log(`Workspaceing asset history from:",
    "console.log(`Workspace response status:",
    "console.log(`Received ${items.length} items. Total items:",
    "console.log(\"Currently loaded asset card count:\",
    "console.log(\"FETCH ASSET HISTORY FINISHED",
    "console.log(\"FETCH ASSET HISTORY FINALLY",
    
    // Style profile logs
    "console.log(\"Executing loadSavedStyles...\");",
    "console.log(\"loadSavedStyles: No active session. Styles cannot be loaded.\");",
    "console.log(\"Fetched styles:\",
    "console.log(\"After removing duplicates:",
    "console.log(\"Processing profile:",
    "console.log(\"Style card with ID",
    "console.log(\"Styles loaded and UI populated",
];

// These logs should be kept as they provide important application state information
const logsToKeep = [
    "console.log('Error loading image onto canvas:",
    "console.error(",
    "console.warn(",
]; 