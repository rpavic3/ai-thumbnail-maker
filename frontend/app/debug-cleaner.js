/**
 * Debug Cleaner - A utility to disable console.log in production
 * 
 * Instructions:
 * 1. Include this file at the top of your HTML before any other scripts
 * 2. In production mode, set window.DEBUG_MODE = false
 * 3. In development mode, set window.DEBUG_MODE = true
 */

(function() {
    // Check if we're in production mode (default to false to be safe)
    window.DEBUG_MODE = false;
    
    // Store original console methods
    const originalConsole = {
        log: console.log,
        warn: console.warn,
        error: console.error,
        info: console.info,
        debug: console.debug
    };
    
    // Only override if not in debug mode
    if (!window.DEBUG_MODE) {
        // Replace console.log with a no-op
        console.log = function() {};
        
        // Keep error and warning functions intact for debugging
        // console.warn and console.error are still active
        
        // Optionally disable less critical methods
        console.info = function() {};
        console.debug = function() {};
        
        // Add a method to temporarily restore console for debugging
        window.enableConsoleLogging = function(temporaryOnly = true, durationMs = 5000) {
            console.log = originalConsole.log;
            console.info = originalConsole.info;
            console.debug = originalConsole.debug;
            
            console.log('Console logging temporarily enabled');
            
            if (temporaryOnly) {
                setTimeout(() => {
                    console.log('Console logging disabled again');
                    console.log = function() {};
                    console.info = function() {};
                    console.debug = function() {};
                }, durationMs);
            }
        };
        
        // Log a single message when this script loads to verify it's working
        originalConsole.log('Console logging disabled. Call window.enableConsoleLogging() to temporarily enable logs.');
    }
})(); 