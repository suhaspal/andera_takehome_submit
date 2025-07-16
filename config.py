"""
Configuration settings for the General-Purpose Browsing Agent
Modify these settings to customize agent behavior.
"""

# LLM Model Configuration
DEFAULT_MODEL = "gpt-4o"  # Options: gpt-4o, gpt-4o-mini, claude-3-opus, etc.
MODEL_TEMPERATURE = 0.1   # Lower = more consistent, Higher = more creative (0.0-1.0)

# Browser Configuration
HEADLESS_MODE = False           # Set to True to hide browser window
BROWSER_WIDTH = 1920           # Browser window width
BROWSER_HEIGHT = 1080          # Browser window height
WAIT_BETWEEN_ACTIONS = 1.0     # Seconds to wait between browser actions
DISABLE_SECURITY = True        # Allows broader website access
KEEP_BROWSER_ALIVE = True      # Keep browser session persistent

# Agent Behavior Configuration
MAX_STEPS_DEFAULT = 50         # Default maximum steps per task
MAX_STEPS_COMPLEX = 75         # Steps for complex tasks (booking, shopping)
MAX_FAILURES = 5               # Retry attempts for failed actions
RETRY_DELAY = 2                # Seconds to wait before retrying
MAX_ACTIONS_PER_STEP = 3       # Actions per step (lower = more reliable)

# Vision and Analysis Configuration
USE_VISION = True              # Enable screenshot analysis
USE_THINKING = True            # Enable AI reasoning
IMAGES_PER_STEP = 1            # Screenshots per step

# Reliability Settings (prioritize reliability over speed)
RELIABILITY_MODE = True        # Enable all reliability features

# Scroll Configuration for Better Information Capture
SCROLL_SETTINGS = {
    "conservative_scrolling": True,     # Enable conservative scroll behavior
    "default_scroll_pages": 0.3,       # Scroll only 30% of viewport (vs 1.0 full page)
    "overlap_scrolling": True,          # Provide content overlap between scrolls
    "scroll_pause_time": 1.0,           # Pause after scrolling to analyze content
}

# If reliability mode is enabled, override settings for maximum reliability
if RELIABILITY_MODE:
    WAIT_BETWEEN_ACTIONS = 0.5    # Faster execution while maintaining reliability
    MAX_FAILURES = 7              # More retry attempts
    RETRY_DELAY = 2               # Moderate wait between retries
    MAX_ACTIONS_PER_STEP = 2      # Fewer actions per step
    MODEL_TEMPERATURE = 0.05      # Very consistent behavior
    
    # Apply conservative scroll settings when reliability mode is enabled
    SCROLL_SETTINGS["default_scroll_pages"] = 0.3  # Even more conservative in reliability mode

# File and Data Configuration
BROWSER_DATA_DIR = "./browser_data"    # Where to store browser profiles
SAVE_CONVERSATIONS = False             # Save conversation history
CONVERSATION_PATH = "./conversations"   # Where to save conversations

# Universal Configuration - No Task-Specific Logic
UNIVERSAL_MAX_STEPS = 100             # Single timeout for all tasks
DEFAULT_SEARCH_ENGINE = "google.com"  # Universal search starting point

# Logging Configuration
LOG_LEVEL = "INFO"                     # DEBUG, INFO, WARNING, ERROR
DETAILED_LOGGING = True                # Enable detailed step logging
LOG_BROWSER_ACTIONS = True             # Log all browser interactions

# Error Handling Configuration
GRACEFUL_DEGRADATION = True            # Try alternative approaches on failures
FALLBACK_STRATEGIES = True             # Use backup plans for common failures
CONTINUE_ON_MINOR_ERRORS = True        # Don't stop task on minor issues

# Safety and Ethics Configuration
RESPECT_ROBOTS_TXT = False             # Follow robots.txt (may limit capabilities)
AVOID_CAPTCHAS = True                  # Skip sites with CAPTCHAs when possible
NO_ACTUAL_PURCHASES = True             # Never complete actual purchases
NO_PERSONAL_DATA_ENTRY = True          # Avoid entering real personal information

# Performance Configuration  
PARALLEL_PROCESSING = False            # Process multiple tasks simultaneously
CACHE_RESULTS = True                   # Cache search results for similar queries
OPTIMIZE_FOR_SPEED = False             # Prioritize speed over reliability

# Memory Optimization Configuration
MEMORY_OPTIMIZATION = True             # Enable memory optimization features
MAX_HISTORY_ITEMS = 15                 # Limit history to prevent memory issues
CLEANUP_INTERVAL = 10                  # Clean up every N steps
REDUCE_IMAGE_QUALITY = True            # Lower screenshot quality to save memory
MAX_PAGE_SIZE_MB = 50                  # Skip pages larger than this

# Advanced Configuration
CUSTOM_USER_AGENT = None               # Custom browser user agent string
PROXY_SETTINGS = None                  # Proxy configuration if needed
CUSTOM_HEADERS = {}                    # Additional HTTP headers

# Feature Flags
ENABLE_EXPERIMENTAL_FEATURES = False   # Enable beta/experimental features
ENABLE_STRUCTURED_OUTPUT = False       # Use structured data extraction
ENABLE_MULTI_TAB = True                # Allow multiple browser tabs
ENABLE_DOWNLOADS = True                # Allow file downloads

def get_config():
    """
    Return the current configuration as a dictionary.
    
    Returns:
        dict: Configuration settings
    """
    return {
        # Model settings
        "model": DEFAULT_MODEL,
        "temperature": MODEL_TEMPERATURE,
        
        # Browser settings
        "headless": HEADLESS_MODE,
        "window_width": BROWSER_WIDTH,
        "window_height": BROWSER_HEIGHT,
        "wait_between_actions": WAIT_BETWEEN_ACTIONS,
        "disable_security": DISABLE_SECURITY,
        "keep_alive": KEEP_BROWSER_ALIVE,
        "user_data_dir": BROWSER_DATA_DIR,
        
        # Agent settings
        "max_steps": MAX_STEPS_DEFAULT,
        "max_failures": MAX_FAILURES,
        "retry_delay": RETRY_DELAY,
        "max_actions_per_step": MAX_ACTIONS_PER_STEP,
        "use_vision": USE_VISION,
        "use_thinking": USE_THINKING,
        "images_per_step": IMAGES_PER_STEP,
        
        # Scroll settings
        "scroll": SCROLL_SETTINGS,
        
        # Universal settings
        "max_steps": UNIVERSAL_MAX_STEPS,
        "search_engine": DEFAULT_SEARCH_ENGINE,
        
        # Safety settings
        "no_purchases": NO_ACTUAL_PURCHASES,
        "no_personal_data": NO_PERSONAL_DATA_ENTRY,
    }

def get_universal_config():
    """
    Get universal configuration that works for all task types.
    No task-specific logic - truly generalizable.
    
    Returns:
        dict: Universal configuration for all tasks
    """
    return get_config() 