"""
Voice AI Chat Service Prompts and Query Logic
Centralized management of Voice AI system prompts, command mappings, and response generation

This module contains:
1. System prompt for PrintChakra AI context
2. Voice command mappings for document control
3. Query generation logic for Ollama API
4. Response formatting and validation
"""

from typing import Dict, Any, List
import copy
import json
import logging
from pathlib import Path

from app.config.settings import AI_PROMPT_CONFIG, COMMAND_MAPPINGS_FILE, SYSTEM_PROMPT_FILE

logger = logging.getLogger(__name__)


# ============================================================================
DEFAULT_SYSTEM_PROMPT = """You are PrintChakra AI - a concise voice assistant for document printing and scanning.

CRITICAL RULES:
1. Keep responses to 20 words MAXIMUM - write complete sentences naturally within this limit.
2. NO truncation - responses should be full, coherent sentences (not cut off mid-way).
3. Be direct and action-focused.
4. Use simple confirmations like "Got it", "Done", "Ready".
5. Do NOT add unnecessary pleasantries or filler words.

RESPONSE STYLE:
- "Starting print." (2 words) NOT long explanations
- "Landscape mode set." (3 words) NOT detailed descriptions
- "3 copies ready." (3 words) NOT verbose confirmations

DOCUMENT CONTROL:
- "Document 2 selected." (3 words)
- "Switched to converted." (3 words)
- "Next document." (2 words)

SETTINGS (confirm briefly):
- "Grayscale." (1 word)
- "300 DPI." (2 words)
- "Landscape." (1 word)
- "2 copies." (2 words)

GLOBAL:
- "Done!" (1 word)
- "Cancelled." (1 word)
- "Say: print, scan, select document." (5 words)
- "Ready." (1 word)
- "Welcome!" (1 word)

Write COMPLETE sentences that naturally fit in 20 words. Do NOT write long responses and expect them to be cut off. Example of GOOD response: "Print mode ready. Select documents or browse files." (8 words - complete)."""


DEFAULT_COMMAND_CONFIG = {
    "voice_commands": {
        "select_document": [
            "select document", "choose document", "pick document", "open document",
            "select original document", "select converted document", "select uploaded document",
            "document number", "file number", "document", "file"
        ],
        "select_multiple_documents": [
            "select first two", "select first three", "select first four", "select first five",
            "select first 2", "select first 3", "select first 4", "select first 5",
            "select two documents", "select three documents", "select four documents",
            "select 2 documents", "select 3 documents", "select 4 documents", "select 5 documents",
            "select documents 1 and 2", "select documents 1 2", "select 1 and 2",
            "first two documents", "first three documents", "first four documents",
            "first 2 documents", "first 3 documents", "first 4 documents", "first 5 documents",
            "select all documents", "select all", "choose all", "all documents",
            "last two documents", "last three documents", "last 2 documents", "last 3 documents",
            "select last two", "select last three", "select last 2", "select last 3"
        ],
        "select_document_range": [
            "select documents 1 to", "select document 1 to", "documents 1 to",
            "from document 1 to", "between document", "documents between",
            "select from 1 to", "from 1 to", "1 to 5", "2 to 4",
            "select 1 through", "1 through 5", "documents 1 through"
        ],
        "deselect_document": [
            "deselect", "unselect", "remove selection", "deselect document",
            "unselect document", "remove document", "deselect all", "clear selection",
            "clear document", "clear documents", "clear"
        ],


        "switch_section": [
            "switch to", "show", "go to", "open section",
            "switch section", "change to"
        ],
        "next_document": [
            "next", "next document", "next file",
            "forward", "move forward", "go forward"
        ],
        "previous_document": [
            "previous", "previous document", "previous file",
            "back", "move back", "go back"
        ],
        "upload_document": [
            "upload", "upload document", "upload file",
            "add document", "add file", "new document"
        ],
        "confirm": [
            "confirm", "confirm print", "confirm scan",
            "yes proceed", "execute", "do it", "start", "begin"
        ],
        "cancel": [
            "cancel", "stop", "abort",
            "don't do it", "nevermind", "forget it"
        ],
        "status": [
            "status", "what's happening", "progress",
            "where are we", "current state"
        ],
        "repeat_settings": [
            "repeat", "repeat settings", "read settings",
            "what settings", "show settings", "current settings"
        ],
        "help": [
            "help", "what can you do", "commands",
            "options", "guide"
        ],
        "stop_recording": [
            "stop recording", "stop listening", "end session",
            "bye", "goodbye", "stop voice"
        ],
        "greeting": [
            "hello", "hi", "hey", "greetings", "howdy",
            "good morning", "good afternoon", "good evening",
            "hi there", "hello there"
        ],
        # Settings commands
        "set_color_mode": [
            "grayscale", "greyscale", "gray scale", "grey scale",
            "black and white", "black & white", "bw", "mono", "monochrome",
            "color mode", "in color", "full color", "set color", "change color",
            "make it gray", "make it grey", "use grayscale", "no color"
        ],
        "set_quality": [
            "high quality", "best quality", "photo quality", "premium quality",
            "draft quality", "low quality", "fast print", "economy mode",
            "normal quality", "standard quality", "set quality", "change quality"
        ],
        "set_format": [
            "save as pdf", "as pdf", "pdf format", "export pdf",
            "save as jpeg", "as jpg", "jpeg format", "as png", "png format",
            "save as tiff", "tiff format", "set format", "change format"
        ],
        "set_resolution": [
            "150 dpi", "300 dpi", "600 dpi", "1200 dpi",
            "low resolution", "high resolution", "set resolution", "change resolution",
            "set dpi", "change dpi"
        ],
        "set_layout": [
            "landscape", "portrait", "horizontal", "vertical",
            "set layout", "change layout", "rotate", "set orientation"
        ],
        "undo_action": [
            "undo", "revert", "go back", "oops", "undo selection",
            "revert selection", "go back one step"
        ],

        "set_paper_size": [
            "a4", "a3", "a5", "letter", "legal",
            "set paper size", "change paper size", "use a4", "use letter"
        ],
        "set_copies": [
            "copies", "print copies", "make copies",
            "set copies", "change copies", "number of copies"
        ],
        "set_duplex": [
            "double sided", "duplex", "both sides", "two sided",
            "single sided", "one side", "front only"
        ],
        "set_margins": [
            "no margins", "narrow margins", "default margins", "borderless",
            "set margins", "change margins"
        ],
        "proceed_action": [
            "proceed", "continue", "next step", "go ahead", "move on",
            "confirm selection", "done selecting", "that's all", "ready",
            "start", "begin", "let's go", "go", "proceed to next"
        ],
        "set_feed_count": [
            "feed", "feed documents", "feed pages", "set feed count",
            "number of pages to feed", "documents to feed"
        ],
        "set_pages": [
            "odd pages", "odd pages only", "even pages", "even pages only",
            "custom pages", "custom range", "pages", "only page", "page range",
            "print odd", "print even", "all pages"
        ],
    },

    "friendly_responses": {
        "select_document": "Document {document_number} selected.",
        "select_multiple_documents": "{count} documents selected.",
        "select_document_range": "Documents {start} to {end} selected.",
        "switch_section": "{section}.",
        "next_document": "Next.",
        "previous_document": "Previous.",
        "upload_document": "Upload.",
        "confirm": "Done!",
        "cancel": "Cancelled.",
        "status": "Ready.",
        "repeat_settings": "Settings...",
        "help": "Say: print, scan, or select document.",
        "stop_recording": "Goodbye!",
        "greeting": "Hi! Print or scan?",
        "proceed_action": "Proceeding!",
        "undo_action": "Undone.",
        "set_color_mode": "{color_mode}.",
        "set_quality": "{quality}.",
        "set_format": "{format}.",
        "set_resolution": "{resolution} DPI.",
        "set_layout": "{layout}.",
        "set_paper_size": "{paper_size}.",
        "set_copies": "{copies} copies.",
        "set_duplex": "Double-sided {duplex}.",
        "set_margins": "{margins} margins.",
    },
    "confirmation_words": [
        "yes", "proceed", "go ahead", "okay", "ok",
        "sure", "yep", "yeah", "ye", "confirm", "do it",
        "yup", "yas", "yass", "alright", "affirmative"
    ],
    "print_keywords": [
        "print", "printing", "printout", "printer",
        "print doc", "print file", "print paper", "print document", 
        "i want to print", "need to print", "let's print", "please print",
        # Common Whisper mishearings for "print"
        "prince", "praying", "prints", "printed", "pring", "prink"
    ],
    "scan_keywords": [
        "scan", "scanning", "capture", "scanner",
        "scan doc", "scan file", "capture doc", "capture document", "scan document",
        "i want to scan", "need to scan", "let's scan", "please scan",
        # Common Whisper mishearings for "scan"
        "scam", "stand", "span", "skin", "scan it"
    ],
    "question_words": [
        "what", "can you", "how do", "help", "how to",
        "tell me", "can i", "what is", "can print", "help me", "show me"
    ],
    "settings_keywords": {
        "color_mode": {
            "bw": ["grayscale", "greyscale", "gray scale", "grey scale", "black and white", "black & white", "bw", "mono", "monochrome", "no color"],
            "color": ["color", "full color", "in color", "color mode"]
        },
        "quality": {
            "draft": ["draft", "fast", "economy", "low quality", "quick"],
            "normal": ["normal", "standard", "regular"],
            "high": ["high", "fine", "detailed", "good quality"],
            "professional": ["best", "premium", "photo", "professional", "excellent", "ultra"]
        },
        "format": {
            "pdf": ["pdf", "as pdf", "save as pdf", "export pdf"],
            "jpeg": ["jpeg", "jpg", "as jpeg", "as jpg"],
            "png": ["png", "as png"],
            "tiff": ["tiff", "tif", "as tiff"]
        },
        "layout": {
            "landscape": ["landscape", "horizontal", "wide"],
            "portrait": ["portrait", "vertical", "tall"]
        }
    },
    "ollama": {
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 100,
            "num_ctx": 1024,
            "repeat_penalty": 1.2,
            "stop": ["\n\n", "User:", "Assistant:", "."],
        },
        "timeout": 60,
    },
}



def _load_system_prompt() -> str:
    prompt_path = Path(AI_PROMPT_CONFIG.get("system_prompt_file", SYSTEM_PROMPT_FILE))
    try:
        if prompt_path.exists():
            content = prompt_path.read_text(encoding="utf-8").strip()
            if content:
                logger.info(f"[OK] Loaded system prompt from {prompt_path}")
                return content
            logger.warning(f"[WARN] System prompt file is empty: {prompt_path}")
        else:
            logger.warning(f"[WARN] System prompt file not found: {prompt_path}")
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        logger.error(f"[ERROR] Failed to load system prompt ({prompt_path}): {exc}")
    return DEFAULT_SYSTEM_PROMPT


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_command_config() -> dict:
    config_path = Path(AI_PROMPT_CONFIG.get("command_mappings_file", COMMAND_MAPPINGS_FILE))
    try:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
                if isinstance(payload, dict):
                    logger.info(f"[OK] Loaded command mappings from {config_path}")
                    return payload
                logger.warning(f"[WARN] Command mapping file must contain a JSON object: {config_path}")
        else:
            logger.warning(f"[WARN] Command mapping file not found: {config_path}")
    except json.JSONDecodeError as exc:
        logger.error(f"[ERROR] Invalid JSON in {config_path}: {exc}")
    except Exception as exc:  # pragma: no cover - filesystem edge cases
        logger.error(f"[ERROR] Failed to read {config_path}: {exc}")
    return {}


def _build_command_config() -> dict:
    override_config = _load_command_config()
    return _deep_merge(copy.deepcopy(DEFAULT_COMMAND_CONFIG), override_config if isinstance(override_config, dict) else {})


VOICE_SYSTEM_PROMPT = _load_system_prompt()
COMMAND_CONFIG = _build_command_config()
VOICE_COMMAND_MAPPINGS = COMMAND_CONFIG["voice_commands"]
COMMAND_RESPONSE_MAPPING = COMMAND_CONFIG["friendly_responses"]
CONFIRMATION_WORDS = COMMAND_CONFIG["confirmation_words"]
PRINT_KEYWORDS = COMMAND_CONFIG["print_keywords"]
SCAN_KEYWORDS = COMMAND_CONFIG["scan_keywords"]
QUESTION_WORDS = COMMAND_CONFIG["question_words"]
OLLAMA_QUERY_OPTIONS = COMMAND_CONFIG["ollama"]["options"]
OLLAMA_API_TIMEOUT = COMMAND_CONFIG["ollama"].get("timeout", 60)  # Default voice AI timeout


# ============================================================================
# RESPONSE GENERATION AND FORMATTING
# ============================================================================

class VoicePromptManager:
    """Manager for Voice AI prompts, commands, and query generation"""
    
    @staticmethod
    def get_system_prompt() -> str:
        """
        Get the system prompt for Voice AI
        
        Returns:
            System prompt string for Ollama API
        """
        return VOICE_SYSTEM_PROMPT
    
    @staticmethod
    def get_command_mappings() -> Dict[str, List[str]]:
        """
        Get voice command keyword mappings
        
        Returns:
            Dict mapping command types to keyword lists
        """
        return VOICE_COMMAND_MAPPINGS
    
    @staticmethod
    def get_confirmation_words() -> List[str]:
        """
        Get list of confirmation words
        
        Returns:
            List of words that indicate user confirmation
        """
        return CONFIRMATION_WORDS
    
    @staticmethod
    def build_ollama_query(
        model_name: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Build Ollama API query with optimized parameters
        
        Args:
            model_name: Model to use (e.g., "smollm2:135m")
            messages: Conversation messages with system prompt
            
        Returns:
            JSON dict for Ollama API POST request
            
        Example:
            >>> messages = [
            ...     {"role": "system", "content": SYSTEM_PROMPT},
            ...     {"role": "user", "content": "Print this document"}
            ... ]
            >>> query = VoicePromptManager.build_ollama_query("smollm2:135m", messages)
            >>> response = requests.post("http://localhost:11434/api/chat", json=query, timeout=60)
        """
        return {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": OLLAMA_QUERY_OPTIONS,
        }
    
    @staticmethod
    def format_response(ai_response: str) -> str:
        """
        Format AI response for display.
        
        Transforms:
        - Remove markdown formatting
        - Remove code blocks and URLs
        - Keep full text for chat display
        
        Args:
            ai_response: Raw response from the model
            
        Returns:
            Clean, full-text response for chat display
        """
        import re
        
        # Clean up markdown formatting
        ai_response = ai_response.replace("**", "").replace("*", "")
        
        # Remove code blocks
        ai_response = re.sub(r'```[\s\S]*?```', '', ai_response)
        ai_response = re.sub(r'`[^`]*`', '', ai_response)
        
        # Remove URLs
        ai_response = re.sub(r'https?://\S+', '', ai_response)
        
        # Remove emojis
        ai_response = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF]', '', ai_response)
        
        # Clean up multiple spaces
        ai_response = re.sub(r'\s+', ' ', ai_response).strip()
        
        # Filter out invalid responses
        words = ai_response.split()
        if len(words) < 1 or not any(c.isalpha() for c in ai_response):
            ai_response = "Ready."
        
        return ai_response
    
    @staticmethod
    def format_response_for_tts(ai_response: str) -> str:
        """
        Format AI response for concise, human-like voice playback (TTS).
        
        Transforms:
        - Remove markdown formatting
        - Limit to complete sentences, max 20 words
        - Keep only essential information
        
        Args:
            ai_response: Raw response from the model
            
        Returns:
            Short, voice-friendly response for TTS (no truncation, complete sentences)
        """
        import re
        
        # Start with clean response
        ai_response = VoicePromptManager.format_response(ai_response)
        
        # Take ONLY first sentence (max 1 sentence for brevity)
        sentences = ai_response.split(". ")
        ai_response = sentences[0]
        if ai_response and ai_response[-1] not in ".!?":
            ai_response += "."
        
        # Strict word limit (max 20 words for concise speech)
        words = ai_response.split()
        if len(words) > 20:
            ai_response = " ".join(words[:20])
            if ai_response[-1] not in ".!?":
                ai_response += "."
        
        # Filter out invalid responses
        words = ai_response.split()
        if len(words) < 1 or not any(c.isalpha() for c in ai_response):
            ai_response = "Ready."
        
        return ai_response
    
    @staticmethod
    def get_friendly_command_response(
        command_type: str,
        command_params: Dict[str, Any]
    ) -> str:
        """
        Get friendly response for a voice command with proper parameter substitution
        
        Args:
            command_type: Type of command detected
            command_params: Parameters parsed from user input
            
        Returns:
            User-friendly response string with all placeholders filled
            
        Example:
            >>> params = {"section": "converted", "document_number": 3}
            >>> response = VoicePromptManager.get_friendly_command_response(
            ...     "select_document", params
            ... )
            >>> print(response)
            Got it, document 3.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        template = COMMAND_RESPONSE_MAPPING.get(
            command_type,
            f"Got it! {command_type.replace('_', ' ').title()}."
        )
        
        # Format template with parameters, with better error handling
        if not command_params:
            return template
            
        try:
            # Try to format with provided parameters
            formatted = template.format(**command_params)
            logger.debug(f"[OK] Formatted response for {command_type}: {formatted}")
            return formatted
        except KeyError as e:
            # Log missing key and return best effort response
            missing_key = str(e).strip("'\"")
            logger.warning(f"[WARN] Missing parameter '{missing_key}' for command {command_type}. Params: {command_params}")
            # Return template as-is if formatting fails - it's better than crashing
            return template
        except (IndexError, ValueError) as e:
            logger.warning(f"[WARN] Format error for {command_type}: {e}. Using template as-is.")
            return template
    
    @staticmethod
    def is_confirmation(user_message: str) -> bool:
        """
        Check if user message is a confirmation
        
        Args:
            user_message: User's text input
            
        Returns:
            True if message confirms an action
            
        Example:
            >>> VoicePromptManager.is_confirmation("yes proceed")
            True
            >>> VoicePromptManager.is_confirmation("no thanks")
            False
        """
        user_lower = user_message.lower().strip()
        return any(
            user_lower == word or user_lower.startswith(word + " ")
            for word in CONFIRMATION_WORDS
        )
    
    @staticmethod
    def is_print_command(user_message: str, exclude_questions: bool = True) -> bool:
        """
        Check if user message is a print command
        
        Args:
            user_message: User's text input
            exclude_questions: If True, exclude question-style inputs
            
        Returns:
            True if message appears to be a print command
            
        Example:
            >>> VoicePromptManager.is_print_command("print this document")
            True
            >>> VoicePromptManager.is_print_command("how do I print?")
            False  # if exclude_questions=True
        """
        user_lower = user_message.lower().strip()
        
        # Check for print keywords
        has_print_keyword = any(kw in user_lower for kw in PRINT_KEYWORDS)
        
        if not has_print_keyword:
            return False
        
        # Filter out questions/help requests
        if exclude_questions:
            has_question_word = any(w in user_lower for w in QUESTION_WORDS)
            return not has_question_word
        
        return True
    
    @staticmethod
    def is_scan_command(user_message: str, exclude_questions: bool = True) -> bool:
        """
        Check if user message is a scan command
        
        Args:
            user_message: User's text input
            exclude_questions: If True, exclude question-style inputs
            
        Returns:
            True if message appears to be a scan command
            
        Example:
            >>> VoicePromptManager.is_scan_command("scan a document")
            True
            >>> VoicePromptManager.is_scan_command("can you scan?")
            False  # if exclude_questions=True
        """
        user_lower = user_message.lower().strip()
        
        # Check for scan keywords
        has_scan_keyword = any(kw in user_lower for kw in SCAN_KEYWORDS)
        
        if not has_scan_keyword:
            return False
        
        # Filter out questions/help requests
        if exclude_questions:
            has_question_word = any(w in user_lower for w in QUESTION_WORDS)
            return not has_question_word
        
        return True
    
    @staticmethod
    def is_greeting(user_message: str) -> bool:
        """
        Check if user message is a greeting
        
        Args:
            user_message: User's text input
            
        Returns:
            True if message appears to be a greeting
            
        Example:
            >>> VoicePromptManager.is_greeting("hello")
            True
            >>> VoicePromptManager.is_greeting("hi there")
            True
        """
        user_lower = user_message.lower().strip()
        
        # Get greeting keywords from config
        greeting_keywords = DEFAULT_COMMAND_CONFIG.get("voice_commands", {}).get("greeting", [])
        
        # Check if message matches greeting pattern
        return any(
            user_lower == kw or user_lower.startswith(kw + " ") or kw in user_lower
            for kw in greeting_keywords
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Voice AI prompt manager
    """
    
    # Get system prompt
    prompt = VoicePromptManager.get_system_prompt()
    print(f"System Prompt Length: {len(prompt)} characters\n")
    
    # Build Ollama query
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "How can I help?"}
    ]
    query = VoicePromptManager.build_ollama_query("smollm2:135m", messages)
    print(f"Ollama Query Keys: {query.keys()}\n")
    
    # Format response
    raw_response = "**Opening print interface** now! This will start the entire printing workflow immediately."
    formatted = VoicePromptManager.format_response(raw_response)
    print(f"Original: {raw_response}")
    print(f"Formatted: {formatted}\n")
    
    # Test command responses
    params = {"document_number": 2, "section": "converted"}
    response = VoicePromptManager.get_friendly_command_response("select_document", params)
    print(f"Command Response: {response}\n")
    
    # Test command detection
    print(f"Is 'print this' a print command? {VoicePromptManager.is_print_command('print this')}")
    print(f"Is 'how to print?' a print command? {VoicePromptManager.is_print_command('how to print?')}")
    print(f"Is 'scan please' a scan command? {VoicePromptManager.is_scan_command('scan please')}")
