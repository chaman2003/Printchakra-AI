"""
Chat Flow Commands - Voice command parsing and handling
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ChatFlowCommands:
    """
    Parses and interprets voice commands for the print/scan workflow.
    
    Supported commands:
    - Mode: "print", "scan"
    - Confirmation: "yes", "proceed", "okay", "confirm"
    - Cancel: "no", "cancel", "stop", "nevermind"
    - Document selection: "select document 1", "select documents 1 to 5", "select all"
    - Settings: "landscape", "portrait", "2 copies", "color", "grayscale", "black and white"
    - Pages: "all pages", "odd pages", "even pages", "pages 1 to 3"
    - Navigation: "back", "next", "done"
    - Exit: "bye printchakra", "goodbye", "exit"
    """
    
    # Number words to digits
    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    }
    
    @classmethod
    def parse_number(cls, text: str) -> Optional[int]:
        """Convert text number to integer"""
        text = text.lower().strip()
        if text in cls.NUMBER_WORDS:
            return cls.NUMBER_WORDS[text]
        try:
            return int(text)
        except ValueError:
            return None
    
    @classmethod
    def parse_command(cls, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parse voice text into command and parameters.
        
        Returns:
            Tuple of (command_name, params_dict)
        """
        text_lower = text.lower().strip()
        
        # Remove filler words
        text_lower = re.sub(r'\b(please|could you|can you|i want to|i\'d like to)\b', '', text_lower).strip()
        
        # === MODE SWITCH (talk/work) - check early ===
        if cls._is_talk_mode(text_lower):
            return "switch_to_talk", {}
        
        if cls._is_work_mode(text_lower):
            return "switch_to_work", {}
        
        # === GREETINGS (check first) ===
        if cls._is_greeting(text_lower):
            return "greeting", {}
        
        # === MODE COMMANDS ===
        if cls._is_print_command(text_lower):
            return "start_print", {}
        
        if cls._is_scan_command(text_lower):
            return "start_scan", {}
        
        # === CONFIRMATION ===
        if cls._is_confirmation(text_lower):
            return "confirm", {}
        
        if cls._is_cancellation(text_lower):
            return "cancel", {}
        
        # === EXIT ===
        if cls._is_exit_command(text_lower):
            return "exit_session", {}
        
        # === SWITCH SECTION (converted/upload) ===
        switch_cmd = cls._parse_switch_section(text_lower)
        if switch_cmd:
            return switch_cmd
        
        # === DOCUMENT SELECTION ===
        doc_cmd = cls._parse_document_selection(text_lower)
        if doc_cmd:
            return doc_cmd
        
        # === SETTINGS ===
        settings_cmd = cls._parse_settings(text_lower)
        if settings_cmd:
            return settings_cmd
        
        # === NAVIGATION ===
        if cls._is_back(text_lower):
            return "go_back", {}
        
        if cls._is_proceed(text_lower):
            return "proceed", {}
        
        # No recognized command
        return None, {}
    
    @classmethod
    def _is_greeting(cls, text: str) -> bool:
        """Check if text is a greeting"""
        greeting_words = ["hello", "hi", "hey", "greetings", "howdy", "good morning", 
                         "good afternoon", "good evening", "hola", "what's up", "whats up"]
        # Exact match or starts with greeting
        return text in greeting_words or any(text.startswith(g) for g in greeting_words)
    
    @classmethod
    def _is_talk_mode(cls, text: str) -> bool:
        """Check if user wants to switch to talk/chat mode"""
        talk_patterns = [
            r"let'?s?\s*talk", r"lets\s*talk", r"let\s*us\s*talk",
            r"chat\s*mode", r"talk\s*mode", r"conversation\s*mode",
            r"just\s*chat", r"i\s*want\s*to\s*talk", r"can\s*we\s*talk",
            r"switch\s*to\s*talk", r"switch\s*to\s*chat",
        ]
        return any(re.search(p, text) for p in talk_patterns)
    
    @classmethod
    def _is_work_mode(cls, text: str) -> bool:
        """Check if user wants to switch to work mode"""
        work_patterns = [
            r"let'?s?\s*work", r"lets\s*work", r"let\s*us\s*work",
            r"work\s*mode", r"back\s*to\s*work", r"start\s*working",
            r"switch\s*to\s*work", r"print\s*mode", r"scan\s*mode",
        ]
        return any(re.search(p, text) for p in work_patterns)
    
    @classmethod
    def _is_print_command(cls, text: str) -> bool:
        """Check if text is a print command"""
        print_patterns = [
            r'^print$', r'^print\s*document', r'^print\s*file',
            r'^i\s*want\s*to\s*print', r'^printing$',
            r'^start\s*print', r'^begin\s*print',
        ]
        return any(re.search(p, text) for p in print_patterns)
    
    @classmethod
    def _is_scan_command(cls, text: str) -> bool:
        """Check if text is a scan command"""
        scan_patterns = [
            r'^scan$', r'^scan\s*document', r'^scan\s*file',
            r'^i\s*want\s*to\s*scan', r'^scanning$',
            r'^start\s*scan', r'^begin\s*scan',
        ]
        return any(re.search(p, text) for p in scan_patterns)
    
    @classmethod
    def _is_confirmation(cls, text: str) -> bool:
        """Check if text is confirmation"""
        confirm_words = ["yes", "yeah", "yep", "yup", "okay", "ok", "sure", 
                        "confirm", "do it", "go ahead", "affirmative", "correct"]
        return text in confirm_words or any(text.startswith(w + " ") for w in confirm_words)
    
    @classmethod
    def _is_cancellation(cls, text: str) -> bool:
        """Check if text is cancellation"""
        cancel_words = ["no", "nope", "cancel", "stop", "abort", "nevermind", 
                       "never mind", "forget it", "don't", "quit"]
        return text in cancel_words or any(text.startswith(w + " ") for w in cancel_words)
    
    @classmethod
    def _is_exit_command(cls, text: str) -> bool:
        """Check if text is exit command"""
        exit_patterns = [
            r'bye\s*print\s*chakra', r'goodbye', r'^exit$', r'^quit$',
            r'end\s*session', r'close\s*voice', r'stop\s*listening',
        ]
        return any(re.search(p, text) for p in exit_patterns)
    
    @classmethod
    def _is_proceed(cls, text: str) -> bool:
        """Check if text is proceed/next command"""
        # Exact matches
        proceed_exact = ["proceed", "next", "continue", "done", "finish", 
                        "move on", "let's go", "lets go", "execute", "run", "start now"]
        if text in proceed_exact:
            return True
        
        # "go" alone or "go ahead" but not "go back" or "go to"
        if text == "go" or text == "go ahead":
            return True
        
        # Starts with specific words (but be careful)
        if text.startswith("proceed") or text.startswith("continue") or text.startswith("execute"):
            return True
            
        return False
    
    @classmethod
    def _is_back(cls, text: str) -> bool:
        """Check if text is back command"""
        back_words = ["back", "previous", "go back", "return", "undo"]
        return text in back_words or any(text.startswith(w) for w in back_words)
    
    @classmethod
    def _parse_switch_section(cls, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse switch section commands (converted files vs upload)"""
        # Switch to converted files
        if re.search(r'(?:switch|go|show|open)\s*(?:to\s*)?(?:converted|converted\s*files?|converted\s*documents?)', text):
            return "switch_section", {"section": "converted"}
        
        # Switch to upload/local files
        if re.search(r'(?:switch|go|show|open)\s*(?:to\s*)?(?:upload|local|local\s*files?|upload\s*files?)', text):
            return "switch_section", {"section": "upload"}
        
        # Switch to current/original
        if re.search(r'(?:switch|go|show|open)\s*(?:to\s*)?(?:current|original)', text):
            return "switch_section", {"section": "current"}
        
        return None
    
    @classmethod
    def _parse_document_selection(cls, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse document selection commands"""
        
        # "select all" / "all documents"
        if re.search(r'select\s*all|all\s*documents', text):
            return "select_all_documents", {}
        
        # "deselect all" / "clear selection" / "clear all"
        if re.search(r'(?:deselect|unselect|clear)\s*(?:all|everything|selection)', text):
            return "deselect_all", {}
        
        # "select document 1 to 5" / "select documents 1 through 5" / "select from 1 to 5"
        range_match = re.search(
            r'(?:select\s*)?(?:document|documents|doc|docs|from)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:to|through|-)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
            text
        )
        if range_match:
            start = cls.parse_number(range_match.group(1))
            end = cls.parse_number(range_match.group(2))
            if start and end:
                return "select_document_range", {"start": start, "end": end}
        
        # "deselect documents 1, 3, 5" / "deselect 1 and 3 and 5"
        deselect_list_match = re.search(
            r'(?:deselect|unselect|remove)\s*(?:document|documents|doc|docs)?\s*([\d\s,and]+)',
            text
        )
        if deselect_list_match:
            numbers_text = deselect_list_match.group(1)
            numbers = re.findall(r'\d+', numbers_text)
            if numbers:
                indices = [int(n) for n in numbers]
                if len(indices) == 1:
                    return "deselect_document", {"index": indices[0]}
                return "deselect_documents", {"indices": indices}
        
        # "select documents 1, 3, 5" / "select 1 and 3 and 5" / "select document 1 and 3"
        list_match = re.search(
            r'select\s*(?:document|documents|doc|docs)?\s*([\d\s,and]+)',
            text
        )
        if list_match:
            numbers_text = list_match.group(1)
            numbers = re.findall(r'\d+', numbers_text)
            if numbers:
                indices = [int(n) for n in numbers]
                if len(indices) == 1:
                    return "select_document", {"index": indices[0]}
                return "select_documents", {"indices": indices}
        
        # "select document 3" / "select the third document" / just "select 3"
        single_match = re.search(
            r'select\s*(?:the\s*)?(?:document|doc)?\s*(\d+|first|second|third|fourth|fifth|one|two|three|four|five|1st|2nd|3rd|4th|5th)',
            text
        )
        if single_match:
            num = cls.parse_number(single_match.group(1))
            if num:
                return "select_document", {"index": num}
        
        # "deselect document 3" / "unselect 3"
        deselect_match = re.search(
            r'(?:deselect|unselect|remove)\s*(?:document|doc)?\s*(\d+|first|second|third|fourth|fifth)',
            text
        )
        if deselect_match:
            num = cls.parse_number(deselect_match.group(1))
            if num:
                return "deselect_document", {"index": num}
        
        return None
    
    @classmethod
    def _parse_settings(cls, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse print/scan settings commands"""
        
        # Orientation: "landscape" / "portrait"
        if re.search(r'\blandscape\b', text):
            return "set_orientation", {"orientation": "landscape"}
        if re.search(r'\bportrait\b', text):
            return "set_orientation", {"orientation": "portrait"}
        
        # Color mode
        if re.search(r'\bcolor\b(?!\s*mode)', text) or re.search(r'\bfull\s*color\b', text):
            return "set_color_mode", {"color_mode": "color"}
        if re.search(r'\b(?:grayscale|grey\s*scale|gray\s*scale)\b', text):
            return "set_color_mode", {"color_mode": "grayscale"}
        if re.search(r'\b(?:black\s*(?:and|&)?\s*white|b\s*w|bw|monochrome)\b', text):
            return "set_color_mode", {"color_mode": "bw"}
        
        # Copies: "3 copies" / "three copies"
        copies_match = re.search(
            r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*cop(?:y|ies)',
            text
        )
        if copies_match:
            num = cls.parse_number(copies_match.group(1))
            if num:
                return "set_copies", {"copies": num}
        
        # Paper size
        paper_sizes = {
            r'\ba4\b': "A4",
            r'\bletter\b': "Letter",
            r'\blegal\b': "Legal",
            r'\ba3\b': "A3",
            r'\ba5\b': "A5",
        }
        for pattern, size in paper_sizes.items():
            if re.search(pattern, text):
                return "set_paper_size", {"paper_size": size}
        
        # Duplex / double-sided
        if re.search(r'\b(?:duplex|double\s*sided|two\s*sided|both\s*sides)\b', text):
            return "set_duplex", {"duplex": True}
        if re.search(r'\b(?:single\s*sided|one\s*sided|one\s*side)\b', text):
            return "set_duplex", {"duplex": False}
        
        # Pages: "all pages", "odd pages", "even pages"
        if re.search(r'\ball\s*pages\b', text):
            return "set_pages", {"pages": "all"}
        if re.search(r'\bodd\s*pages\b', text):
            return "set_pages", {"pages": "odd"}
        if re.search(r'\beven\s*pages\b', text):
            return "set_pages", {"pages": "even"}
        
        # Custom page range: "pages 1 to 3" / "page 1-5"
        page_range_match = re.search(
            r'pages?\s*(\d+)\s*(?:to|-|through)\s*(\d+)',
            text
        )
        if page_range_match:
            start = int(page_range_match.group(1))
            end = int(page_range_match.group(2))
            return "set_pages", {"pages": "custom", "custom_range": f"{start}-{end}"}
        
        # Quality
        if re.search(r'\b(?:draft|fast|quick)\b', text):
            return "set_quality", {"quality": "draft"}
        if re.search(r'\b(?:high|best|fine)\s*quality\b', text):
            return "set_quality", {"quality": "high"}
        
        # Scan DPI
        dpi_match = re.search(r'(\d+)\s*dpi', text)
        if dpi_match:
            dpi = int(dpi_match.group(1))
            return "set_scan_dpi", {"dpi": dpi}
        
        return None
