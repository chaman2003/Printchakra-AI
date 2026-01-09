"""
Chat Flow Routes - API endpoints for voice workflow
"""

import logging
import os
from flask import Blueprint, request, jsonify
from .service import chat_flow_service
from .types import ChatFlowResponse

logger = logging.getLogger(__name__)

chat_flow_bp = Blueprint('chat_flow', __name__, url_prefix='/chat-flow')


@chat_flow_bp.route('/start', methods=['POST'])
def start_session():
    """Start a new voice chat session"""
    try:
        result = chat_flow_service.start_session()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error starting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/end', methods=['POST'])
def end_session():
    """End the voice chat session"""
    try:
        result = chat_flow_service.end_session()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error ending session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/state', methods=['GET'])
def get_state():
    """Get current workflow state"""
    try:
        state = chat_flow_service.get_state()
        return jsonify({"success": True, "state": state}), 200
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error getting state: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/process', methods=['POST'])
def process_text():
    """
    Process transcribed voice text.
    
    Expected JSON body:
    {
        "text": "select document 1 to 5"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        response: ChatFlowResponse = chat_flow_service.process_text(text)
        return jsonify(response.to_dict()), 200
        
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error processing text: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/update-documents', methods=['POST'])
def update_documents():
    """
    Update selected documents from frontend.
    
    Expected JSON body:
    {
        "documents": ["doc1.pdf", "doc2.pdf"]
    }
    """
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        
        chat_flow_service.update_selected_documents(documents)
        return jsonify({"success": True, "message": f"Updated {len(documents)} documents"}), 200
        
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error updating documents: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/job-completed', methods=['POST'])
def job_completed():
    """
    Notify that a print/scan job has completed.
    
    Expected JSON body:
    {
        "success": true,
        "message": "optional error message"
    }
    """
    try:
        data = request.get_json()
        success = data.get('success', True)
        message = data.get('message', '')
        
        response = chat_flow_service.job_completed(success, message)
        return jsonify(response.to_dict()), 200
        
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error handling job completion: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/process-audio', methods=['POST'])
def process_audio():
    """
    Process audio from frontend - transcribe and run through chat flow.
    
    Expects multipart form data with 'audio' file.
    Returns transcribed text and chat flow response.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Import transcription from voice module
        try:
            from app.modules.voice import transcribe_audio_file
        except ImportError:
            # Fallback to app.py level
            from app import transcribe_audio_data as transcribe_audio_file
        
        # Save temp file for transcription
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Transcribe
            transcription_result = transcribe_audio_file(tmp_path)
            
            if not transcription_result or not transcription_result.get('text'):
                return jsonify({
                    "success": False,
                    "error": "Transcription failed or empty"
                }), 200
            
            user_text = transcription_result['text'].strip()
            
            # Process through chat flow
            chat_response = chat_flow_service.process_text(user_text)
            
            return jsonify({
                "success": True,
                "user_text": user_text,
                "chat_flow_response": chat_response.to_dict()
            }), 200
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error processing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@chat_flow_bp.route('/speak', methods=['POST'])
def speak():
    """
    Generate speech from text using TTS.
    
    Expected JSON body:
    {
        "text": "Hello, I'm your assistant"
    }
    
    Returns audio file or success status.
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Import TTS from voice module
        try:
            from app.modules.voice import speak_text
            audio_path = speak_text(text)
            
            if audio_path and os.path.exists(audio_path):
                from flask import send_file
                return send_file(audio_path, mimetype='audio/mp3')
            else:
                # TTS might have played directly
                return jsonify({"success": True, "message": "Speech generated"}), 200
                
        except ImportError:
            # No TTS module, use browser TTS
            return jsonify({
                "success": True,
                "use_browser_tts": True,
                "text": text
            }), 200
        
    except Exception as e:
        logger.error(f"[CHAT_FLOW] Error with TTS: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
