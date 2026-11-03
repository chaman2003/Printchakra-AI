"""
PrintChakra Backend - Pipeline Info Routes

Provides lightweight pipeline metadata endpoints used by frontend checks.
"""

from flask import jsonify

from app.core.config import get_data_dirs
from app.features.dashboard.routes import dashboard_bp


def _build_pipeline_info() -> dict:
    """Build pipeline metadata with safe fallback if optional modules are unavailable."""
    try:
        from app.modules.pipeline.enhanced import EnhancedDocumentPipeline

        processed_dir = get_data_dirs()["PROCESSED_DIR"]
        pipeline = EnhancedDocumentPipeline(storage_dir=processed_dir)
        return {
            "available": True,
            "pipeline": "enhanced",
            "version": "1.0",
            "config": pipeline.get_pipeline_info(),
        }
    except Exception as exc:
        return {
            "available": False,
            "pipeline": "enhanced",
            "version": "1.0",
            "error": str(exc),
            "config": {},
        }


@dashboard_bp.route("/pipeline/info", methods=["GET"])
def pipeline_info():
    """Frontend-compatible pipeline info endpoint."""
    info = _build_pipeline_info()
    return jsonify({"success": bool(info.get("available")), **info}), 200 if info.get("available") else 503


@dashboard_bp.route("/pipeline/config", methods=["GET"])
def pipeline_config():
    """Alias endpoint for clients expecting /pipeline/config."""
    info = _build_pipeline_info()
    return jsonify({"success": bool(info.get("available")), **info}), 200 if info.get("available") else 503
