"""
PrintChakra Backend - Phone Quality Validation
"""

from app.features.phone.quality.validator import QualityValidator

# Alias for backwards compatibility
ImageQualityValidator = QualityValidator

__all__ = ['QualityValidator', 'ImageQualityValidator']
