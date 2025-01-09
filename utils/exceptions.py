from fastapi import HTTPException

class ResearchPlanException(HTTPException):
    """Base exception for research plan related errors"""
    pass

class ValidationException(ResearchPlanException):
    """Validation related errors"""
    pass
