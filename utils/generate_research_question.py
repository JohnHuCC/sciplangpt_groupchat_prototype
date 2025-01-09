from typing import Optional
from fastapi import HTTPException

async def generate_research_question(area_of_interest: str) -> str:
    """
    非同步生成研究問題提示
    
    Args:
        area_of_interest: Research area of interest
    Returns:
        Research question prompt
    Raises:
        HTTPException: If generation fails
    """
    try:
        if not area_of_interest:
            raise HTTPException(
                status_code=400,
                detail="Area of interest cannot be empty"
            )

        research_question_prompt = f"""
        **Instructions:**
        Based on the following research area, generate a well-defined and specific research question that addresses a significant gap or problem within the field.
        
        **Research Area:**
        {area_of_interest}
        
        **Requirements:**
        1. The research question should be concise and focused.
        2. It should highlight the significance and relevance of the problem.
        3. The question must be answerable through empirical research.
        """
        return research_question_prompt
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating research question: {str(e)}"
        )