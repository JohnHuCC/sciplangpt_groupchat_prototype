from typing import Dict, Any
from fastapi import HTTPException

async def extract_last_agent_response(
    chat_result: Dict[str, Any], 
    termination_phrase: str = "end chat", 
    min_word_count: int = 20
) -> str:
    """
    非同步提取代理的最近有意義回應
    
    Args:
        chat_result: Chat history result object
        termination_phrase: Phrase indicating chat termination
        min_word_count: Minimum word count for valid response
    Returns:
        Latest meaningful agent response
    Raises:
        HTTPException: If extraction fails
    """
    try:
        if not chat_result or 'chat_history' not in chat_result:
            raise HTTPException(
                status_code=400,
                detail="Invalid chat result format"
            )

        for message in reversed(chat_result.chat_history):
            if len(message["content"].split()) >= min_word_count:
                return message["content"]
        return ""
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting agent response: {str(e)}"
        )