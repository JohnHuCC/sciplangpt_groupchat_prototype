from typing import List, Dict, Any
from fastapi import HTTPException

async def aggregate_research_plan_with_cleanup(chat_history: List[Dict[str, Any]]) -> str:
    """
    非同步聚合研究計畫並清理內容
    
    Args:
        chat_history: List of chat messages
    Returns:
        Formatted research plan
    Raises:
        HTTPException: If chat history processing fails
    """
    try:
        sections = {
            "Research Question": "",
            "Research Background": "",
            "Experiment Plan": "",
            "Research Output": "",
            "Research Summary": ""
        }

        for message in chat_history:
            speaker = message.get('name', "")
            content = message.get('content', "")

            if speaker == "ResearchQuestionGenerator":
                content = await remove_relevant_literature(content)
                sections["Research Question"] += content + "\n\n"
            elif speaker == "ResearchBackgroundGenerator":
                sections["Research Background"] += content + "\n\n"
            elif speaker == "ExperimentPlanGenerator":
                sections["Experiment Plan"] += content + "\n\n"
            elif speaker == "ResearchOutputGenerator":
                sections["Research Output"] += content + "\n\n"
            elif speaker == "ResearchSummarizer":
                sections["Research Summary"] += content + "\n\n"

        research_plan = ""
        for section, content in sections.items():
            research_plan += f"**{section}**:\n{content.strip()}\n\n"

        return research_plan.strip()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error aggregating research plan: {str(e)}"
        )

async def remove_relevant_literature(content: str) -> str:
    """
    非同步移除內容中的相關文獻部分
    
    Args:
        content: Original content string
    Returns:
        Cleaned content string
    """
    try:
        start_marker = "Relevant Literature Content:"
        end_marker = "Please design the experiment according to the following framework:"
        
        if start_marker in content:
            start_index = content.find(start_marker)
            end_index = content.find(end_marker, start_index)
            if end_index != -1:
                content = content[:start_index] + content[end_index + len(end_marker):]
                
        return content.strip()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error removing relevant literature: {str(e)}"
        )