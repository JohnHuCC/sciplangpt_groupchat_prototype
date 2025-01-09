from typing import List, Dict
from fastapi import HTTPException

async def form_prompt_with_context(
    question: str, 
    relevant_docs: List[Dict[str, str]]
) -> str:
    """
    非同步組合研究問題和相關文獻形成實驗計畫提示
    
    Args:
        question: Research question
        relevant_docs: List of relevant documents
    Returns:
        Formatted prompt string
    Raises:
        HTTPException: If prompt formation fails
    """
    try:
        if not question:
            raise HTTPException(
                status_code=400,
                detail="Research question cannot be empty"
            )

        # 組合相關文獻內容
        if relevant_docs:
            context = "\n\n".join([
                f"- {doc['text']} (Source: {doc['source']})" 
                for doc in relevant_docs
            ])
        else:
            context = "No relevant literature was retrieved for this query."

        # 形成提示
        prompt = f"""Based on the following research question and relevant literature, design a detailed experimental plan:

Research Question:
{question}

Relevant Literature Content:
{context}

Please design the experiment according to the following framework:

"""
        print("\nGenerated Prompt:")
        print(prompt)
        return prompt
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error forming prompt with context: {str(e)}"
        )