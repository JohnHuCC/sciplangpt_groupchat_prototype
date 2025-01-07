def generate_research_question(area_of_interest: str):
    # Define the research question generator prompt for Autogen
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