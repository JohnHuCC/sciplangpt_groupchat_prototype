def extract_last_agent_response(chat_result, termination_phrase="end chat", min_word_count=20):
    """
    Extracts the most recent meaningful response from the agent in a back-and-forth conversation.
    
    Args:
        chat_result: The result object containing the chat history.
        termination_phrase: The phrase indicating termination (default is "end chat").
    
    Returns:
        The content of the latest meaningful agent response, or an empty string if none is found.
    """
    # print(chat_result.chat_history)
    for message in reversed(chat_result.chat_history):
        # Check for agent's response and exclude termination messages
        if len(message["content"].split()) >= min_word_count:
            # print("this is the message got saved \n\n" + message["content"])
            return message["content"]
    return ""