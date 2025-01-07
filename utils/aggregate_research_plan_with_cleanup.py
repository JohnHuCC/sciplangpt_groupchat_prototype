def aggregate_research_plan_with_cleanup(chat_history):
    # Define the sections and initialize them
    sections = {
        "Research Question": "",
        "Research Background": "",
        "Experiment Plan": "",
        "Research Output": "",
        "Research Summary": ""
    }

    # Iterate over the chat history to populate the sections
    for message in chat_history:
        speaker = message.get('name', "")
        content = message.get('content', "")

        # Clean up "Relevant Literature" from the Research Question content
        if speaker == "ResearchQuestionGenerator":
            content = remove_relevant_literature(content)
            sections["Research Question"] += content + "\n\n"
        elif speaker == "ResearchBackgroundGenerator":
            sections["Research Background"] += content + "\n\n"
        elif speaker == "ExperimentPlanGenerator":
            sections["Experiment Plan"] += content + "\n\n"
        elif speaker == "ResearchOutputGenerator":
            sections["Research Output"] += content + "\n\n"
        elif speaker == "ResearchSummarizer":
            sections["Research Summary"] += content + "\n\n"

    # Format the aggregated research plan
    research_plan = ""
    for section, content in sections.items():
        research_plan += f"**{section}**:\n{content.strip()}\n\n"

    return research_plan.strip()


def remove_relevant_literature(content):
    """Remove the 'Relevant Literature' section from the content."""
    # Identify the start and end of the 'Relevant Literature' section
    start_marker = "Relevant Literature Content:"
    end_marker = "Please design the experiment according to the following framework:"
    if start_marker in content:
        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index)
        if end_index != -1:
            # Remove the 'Relevant Literature' section
            content = content[:start_index] + content[end_index + len(end_marker):]
    return content.strip()
