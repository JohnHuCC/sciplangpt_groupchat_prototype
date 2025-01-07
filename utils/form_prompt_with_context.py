
def form_prompt_with_context(question: str, relevant_docs: list[dict[str, str]]) -> str:
    """Combine the research question and relevant literature to form the experimental plan prompt."""
    # Combine the relevant literature content
    if relevant_docs:
        context = "\n\n".join([f"- {doc['text']} (Source: {doc['source']})" for doc in relevant_docs])
    else:
        context = "No relevant literature was retrieved for this query."

    # Form the prompt
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