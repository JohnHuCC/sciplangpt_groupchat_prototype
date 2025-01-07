import os
from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
import dotenv

dotenv.load_dotenv()

# import necessary function
from utils.generate_research_question import generate_research_question
from utils.query_knowledge import query_knowledge
from utils.form_prompt_with_context import form_prompt_with_context
from utils.aggregate_research_plan_with_cleanup import aggregate_research_plan_with_cleanup

# Configure the LLM
llm_config = {
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

refine_llm_config = {
    "model": "gpt-4o",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

# Initial Group chat agents
research_question_generator = ConversableAgent(
    name="ResearchQuestionGenerator",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
        "You are an expert in generating research questions. "
        "Based on the user's area of interest, generate insightful and impactful research questions. "
        "Engage with the user to refine these questions until they are satisfied."
    ),
    is_termination_msg=lambda msg: "end chat" in msg["content"].lower(),
)

research_background_generator = ConversableAgent(
    name="ResearchBackgroundGenerator",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
       "You are an expert in generating research background and goals. "
        "For each statement you make, explicitly cite the source of the information directly after the sentence, "
        "following this format: (Source: 'filename'). "
        "Ensure that the citations are accurate and correspond to the relevant sections of the referenced document. "
        "Clearly state the purpose of the research and provide a concise explanation of its background. "
    #    "Define specific research questions. "
    #    "Propose testable hypotheses. "
    #    "Highlight the innovative aspects of the research. "
    ),
    is_termination_msg=lambda msg: "end chat" in msg["content"].lower(),
)

experiment_plan_generator = ConversableAgent(
    name="ExperimentPlanGenerator",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
        "You are an expert in generating experiment plan for research. "
        "Provide a detailed description of experimental methods. "
        "List necessary equipment and materials. "
        "Explain the data collection process. "
        "Include quality control measures. "
        "Provide solutions for technical challenges. "
        "If any cited source information is used, retain the original citation format "
        "(e.g., '(Source: \"filename\")') within your response. "
    ),
    is_termination_msg=lambda msg: "end chat" in msg["content"].lower(),
)

research_output_generator = ConversableAgent(
    name="ResearchOutputGenerator",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
        "You are an expert in generating expected research output. "
        "Describe the expected experimental results. "
        "Explain the significance and impact of the results. "
        "Discuss the potential applications of the results. "
        "If any cited source information is used, retain the original citation format "
        "(e.g., '(Source: \"filename\")') within your response. "
    ),
    is_termination_msg=lambda msg: "end chat" in msg["content"].lower(),
)

research_summarizer = ConversableAgent(
    name="ResearchSummarizer",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message=(
        "You are an expert in summarizing the entire research plan. "
        "Provide an overview of the research background, emphasizing the urgency and importance of the problem. "
        "Clearly articulate the purpose and objectives of the research. "
        "Briefly describe the research methods and technical approach. "
        "Explain the expected outcomes and their contributions and impact on the field. "
        "If any cited source information is used, retain the original citation format "
        "(e.g., '(Source: \"filename\")') within your response. "
    ),
    is_termination_msg=lambda msg: "end chat" in msg["content"].lower(),
)

# add description for agent
research_question_generator.description = "Generating Comprehensive Research Plan"
research_background_generator.description = "Generate the research background information based on research plan"
experiment_plan_generator.description = "design the experiment for the research "
research_output_generator.description = "define the output of the research given"
research_summarizer.description = "Put the research plan, research background, experiment, and output all together and summarize"

# Initialize GroupChat
group_chat = GroupChat(
    agents=[
        research_question_generator,
        research_background_generator,
        experiment_plan_generator,
        research_output_generator,
        research_summarizer,
        # research_combiner
    ],
    messages=[],
    max_round=5,
    speaker_selection_method="round_robin",
    send_introductions=True,
)

# Create GroupChatManager
group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# 
helper = ConversableAgent(
    "helper",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Human Proxy Agent
human_proxy = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    code_execution_config={"use_docker": False},
)

refine_agent  = ConversableAgent(
    name="refineAgent",
    llm_config=refine_llm_config,
    human_input_mode="NEVER",
    system_message=(
        "You are an assistant dedicated to refining the research plan based on user feedback. "
        "You will receive the current research plan and the user's requested changes or feedback. "
        "Apply the changes to the research plan and return the updated version. "
        "Ensure that the refined research plan maintains coherence and completeness."
    ),
    is_termination_msg=lambda msg: "end chat" in msg["content"].lower(),
)


def main():
    # Step 1: User inputs their area of interest
    area_of_interest = input("Enter your area of interest for the research: ")
    formatted_prompt = generate_research_question(area_of_interest)

    research_question_conversation = helper.initiate_chat(
        research_question_generator, 
        message=formatted_prompt,
        max_turns=1
    )

    research_question = research_question_conversation.chat_history[-1]["content"]

    # query embedding
    query_result = query_knowledge(research_question, 5)
    print(query_result)
    research_plan_prompt = form_prompt_with_context(research_question, query_result)
    
    # Step 2: Initiate the research plan generation
    print("\nGenerating initial research plan...\n")
    initial_chat = research_question_generator.initiate_chat(
        group_chat_manager,
        message=research_plan_prompt,
        summary_method="last_msg",
    )

    complete_research_plan = aggregate_research_plan_with_cleanup(initial_chat.chat_history)
    print("Complete Research Plan:\n")
    print(complete_research_plan)

    # prompt if user need refinement
    # Step 4: Prompt user if they want to refine the research plan
    while True:
        user_choice = input("\nDo you want to refine the research plan? (yes/no): ").strip().lower()
        if user_choice in ['no', 'n']:
            print("\nFinal Research Plan:")
            print(complete_research_plan)
            break
        elif user_choice in ['yes', 'y']:
            print("\nCurrent Research Plan:")
            print(complete_research_plan)
            user_feedback = input("\nEnter your changes or feedback for the research plan: ").strip()
            
            # Combine current plan and user feedback
            refinement_message = f"Current Research Plan:\n{complete_research_plan}\n\nUser Feedback/Changes:\n{user_feedback}"
            
            # Initiate refinement chat
            refinement_chat = human_proxy.initiate_chat(
                refine_agent,
                message=refinement_message,
                max_turns=2  # Adjust as needed
            )
            
            # Update the final_research_plan with the refined version
            updated_plan = refinement_chat.chat_history[-1]["content"]
            if updated_plan:
                complete_research_plan = updated_plan
                print("\nRefined Research Plan:")
                print(complete_research_plan)
            else:
                print("\nNo changes were made to the research plan.")
            break
        else:
            print("Please enter a valid response (yes/no).")

        
if __name__ == "__main__":
    main()
