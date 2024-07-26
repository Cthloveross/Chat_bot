import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
import json
import datetime
import re

# Load environment variables
load_dotenv()

# Initialize the models using OpenAI API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
conversational_llm = OpenAI(api_key=openai_api_key)  # For conversation
extraction_llm = OpenAI(api_key=openai_api_key)      # For extraction

def extract_information_with_llm(user_input, current_info):
    # Define the extraction instructions for the LLM
    extraction_prompt_template = """
    Please analyze the user_input information and extract the following information. Finally you need to update the information to the JSON structure.

    User Input: {user_input}

    If the information is present, extract:

    - Personal Information: (Gender, Nationality, Undergraduate Country, Undergraduate School, Undergraduate Major, Intended Graduate School Country, Intended Degree, Intended Major, Second Undergraduate Major, Graduation Year)
    - Standard Grades: (GPA, GPA Total, Rank, Rank Total, Language Test Type, Language Test Score, GRE, GMAT)
    - Professional Experiences: (Employer, Company Size, Title, Start Date, End Date, Job Description)
    - Academic Experience: (Category, Project Name, Title, Start Date, End Date, Outcome)
    - Honors: (Category, Project Name, Name, Pool, Earn Date, Description)
    - Activities: (Name, Organization, Title, Start Date, End Date, Details)
    - Other Information: (Personal Website, Other Info)
    - Applied Programs: (Program Level, Program, School, Result)

    Current Information: {current_info}

    Extracted Information (JSON):
    """

    formatted_info = json.dumps(current_info, indent=2)
    prompt = extraction_prompt_template.format(user_input=user_input, current_info=formatted_info)
    
    response = extraction_llm.predict(prompt)
    print(f"Extraction LLM Response: {response}")

    try:
        # Attempt to find and parse the JSON response containing the extracted information
        extracted_info_match = re.search(r'{[\s\S]*}', response)
        if extracted_info_match:
            extracted_json_str = extracted_info_match.group()

            extracted_info = json.loads(extracted_json_str)
            # print(f"Extracted Info: {json.dumps(extracted_info, indent=2)}")
            current_info.update(extracted_info)
        else:
            print("No JSON data found in the response.")
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"Failed to extract or parse information from the response: {e}")
    print("Current User Info:", json.dumps(current_info, indent=2))
    return current_info


# Function to chat with the bot
def chat_with_bot():
    
    # Define the prompt template for the conversational agent
    conversational_prompt_template = """
    You are a professional study abroad consultant, please take a deep breath, I need you. 
    Please guide users to tell more personal information about studying abroad and help them understand more knowledge about studying abroad! 
    Please note that your main purpose is to help users better understand their study abroad status.
    Please guide users to provide the following information in a gentle and friendly conversation:
    {user_info}


    Please use the conversation history to provide more accurate and personalized responses.
    {conversation_history}

    User: {user_input}
    Agent:
    """
    chat_history = []
    user_info = {
    "personalInfo": {
        "gender": None,
        "nationality": None,
        "undergradCountry": None,
        "undergradSchool": None,
        "undergradMajor": None,
        "intendGradSchoolCountry": None,
        "intendedDegree": None,
        "IntendedMajor": None,
        "secondUndergradMajor": None,
        "GraduationYear": None
    },
    "standardGrades": {
        "gpa": None,
        "gpaTotal": None,
        "rank": None,
        "rankTotal": None,
        "languageTestType": None,
        "languageTestScore": None,
        "gre": None,
        "gmat": None
    },
    "professionalExperiences": [
        {
            "employer": None,
            "companySize": None,
            "title": None,
            "startDate": None,
            "endDate": None,
            "jd": None  # Job description
        }
    ],
    "academicExperience": [
        {
            "category": None,
            "projectName": None,
            "title": None,
            "startDate": None,
            "endDate": None,
            "outcome": None
        }
    ],
    "honors": [
        {
            "category": None,
            "projectName": None,
            "name": None,
            "pool": None,
            "earnDate": None,
            "description": None
        }
    ],
    "activities": [
        {
            "name": None,
            "organization": None,
            "title": None,
            "startDate": None,
            "endDate": None,
            "details": None
        }
    ],
    "other": {
        "personalWebsite": None,
        "otherInfo": None
    },
    "appliedPrograms": [
        {
            "programLevel": None,
            "program": None,
            "school": None,
            "result": None
        }
    ]
}


    print("Start chatting with the bot (type 'exit' to stop):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        try:
            # Format conversation history for the prompt
            conversation_history = "\n".join([f"User: {entry['user']}\nAgent: {entry['bot']}" for entry in chat_history])
            formatted_input = conversational_prompt_template.format(conversation_history=conversation_history, user_input=user_input, user_info=json.dumps(user_info, indent=2))
            
            # Use the conversational LLM to respond
            bot_response = conversational_llm.predict(formatted_input)
            print(f"Bot: {bot_response}")
            
            # Use the extraction LLM to extract information
            user_info = extract_information_with_llm(user_input, user_info)
            print(f"Updated User Information: {json.dumps(user_info, indent=2)}")
            
            # Append to chat history
            chat_history.append({"user": user_input, "bot": bot_response})
        except Exception as e:
            print(f"An error occurred: {e}")

    # Save chat history and user info to a file
    save_chat_history(chat_history, user_info)

def save_chat_history(chat_history, user_info):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_filename = f"chat_history_{timestamp}.json"
    info_filename = f"user_info_{timestamp}.json"

    with open(history_filename, "w") as file:
        json.dump(chat_history, file, indent=4)
    print(f"Chat history saved to {history_filename}")

    with open(info_filename, "w") as file:
        json.dump(user_info, file, indent=4)
    print(f"User information saved to {info_filename}")

# Run the chatbot function
chat_with_bot()
