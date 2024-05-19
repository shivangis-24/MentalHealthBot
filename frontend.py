from main import ChatBot
import streamlit as st

# Initialize the chatbot
bot = ChatBot()

st.set_page_config(page_title="Symptom-chatbot")

with st.sidebar:
    st.title('Hi there! I am a mental health symptom analyzing chatbot!')

# Function for formatting past messages
def conv_past(inp):
    ret = []
    for num, comb in enumerate(inp):
        ret.append(f"Message {num % 2} by the {comb['role']}: {comb['content']}\n")
    return ret

# Function for generating LLM response
def generate_response(input):
    result = bot.generate_response(
        context=bot.docsearch.as_retriever(),
        question=input,
        pasts=conv_past(st.session_state.messages)
    )
    return result

# Function for processing the response
def afterRes(input_string):
    question_index = input_string.find("Question:")
    if question_index == -1:
        return input_string

    answer_index = input_string.find("Answer:", question_index)
    if answer_index == -1:
        return input_string

    text_after_answer = input_string[answer_index + len("Answer:"):].strip()
    return text_after_answer

# Initialize chat messages in session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi there!"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                response = generate_response(input)
                response = afterRes(response)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
