import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 College ChatGPT")
st.write(
    "This is a simple chatbot that guides you to get eligible college list based on your scores . "
    "To use this app, you need to provide your acadamic information,  "
    ""
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
#openai_api_key = st.text_input("OpenAI API Key", type="password")
#if not openai_api_key:
#    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
#else:

    # Create an OpenAI client.
print(st.secrets["OpenAI_key"])
client = OpenAI(api_key=st.secrets["OpenAI_key"])

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
admission_level = st.radio(
    "What kind of admission you are looking for?",
    ["***B.S.***", "***M.D.***", "***LAW***"],
    captions=[
        "Bachloer of Science",
        "Doctor of Medicine",
        "Lawyer",
    ],
)

if admission_level == "***B.S.***":
    st.write("Congradulations on your high school graduation")
    gpa = st.chat_input("Enter your unweighted GPA (0.00-4.00)")
    user_sat_reading = st.chat_input("Enter your SAT Critical Reading score: ")
    user_sat_math = st.chat_input("Enter your SAT Math score: ")
    user_sat_writing = st.chat_input("Enter your SAT Writing score: ")
elif admission_level == "***M.D.***":
    st.write("Congradulations on your PreMed undergraduation")
else:
    st.write("Congradulations on your under graduation")
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API.
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    )

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
