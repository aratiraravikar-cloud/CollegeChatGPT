import streamlit as st
from openai import OpenAI
import agents

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
#client = OpenAI(api_key=st.secrets["OpenAI_key"])

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.markdown(message["content"])

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
    gpa = st.text_input("Enter your unweighted GPA (0.00-4.00)")
    st.session_state.messages.append({"role": "user", "content": gpa})
    user_sat_reading = st.text_input("Enter your SAT Critical Reading score: ")
    st.session_state.messages.append({"role": "user", "content": user_sat_reading})
    user_sat_math = st.text_input("Enter your SAT Math score: ")
    st.session_state.messages.append({"role": "user", "content": user_sat_math})
    user_sat_writing = st.text_input("Enter your SAT Writing score: ")
    st.session_state.messages.append({"role": "user", "content": user_sat_writing})    
elif admission_level == "***M.D.***":
    st.write("Congradulations on your PreMed undergraduation")
else:
    st.write("Congradulations on your under graduation")

if st.button("Submit",type="primary"):
    st.write("Button pressed")
    st.write(user_sat_reading)
    msg_list=[]

    user_input="Give me the college list for BS with," + user_sat_reading + "," + user_sat_math + "," + user_sat_writing
    print(user_input)
    msg_list.append({"role":"user","content":user_input})
    app=agents.ChatbotAgent(api_key=st.secrets["OpenAI_key"])
    thread_id = 10
    thread={"configurable":{"thread_id":thread_id}}
    full_resp = ""

    final_state = app.graph.invoke({'messages': msg_list}, thread)
    full_resp = final_state['response']
    #print(app.graph.stream({'messages': msg_list}, thread))
    for s in app.graph.stream({'messages': msg_list}, thread):
    #print(s)
        for k,v in s.items():
            if resp_gen := v.get("response"):
                print(f"Assistant: ")
                print(resp_gen)
                for chunk in resp_gen:
                    text = getattr(chunk, "content", None) or getattr(chunk, "delta", None) # or str(chunk)
                    if text:
                        print(text, end="", flush=True)
                        full_resp += text

    if full_resp:
        msg_list.append({"role":"assistant","content":full_resp})
        print("----------")
        print(msg_list)
