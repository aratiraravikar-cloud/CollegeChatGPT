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

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
admission_level = st.radio(
    "What kind of admission you are looking for?",
    ["***B.S.***", "***M.D.***", "***LAW***"],
    captions=[
        "Bachelor of Science",
        "Doctor of Medicine",
        "Law School",
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
    collegename = st.text_input("Enter your college name: ")
else:
    st.write("Congradulations on your under graduation")

if st.button("Submit",type="primary"):
    if admission_level == "***B.S.***":
        user_input="Give me the college list for BS with," + user_sat_reading + "," + user_sat_math + "," + user_sat_writing
        st.write("You are eligible for admission to the following universities based on your scores:")
    elif admission_level == "***M.D.***":
        user_input="Give me the Admission Policies and Information for MD, " + collegename
        st.write("Here is the information for Admission Policies for the selected college.")

    msg_list=[]
    msg_list.append({"role":"user","content":user_input})
    app=agents.ChatbotAgent(api_key=st.secrets["OpenAI_key"])
    thread_id = 10
    thread={"configurable":{"thread_id":thread_id}}
    full_resp = ""

    for s in app.graph.stream({'messages': msg_list}, thread):
        print("response ===== " + str(s))
        
        for key, value in s.items():
            print(f"Key : {key}, value: {value}")
            if resp_gen := value.get("response"):
                print(f"Assistant: ")
                #print(resp_gen[0])
                for chunk in resp_gen:
                    #print(chunk)
                    text = getattr(chunk, "content", None) or str(chunk) #or getattr(chunk, "delta", None) # or 
                    if text:
                        print(text, end="", flush=True)
                        full_resp += text

    if full_resp:
        msg_list.append({"role":"assistant","content":full_resp})
        print("----------")
        print(msg_list)
        st.write(full_resp)
