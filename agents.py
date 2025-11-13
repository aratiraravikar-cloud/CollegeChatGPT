from csv import reader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

def create_llm_msg(system_prompt,history):
    resp=[SystemMessage(content=system_prompt)]
    print(resp)
    msgs = history
    print("----------")
    print(msgs)
    for m in msgs:
        if m["role"] == "user":
            resp.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            resp.append(AIMessage(content=m["content"]))
    print(f"DEBUG CREATE LLM MSGS: {history=}\n{resp=}")
    return resp

class AgentState(BaseModel):
    """State of the agent."""
    messages: list = []
    response: str = ""
    category: str = ""

class Category(BaseModel):
    """Category for the agent."""
    category: str

class ChatbotAgent():
    """A chatbot agent that interacts with users."""

    def __init__(self, api_key: str):
        self.key = api_key
        self.model = ChatOpenAI(model_name="gpt-5-nano", max_tokens = 2000, openai_api_key=api_key)
        workflow = StateGraph(AgentState)
        workflow.add_node("classifier", self.classifier)
        workflow.add_node("BS", self.BS)
        workflow.add_node("MD", self.MD)
        workflow.add_node("LAW", self.LAW)
        #workflow.add_node("feedback_agent", self.feedback_agent)
        workflow.add_edge(START, "classifier")
        workflow.add_conditional_edges("classifier", self.main_router)
        #workflow.add_edge("classifier", "smalltalk_agent")
        #workflow.add_edge("classifier", "complaint_agent")
        #workflow.add_edge("classifier", "status_agent")
        #workflow.add_edge("classifier", "feedback_agent")
        workflow.add_edge("BS", END)
        workflow.add_edge("MD", END)
        workflow.add_edge("LAW", END)
        #workflow.add_edge("feedback_agent", END)

        self.graph = workflow.compile()



    def classifier(self, state: AgentState):
        #print("Initial classsifier")
        messages=state.messages
        CLASSIFIER_PROMPT = """
        You are a helpful assistant that classifies user messages into categories.
        Given the following messages, classify them into one of the following categories:
        - BS
        - MD
        - LAW


        If you don't know the category, classify it as "smalltalk_agent".
        """
        llm_messages = create_llm_msg(CLASSIFIER_PROMPT, state.messages)
        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)
        category=llm_response.category
        print(f"Classified category: {category}")
        return {"category":category}

    def main_router(self, state: AgentState):
        #print("Routing to appropriate agent based on category")
        #print(f"DEBUG: Current state: {state}")
        #print(f"DEBUG: Current category: {state.category}")
        return state.category

    def BS(self, state: AgentState):
        print("BS agent processing....")
        BS_PROMPT = f"""
        You are a BS agent that engages in admission process for BS students.
        Given the following messages, respond appropriately to the user's message.
        """
        rag_result = create_bs_msg(self,BS_PROMPT, state.messages)
        final_response_text = rag_result
        #return {"response": self.model.stream(llm_messages), "category": "BS"}
        return {"response":self.model.stream(final_response_text), "category":"BS"}

    def MD(self, state: AgentState):
        print("MD agent processing....")
        MD_PROMPT = f"""
        You are a MD agent that addresses MD students conversation.
        Given the following messages, respond appropriately to the user's message.
        """
        rag_result = create_md_msg(self,MD_PROMPT, state.messages)
        final_response_text = rag_result
        return {"response":self.model.stream(final_response_text), "category":"MD"}

    def LAW(self, state: AgentState):
        print("LAW agent processing....")
        LAW_PROMPT = f"""
        You are a LAW agent that addresses LAW conversation
        Given the following messages, respond appropriately to the user's message.
        """
        llm_messages = create_llm_msg(LAW_PROMPT, state.messages)
        return {"response": self.model.stream(llm_messages), "category": "LAW"}
    
def create_md_msg(self,system_prompt,history):
    
    reader = PdfReader("/workspaces/CollegeChatGPT/datasets/MSAR018 - MSAR Admission Policies and Information.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=10000,
            chunk_overlap=100,
            length_function=len
        )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(api_key=self.key)

    #creating a vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    match = vector_store.similarity_search(history[0].get('content'),k=3)  # Get top 3 matches

    chain = load_qa_chain(self.model,chain_type="stuff")

    response = chain.invoke(input={"input_documents":match, "question":history[0].get('content')})
    #print(response['output_text'])
    return response['output_text']

def predict_admission(row, cr_score, math_score, writing_score):
    if (row['SAT Critical Reading 25th percentile score'] <= cr_score <= row['SAT Critical Reading 75th percentile score'] and
        row['SAT Math 25th percentile score'] <= math_score <= row['SAT Math 75th percentile score'] and
        row['SAT Writing 25th percentile score'] <= writing_score <= row['SAT Writing 75th percentile score']):
        return 'Yes'
    else:
        return 'No'
    
def create_bs_msg(self,system_prompt,history):
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_openai import OpenAIEmbeddings
  from langchain_community.vectorstores.faiss import FAISS
  from langchain_classic.chains.question_answering import load_qa_chain
  import pandas as pd
  my_data = history[0].get('content').split(',')
 
  df = pd.read_csv("/workspaces/CollegeChatGPT/datasets/BSCollegeAdmission.csv")
  df = df[['Name', 'SAT Critical Reading 25th percentile score', 'SAT Critical Reading 75th percentile score',
         'SAT Math 25th percentile score', 'SAT Math 75th percentile score',
         'SAT Writing 25th percentile score', 'SAT Writing 75th percentile score',
         'Percent admitted - total']]
  
  df['Predicted Admission'] = df.apply(predict_admission, 
                                     cr_score=int(my_data[1]), 
                                     math_score=int(my_data[2]), 
                                     writing_score=int(my_data[3]), axis=1)

# Show the results for each university

  print(df[['Name','Predicted Admission']])

  
  matching_universities = df[df['Predicted Admission'] == 'Yes']
  print("You are eligible for admission to the following universities based on your SAT scores:")
  print(matching_universities[['Name', 'SAT Critical Reading 25th percentile score', 'SAT Critical Reading 75th percentile score', 
                             'SAT Math 25th percentile score', 'SAT Math 75th percentile score', 
                             'SAT Writing 25th percentile score', 'SAT Writing 75th percentile score']])
  response = matching_universities
  #print(response['output_text'])
  #return response['output_text']