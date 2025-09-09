from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage #represents input from user
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

#Typed dictionary example
class PersonalDictionary(TypedDict): #TypeDict acts as a super-class
    name: str
    age: int
    is_student: bool

# create out own typed dictionary 
our_person: PersonalDictionary={
    "name":'Hasib',
    "age": 25,
    "is_student":True,
}

def myfunction(value: Union[int, float]):
    return value
# print(our_person)

# defining the state of our agent
class AgentState(TypedDict):
    message: List[Union[HumanMessage,AIMessage, SystemMessage]]




load_dotenv()
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.7) #temperature control creativity of the model’s responses.{For coding/technical work → keep it low.For brainstorming/creative writing → increase it.}

conversation_history=[SystemMessage(content="You are an AI assistant that speaks like a human! Answer all my question properly")]


#purpose of the node is to pass the user's message to the LLM.
def first_node(state: AgentState) -> AgentState:
    response=llm.invoke(state["message"])
    print(f"\nAI: {response.content}")
    return state


def our_processing_node(state: AgentState) -> AgentState:
    response=llm.invoke(state["message"])
    state["message"].append(AIMessage(response.content))
    print(f"\nAI: {response.content}")
    # print(f"Our current state look like: {state['message']}")
    return state

#building graph
graph=StateGraph(AgentState)
# graph.add_node("node1",first_node)
graph.add_node("llm_node",our_processing_node)
graph.add_edge(START,"llm_node")
graph.add_edge("llm_node",END)
agent=graph.compile()


while True:
    user_input=input("Enter: ")
    conversation_history.append(HumanMessage(user_input))
    if user_input!="exit":
        result=agent.invoke({"message": conversation_history})
        conversation_history=result["message"]
    else:
        break