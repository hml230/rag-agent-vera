"""Main function to run app"""
import logging
import os
import getpass

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph, END
from dotenv import load_dotenv

from storage import PapersDB
from vector_store import PaperVectorStore
from query import FreeDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")


try:
    db = PapersDB()
    embed_store = PaperVectorStore(db)
    if db.get_paper_count() == 0:
        api_query = FreeDataCollector(db)
        api_query.query_data(topics=['machine learning', 'AI safety',
                                     'spatial statistics', 'disease statistics'])
        logger.info("Queried %s papers", db.get_paper_count())
    logger.info("Sucessfully build and stored %s papers", db.get_paper_count())

except Exception as e:
    logger.error("Exception %s raised while attempting to initialise...", e)


graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = embed_store.vstore.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def lookup_by_keyword(keyword: str) -> tuple:
    """Perform a keyword-based search in paper titles"""
    cursor = db.conn.execute("""
                             SELECT title, url
                             FROM papers
                             WHERE title LIKE ?
                             OR abstract LIKE ?
                             """, (f"%{keyword}%",f"%{keyword}%"))
    results = cursor.fetchall()
    formatted_results ="\n".join([f"{title} - {url}" for title, url in results])
    return formatted_results, results


# Build agent
tools = [retrieve, lookup_by_keyword]
llm = init_chat_model("mistral-small-2503", model_provider="mistralai")


# Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Execute the retrieval and lookup.
tool_node = ToolNode(tools)

# Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessage4 as per usage policy
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(message.content for message in tool_messages if message.content)
    system_message_content = (
        "You are a statistics graduate research assistant "
        "responsible for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum, list maximum 5 points"
        "and keep the answer concise. Format your answer as usual text,"
        "not markdown."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Router function for switching edges
def router(state: MessagesState):
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    # If the last message has tool calls, direct the edge to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return END


graph_builder.add_node("user_or_tool", query_or_respond)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate", generate)

# Define transitions
graph_builder.set_entry_point("user_or_tool")
graph_builder.add_conditional_edges(
    "user_or_tool",
    router,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()
