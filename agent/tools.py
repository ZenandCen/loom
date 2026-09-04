"""Tool definitions for the loom agent."""

from langchain_core.tools import tool
from tavily import TavilyClient

from agent.config import store

tavily_client = TavilyClient()


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query to execute
    """
    results = tavily_client.search(query, max_results=5)
    return "\n\n".join(
        f"**{r['title']}**\n{r['url']}\n{r['content']}" for r in results["results"]
    )


@tool(parse_docstring=True)
def send_email(to: str, subject: str, content: str) -> str:
    """Send an email to a recipient.

    Args:
        to: Recipient email address
        subject: Email subject line
        content: Email body content
    """
    return f"Email sent to {to}: '{subject}'"


@tool(parse_docstring=True)
def remember(fact: str, category: str = "preference") -> str:
    """Save a fact to long-term memory for future sessions.

    Args:
        fact: The fact or preference to remember
        category: Category — preference, context, or rule
    """
    from langgraph.config import get_config

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")
    store.put(("loom", user_id, category), fact, {"fact": fact, "category": category})
    return f"Remembered ({category}): {fact}"


@tool(parse_docstring=True)
def recall(query: str) -> str:
    """Search long-term memory for relevant facts.

    Args:
        query: What to recall from memory
    """
    from langgraph.config import get_config

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")
    results = store.search(("loom", user_id), query=query)
    if not results:
        return "No relevant memories found."
    return "\n".join(item.value["fact"] for item in results)


# --- Tools list (pass to agent) ---
all_tools = [tavily_search, send_email, remember, recall]
