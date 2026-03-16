"""
Singleton Tool Pool

Provides a shared pool of builtin tools that can be used across
multiple agentic frameworks (LangChain, CrewAI, smolagents, Semantic Kernel).

Tools in the pool:
    - web_search  : DuckDuckGo web search (via langchain-community)
    - web_browser : Playwright browser navigation (via langchain-community)
    - wikipedia   : Wikipedia lookup (via langchain-community)

Each framework requests a clone of the tool via get_tool(name, framework).
The pool itself is a singleton — only one instance is ever created.
"""

import copy
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Base tool definition (framework-agnostic)
# ---------------------------------------------------------------------------

class ToolDefinition:
    """
    Framework-agnostic tool definition stored in the pool.

    Attributes:
        name:        Canonical tool name.
        description: Human-readable description of what the tool does.
        func:        Underlying callable that executes the tool logic.
    """

    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def __repr__(self):
        return f"ToolDefinition(name={self.name!r})"


# ---------------------------------------------------------------------------
# Adapters — wrap a ToolDefinition into the format expected by each framework
# ---------------------------------------------------------------------------

def _to_langchain(tool_def: ToolDefinition):
    """Return a langchain_community Tool wrapping this definition."""
    from langchain_community.tools import Tool as LCTool
    return LCTool(
        name=tool_def.name,
        func=tool_def.func,
        description=tool_def.description,
    )


def _to_crewai(tool_def: ToolDefinition):
    """Return a crewai BaseTool subclass instance wrapping this definition."""
    from crewai.tools import BaseTool as CrewBaseTool
    from pydantic import BaseModel, Field

    class _Input(BaseModel):
        query: str = Field(description="Input query or argument for the tool.")

    # Dynamically create a named subclass so CrewAI can identify the tool
    def _run(self, query: str) -> str:
        return tool_def.func(query)

    CrewTool = type(
        tool_def.name,
        (CrewBaseTool,),
        {
            "name": tool_def.name,
            "description": tool_def.description,
            "args_schema": _Input,
            "_run": _run,
        },
    )
    return CrewTool()


def _to_smolagents(tool_def: ToolDefinition):
    """Return a smolagents Tool instance wrapping this definition."""
    from smolagents import Tool as SmolTool

    class _SmolWrapper(SmolTool):
        name = tool_def.name
        description = tool_def.description
        inputs = {"query": {"type": "string", "description": "Input query or argument."}}
        output_type = "string"

        def forward(self, query: str) -> str:
            return tool_def.func(query)

    return _SmolWrapper()


def _to_semantic_kernel(tool_def: ToolDefinition):
    """Return a Semantic Kernel KernelFunction wrapping this definition.

    The function is registered on a throw-away kernel instance so it can be
    retrieved as a standalone KernelFunction and added to any kernel via
    kernel.add_plugin() by the caller.
    """
    import semantic_kernel as sk
    from semantic_kernel.functions.kernel_function_decorator import kernel_function

    # Dynamically create a plugin class with a single kernel_function method
    @kernel_function(name=tool_def.name, description=tool_def.description)
    def _sk_func(query: str) -> str:
        return tool_def.func(query)

    plugin_class = type(tool_def.name, (), {tool_def.name: staticmethod(_sk_func)})
    temp_kernel = sk.Kernel()
    plugin = temp_kernel.add_plugin(
        plugin=plugin_class(),
        plugin_name=tool_def.name
    )
    return plugin[tool_def.name]


_ADAPTERS = {
    "langchain":        _to_langchain,
    "crewai":           _to_crewai,
    "smolagents":       _to_smolagents,
    "semantic_kernel":  _to_semantic_kernel,
}


# ---------------------------------------------------------------------------
# Tool pool — singleton
# ---------------------------------------------------------------------------

class _ToolPool:
    """
    Singleton registry of shared builtin tools.

    Usage:
        from tool_pool import ToolPool

        # Get a LangChain-compatible clone of web_search
        tool = ToolPool.get_tool("web_search", framework="langchain")

        # Get all tools for CrewAI
        tools = ToolPool.get_all_tools(framework="crewai")
    """

    _instance: Optional["_ToolPool"] = None
    _registry: Dict[str, ToolDefinition] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build_pool()
        return cls._instance

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pool(self):
        """Instantiate and register all builtin tools."""
        self._registry = {}
        self._register_web_search()
        self._register_web_browser()
        self._register_wikipedia()

    def _register_web_search(self):
        """DuckDuckGo web search — works natively with LangChain, CrewAI, smolagents."""
        from duckduckgo_search import DDGS

        def _search(query: str) -> str:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            return "\n\n".join(
                f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
                for r in results
            )

        self._registry["web_search"] = ToolDefinition(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. "
                "Input should be a plain-text search query. "
                "Returns titles, URLs, and snippets from the top results."
            ),
            func=_search,
        )

    def _register_web_browser(self):
        """Playwright headless browser — navigate to a URL and return page text."""
        def _browse(url: str) -> str:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=15000)
                    text = page.inner_text("body")
                    browser.close()
                return text[:4000]  # Truncate to avoid token overflow
            except Exception as e:
                return f"Browser error: {str(e)}"

        self._registry["web_browser"] = ToolDefinition(
            name="web_browser",
            description=(
                "Navigate a headless browser to a URL and return the visible page text. "
                "Input should be a valid URL string (including https://). "
                "Useful for reading web pages that require JavaScript rendering."
            ),
            func=_browse,
        )

    def _register_wikipedia(self):
        """Wikipedia lookup — well-supported across all three frameworks."""
        def _wiki(query: str) -> str:
            try:
                import wikipedia
                summary = wikipedia.summary(query, sentences=5, auto_suggest=False)
                return summary
            except Exception as e:
                return f"Wikipedia error: {str(e)}"

        self._registry["wikipedia"] = ToolDefinition(
            name="wikipedia",
            description=(
                "Look up a topic on Wikipedia and return a short summary. "
                "Input should be the name of a concept, person, place, or event. "
                "Useful for factual background information."
            ),
            func=_wiki,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool(self, name: str, framework: str):
        """
        Return a framework-specific clone of the requested tool.

        Args:
            name:      One of "web_search", "web_browser", "wikipedia".
            framework: One of "langchain", "crewai", "smolagents", "semantic_kernel".

        Returns:
            A tool object ready to be used by the specified framework.

        Raises:
            KeyError:   If the tool name is not in the pool.
            ValueError: If the framework is not supported.
        """
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' not found in pool. Available: {list(self._registry)}")
        if framework not in _ADAPTERS:
            raise ValueError(f"Framework '{framework}' not supported. Choose from: {list(_ADAPTERS)}")

        tool_def = self._registry[name]
        return _ADAPTERS[framework](tool_def)

    def get_all_tools(self, framework: str) -> list:
        """
        Return framework-specific clones of all tools in the pool.

        Args:
            framework: One of "langchain", "crewai", "smolagents", "semantic_kernel".

        Returns:
            List of tool objects ready to be used by the specified framework.
        """
        return [self.get_tool(name, framework) for name in self._registry]

    @property
    def available_tools(self) -> list:
        """List of tool names currently registered in the pool."""
        return list(self._registry.keys())


# Singleton instance — import this directly
ToolPool = _ToolPool()