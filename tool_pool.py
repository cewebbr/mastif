"""
Singleton Tool Pool

Provides a shared pool of builtin tools that can be used across
multiple agentic frameworks (LangChain, CrewAI, smolagents, LlamaIndex, Semantic Kernel).

Tools in the pool:
    - web_search            : DuckDuckGo web search (via duckduckgo-search)
    - web_browser           : Playwright headless browser navigation
    - wikipedia             : Wikipedia lookup (via wikipedia)
    - arxiv                 : Academic paper search (via arxiv)
    - python_repl           : Sandboxed Python code execution (via RestrictedPython)
    - requests_get          : Simple HTTP GET requests (via requests)
    - beautifulsoup_scraper : HTML structure extraction (via beautifulsoup4)
    - pdf_reader            : PDF text extraction (via pypdf)
    - datetime              : Current date and time (stdlib)
    - json_parser           : JSON string parsing and querying (stdlib)
    - pubmed                : Biomedical paper search (via biopython)
    - youtube_transcript    : YouTube video transcript retrieval (via youtube-transcript-api)
    - sympy                 : Symbolic math and logic evaluation (via sympy)
    - web_interaction       : Interactive browser tool for multi-step web interactions (via Playwright)
    - keyboard_interaction  : Keyboard-driven browser interaction optimized for accessibility-style tasks (via Playwright)

Each framework requests a clone of the tool via get_tool(name, framework).
The pool itself is a singleton — only one instance is ever created.
"""

import os
import copy
import time as _time
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
        self._raw_func = func
        self.func = self._make_instrumented(func)

    def _make_instrumented(self, func):
        """Wrap func to log every invocation to ToolPool.invocation_log."""
        tool_name = self.name

        def _instrumented(query):
            t0 = _time.perf_counter()
            try:
                result = func(query)
                duration_ms = round((_time.perf_counter() - t0) * 1000, 2)
                success = True
                error = None
            except Exception as e:
                duration_ms = round((_time.perf_counter() - t0) * 1000, 2)
                result = f"Tool error: {str(e)}"
                success = False
                error = str(e)

            ToolPool.invocation_log.append({
                "tool":        tool_name,
                "input":       str(query)[:300],
                "output":      str(result)[:300],
                "duration_ms": duration_ms,
                "success":     success,
                "error":       error,
            })
            if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                summary = f"🔧 Tool call | {tool_name} | success={success} | duration={duration_ms}ms"
                if error:
                    summary += f" | error={error}"
                summary += f" | input={str(query)[:120]!r}"
                print(summary)
            return result

        return _instrumented

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
            "__module__": __name__,
            "__annotations__": {
                "name": str,
                "description": str,
                "args_schema": type,
            },
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


def _to_llamaindex(tool_def: ToolDefinition):
    """Return a LlamaIndex FunctionTool wrapping this definition."""
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=tool_def.func,
        name=tool_def.name,
        description=tool_def.description
    )


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
    "llamaindex":       _to_llamaindex,
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
    invocation_log: list = []          # shared across all tool calls

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._build_pool()
        return cls._instance

    def reset_log(self):
        """Clear the invocation log. Call before each test run."""
        _ToolPool.invocation_log.clear()

    def get_log(self) -> list:
        """Return a snapshot of the current invocation log."""
        return list(_ToolPool.invocation_log)

    def get_log_summary(self) -> Dict:
        """
        Return aggregated statistics from the current invocation log.

        Returns a dict with:
            total_calls      : total number of tool invocations
            successful_calls : number of successful invocations
            failed_calls     : number of failed invocations
            tools_used       : list of distinct tool names called
            per_tool         : dict of per-tool stats (calls, avg_duration_ms, failures)
        """
        log = _ToolPool.invocation_log
        if not log:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "tools_used": [],
                "per_tool": {},
            }

        per_tool: Dict[str, Dict] = {}
        for entry in log:
            name = entry["tool"]
            if name not in per_tool:
                per_tool[name] = {"calls": 0, "duration_ms": [], "failures": 0}
            per_tool[name]["calls"] += 1
            per_tool[name]["duration_ms"].append(entry["duration_ms"])
            if not entry["success"]:
                per_tool[name]["failures"] += 1

        per_tool_summary = {
            name: {
                "calls":          s["calls"],
                "avg_duration_ms": round(sum(s["duration_ms"]) / len(s["duration_ms"]), 2),
                "failures":       s["failures"],
            }
            for name, s in per_tool.items()
        }

        return {
            "total_calls":      len(log),
            "successful_calls": sum(1 for e in log if e["success"]),
            "failed_calls":     sum(1 for e in log if not e["success"]),
            "tools_used":       sorted(per_tool.keys()),
            "per_tool":         per_tool_summary,
        }

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _build_pool(self):
        """Instantiate and register all builtin tools."""
        self._registry = {}
        self._register_web_search()
        self._register_web_browser()
        self._register_wikipedia()
        self._register_arxiv()
        self._register_python_repl()
        self._register_requests_get()
        self._register_beautifulsoup_scraper()
        self._register_pdf_reader()
        self._register_datetime()
        self._register_json_parser()
        self._register_pubmed()
        self._register_youtube_transcript()
        self._register_sympy()
        self._register_web_interaction()
        self._register_keyboard_interaction()

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
        """Wikipedia lookup — well-supported across all frameworks."""
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

    def _register_arxiv(self):
        """Arxiv academic paper search — no API key required."""
        def _arxiv(query: str) -> str:
            try:
                import arxiv
                search = arxiv.Search(query=query, max_results=5)
                results = list(search.results())
                if not results:
                    return "No papers found."
                return "\n\n".join(
                    f"Title: {r.title}\nAuthors: {', '.join(a.name for a in r.authors[:3])}\n"
                    f"Published: {r.published.strftime('%Y-%m-%d')}\nSummary: {r.summary[:300]}..."
                    for r in results
                )
            except Exception as e:
                return f"Arxiv error: {str(e)}"

        self._registry["arxiv"] = ToolDefinition(
            name="arxiv",
            description=(
                "Search academic papers on arXiv. "
                "Input should be a plain-text search query about a research topic. "
                "Returns titles, authors, publication dates, and abstracts of the top results."
            ),
            func=_arxiv,
        )

    def _register_python_repl(self):
        """Sandboxed Python REPL — executes code via RestrictedPython."""
        def _python_repl(code: str) -> str:
            try:
                from RestrictedPython import compile_restricted, safe_globals
                from RestrictedPython.Eval import default_guarded_getiter
                from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence

                byte_code = compile_restricted(code, "<string>", "exec")
                local_vars = {}
                restricted_globals = {
                    **safe_globals,
                    "__builtins__": safe_builtins,
                    "_getiter_": default_guarded_getiter,
                    "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                }
                exec(byte_code, restricted_globals, local_vars)  # noqa: S102
                output = local_vars.get("result", restricted_globals.get("_print_", None))
                return str(output) if output is not None else "Code executed successfully with no output."
            except Exception as e:
                return f"Python REPL error: {str(e)}"

        self._registry["python_repl"] = ToolDefinition(
            name="python_repl",
            description=(
                "Execute a snippet of Python code in a sandboxed environment. "
                "Input should be valid Python code as a string. "
                "Assign the final result to a variable named 'result' to capture output. "
                "Useful for calculations, data transformations, and logic evaluation."
            ),
            func=_python_repl,
        )

    def _register_requests_get(self):
        """Simple HTTP GET — fetches raw content from a URL via requests."""
        def _get(url: str) -> str:
            try:
                import requests
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                # Return plain text truncated to avoid token overflow
                return response.text[:4000]
            except Exception as e:
                return f"HTTP GET error: {str(e)}"

        self._registry["requests_get"] = ToolDefinition(
            name="requests_get",
            description=(
                "Perform an HTTP GET request to a URL and return the raw response text. "
                "Input should be a valid URL string (including https://). "
                "Useful for fetching JSON APIs, plain-text files, or lightweight web pages."
            ),
            func=_get,
        )

    def _register_beautifulsoup_scraper(self):
        """HTML structure extraction — parses and returns structured content from a URL."""
        def _scrape(url: str) -> str:
            try:
                import requests
                from bs4 import BeautifulSoup
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                # Remove script and style elements
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                headings = [f"{tag.name.upper()}: {tag.get_text(strip=True)}"
                            for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])]
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
                output = ""
                if headings:
                    output += "Headings:\n" + "\n".join(headings[:20]) + "\n\n"
                if paragraphs:
                    output += "Content:\n" + "\n".join(paragraphs[:20])
                return output[:4000] if output else "No structured content found."
            except Exception as e:
                return f"BeautifulSoup scraper error: {str(e)}"

        self._registry["beautifulsoup_scraper"] = ToolDefinition(
            name="beautifulsoup_scraper",
            description=(
                "Extract structured HTML content (headings and paragraphs) from a URL. "
                "Input should be a valid URL string (including https://). "
                "Lighter than web_browser — does not execute JavaScript. "
                "Useful for HTML structure analysis and accessibility evaluation."
            ),
            func=_scrape,
        )

    def _register_pdf_reader(self):
        """PDF text extraction — reads text from a PDF at a URL or local file path."""
        def _read_pdf(source: str) -> str:
            try:
                import io
                import pypdf
                import requests
                if source.startswith("http://") or source.startswith("https://"):
                    response = requests.get(source, timeout=15)
                    response.raise_for_status()
                    file = io.BytesIO(response.content)
                else:
                    file = open(source, "rb")
                reader = pypdf.PdfReader(file)
                text = "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
                if hasattr(file, "close"):
                    file.close()
                return text[:4000] if text.strip() else "No text could be extracted from the PDF."
            except Exception as e:
                return f"PDF reader error: {str(e)}"

        self._registry["pdf_reader"] = ToolDefinition(
            name="pdf_reader",
            description=(
                "Extract text content from a PDF file. "
                "Input should be a URL to a PDF (including https://) or a local file path. "
                "Useful for reading accessibility standards, technical documents, and reports."
            ),
            func=_read_pdf,
        )

    def _register_datetime(self):
        """Current date and time — zero dependencies, stdlib only."""
        def _datetime(_: str = "") -> str:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            return (
                f"Current UTC date and time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                f"ISO 8601: {now.isoformat()}\n"
                f"Day of week: {now.strftime('%A')}"
            )

        self._registry["datetime"] = ToolDefinition(
            name="datetime",
            description=(
                "Return the current UTC date and time. "
                "Input is ignored — pass any string or leave empty. "
                "Useful for time-aware reasoning, logging, and report generation."
            ),
            func=_datetime,
        )

    def _register_json_parser(self):
        """JSON parsing and querying — stdlib only, zero dependencies."""
        def _parse_json(input_str: str) -> str:
            import json
            try:
                # Support "key::json_string" syntax for targeted key lookup
                if "::" in input_str:
                    key, json_str = input_str.split("::", 1)
                    data = json.loads(json_str.strip())
                    result = data.get(key.strip(), f"Key '{key.strip()}' not found.")
                    return json.dumps(result, indent=2)
                else:
                    data = json.loads(input_str.strip())
                    return json.dumps(data, indent=2)
            except Exception as e:
                return f"JSON parser error: {str(e)}"

        self._registry["json_parser"] = ToolDefinition(
            name="json_parser",
            description=(
                "Parse and pretty-print a JSON string, or extract a specific key from it. "
                "For full parsing, input should be a valid JSON string. "
                "For key lookup, use format: 'key::json_string'. "
                "Useful for processing API responses and structured data."
            ),
            func=_parse_json,
        )

    def _register_pubmed(self):
        """PubMed biomedical paper search — via biopython, no API key required."""
        def _pubmed(query: str) -> str:
            try:
                from Bio import Entrez
                Entrez.email = "agent@toolpool.local"
                handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
                record = Entrez.read(handle)
                handle.close()
                ids = record["IdList"]
                if not ids:
                    return "No PubMed results found."
                handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
                abstracts = handle.read()
                handle.close()
                return abstracts[:4000]
            except Exception as e:
                return f"PubMed error: {str(e)}"

        self._registry["pubmed"] = ToolDefinition(
            name="pubmed",
            description=(
                "Search biomedical and life science literature on PubMed. "
                "Input should be a plain-text search query about a medical or biological topic. "
                "Returns abstracts of the top matching papers. No API key required."
            ),
            func=_pubmed,
        )

    def _register_youtube_transcript(self):
        """YouTube transcript retrieval — via youtube-transcript-api, no API key required."""
        def _transcript(input_str: str) -> str:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                # Accept full URL or bare video ID
                if "v=" in input_str:
                    video_id = input_str.split("v=")[-1].split("&")[0]
                elif "youtu.be/" in input_str:
                    video_id = input_str.split("youtu.be/")[-1].split("?")[0]
                else:
                    video_id = input_str.strip()
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                text = " ".join(entry["text"] for entry in transcript)
                return text[:4000]
            except Exception as e:
                return f"YouTube transcript error: {str(e)}"

        self._registry["youtube_transcript"] = ToolDefinition(
            name="youtube_transcript",
            description=(
                "Retrieve the transcript of a YouTube video. "
                "Input should be a YouTube video URL or bare video ID. "
                "Useful for media accessibility evaluation and content analysis."
            ),
            func=_transcript,
        )

    def _register_sympy(self):
        """Symbolic math and logic evaluation — via sympy, no API key required."""
        def _sympy(expression: str) -> str:
            try:
                import sympy
                result = sympy.sympify(expression)
                simplified = sympy.simplify(result)
                return (
                    f"Input:      {expression}\n"
                    f"Parsed:     {result}\n"
                    f"Simplified: {simplified}\n"
                    f"Numeric:    {float(simplified.evalf()) if simplified.is_number else 'N/A'}"
                )
            except Exception as e:
                return f"SymPy error: {str(e)}"

        self._registry["sympy"] = ToolDefinition(
            name="sympy",
            description=(
                "Evaluate, simplify, or solve a symbolic mathematical expression. "
                "Input should be a valid mathematical expression string (e.g. 'x**2 + 2*x + 1'). "
                "Useful for calculations, formula verification, and logic evaluation."
            ),
            func=_sympy,
        )

    def _register_web_interaction(self):
        """Interactive browser tool."""
        def _interact(input_str: str) -> str:
            """
            Expected input format (JSON string):

            {
                "url": "https://example.com",
                "actions": [
                    {"type": "navigate"},
                    {"type": "click", "selector": "button.login"},
                    {"type": "type", "selector": "input#username", "text": "myuser"},
                    {"type": "type", "selector": "input#password", "text": "mypassword"},
                    {"type": "press", "key": "Enter"},
                    {"type": "wait", "ms": 2000},
                    {"type": "select", "selector": "select#country", "value": "Brazil"},
                    {"type": "hover", "selector": ".menu"},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "extract", "selector": "body"},
                    {"type": "extract_all", "selector": ".item"},
                    {"type": "get_url"},
                    {"type": "get_title"},
                    {"type": "screenshot"}
                ]
            }
            """
            import json

            try:
                from playwright.sync_api import sync_playwright

                data = json.loads(input_str)
                url = data.get("url")
                actions = data.get("actions", [])

                if not url:
                    return "Error: 'url' is required."

                results = []

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context()
                    page = context.new_page()

                    page.goto(url, timeout=15000)

                    for action in actions:
                        action_type = action.get("type")

                        try:
                            if action_type == "navigate":
                                continue

                            elif action_type == "click":
                                page.click(action["selector"], timeout=5000)

                            elif action_type == "type":
                                page.fill(action["selector"], action.get("text", ""), timeout=5000)

                            elif action_type == "press":
                                page.keyboard.press(action.get("key", "Enter"))

                            elif action_type == "wait":
                                page.wait_for_timeout(action.get("ms", 1000))

                            elif action_type == "wait_for_selector":
                                page.wait_for_selector(action["selector"], timeout=5000)

                            elif action_type == "select":
                                page.select_option(action["selector"], action.get("value"))

                            elif action_type == "hover":
                                page.hover(action["selector"])

                            elif action_type == "scroll":
                                direction = action.get("direction", "down")
                                amount = action.get("amount", 1000)
                                if direction == "down":
                                    page.mouse.wheel(0, amount)
                                else:
                                    page.mouse.wheel(0, -amount)

                            elif action_type == "go_back":
                                page.go_back()

                            elif action_type == "go_forward":
                                page.go_forward()

                            elif action_type == "extract":
                                selector = action.get("selector", "body")
                                text = page.inner_text(selector)
                                results.append(text[:2000])

                            elif action_type == "extract_all":
                                selector = action["selector"]
                                elements = page.query_selector_all(selector)
                                texts = [el.inner_text() for el in elements[:20]]
                                results.append("\n".join(texts))

                            elif action_type == "get_url":
                                results.append(page.url)

                            elif action_type == "get_title":
                                results.append(page.title())

                            elif action_type == "get_html":
                                results.append(page.content()[:4000])

                            elif action_type == "screenshot":
                                path = action.get("path", "/tmp/screenshot.png")
                                page.screenshot(path=path)
                                results.append(f"Screenshot saved to {path}")

                            else:
                                results.append(f"Unknown action: {action_type}")

                        except Exception as step_error:
                            results.append(f"[{action_type} error]: {str(step_error)}")

                    browser.close()

                return "\n\n".join(results) if results else "No output."

            except Exception as e:
                return f"Web interaction error: {str(e)}"

        self._registry["web_interaction"] = ToolDefinition(
            name="web_interaction",
            description=(
                "Interact with web pages using a headless browser with multi-step actions. "
                "Supports navigation, clicking, typing, key presses, dropdown selection, scrolling, "
                "waiting, extracting content, and retrieving page metadata. "
                "Input must be a JSON string specifying a URL and a sequence of actions."
            ),
            func=_interact,
        )

    def _register_keyboard_interaction(self):
        """Keyboard-driven browser interaction — optimized for accessibility-style and Mind2Web tasks."""
        def _interact(input_str: str) -> str:
            """
            Expected input format (JSON string):

            {
                "url": "https://example.com",
                "actions": [
                    {"type": "navigate"},
                    {"type": "press", "key": "Tab", "count": 5},
                    {"type": "press", "key": "Enter"},
                    {"type": "type", "text": "search query"},
                    {"type": "press", "key": "Enter"},
                    {"type": "shortcut", "keys": ["Control", "L"]},
                    {"type": "wait", "ms": 1000},
                    {"type": "extract_focused"},
                    {"type": "extract_active_element"},
                    {"type": "get_focus_path"},
                    {"type": "get_url"},
                    {"type": "get_title"}
                ]
            }
            """
            import json

            try:
                from playwright.sync_api import sync_playwright

                data = json.loads(input_str)
                url = data.get("url")
                actions = data.get("actions", [])

                if not url:
                    return "Error: 'url' is required."

                results = []

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context()
                    page = context.new_page()

                    page.goto(url, timeout=15000)
                    page.wait_for_load_state("domcontentloaded")

                    for action in actions:
                        action_type = action.get("type")

                        try:
                            if action_type == "navigate":
                                continue

                            elif action_type == "press":
                                key = action.get("key", "Tab")
                                count = action.get("count", 1)
                                for _ in range(count):
                                    page.keyboard.press(key)

                            elif action_type == "type":
                                text = action.get("text", "")
                                page.keyboard.type(text, delay=20)

                            elif action_type == "shortcut":
                                keys = action.get("keys", [])
                                combo = "+".join(keys)
                                page.keyboard.press(combo)

                            elif action_type == "wait":
                                page.wait_for_timeout(action.get("ms", 1000))

                            elif action_type == "extract_focused":
                                el = page.evaluate_handle("document.activeElement")
                                text = el.evaluate("(e) => e ? e.innerText || e.value || '' : ''")
                                results.append(str(text)[:1000])

                            elif action_type == "extract_active_element":
                                html = page.evaluate(
                                    "e => e ? e.outerHTML : ''",
                                    page.evaluate_handle("document.activeElement")
                                )
                                results.append(str(html)[:2000])

                            elif action_type == "get_focus_path":
                                path = page.evaluate("""
                                    () => {
                                        let el = document.activeElement;
                                        if (!el) return "";
                                        let path = [];
                                        while (el) {
                                            let name = el.tagName.toLowerCase();
                                            if (el.id) name += "#" + el.id;
                                            if (el.className) name += "." + el.className.split(" ").join(".");
                                            path.unshift(name);
                                            el = el.parentElement;
                                        }
                                        return path.join(" > ");
                                    }
                                """)
                                results.append(path)

                            elif action_type == "get_url":
                                results.append(page.url)

                            elif action_type == "get_title":
                                results.append(page.title())

                            elif action_type == "extract_page_text":
                                text = page.inner_text("body")
                                results.append(text[:2000])

                            else:
                                results.append(f"Unknown action: {action_type}")

                        except Exception as step_error:
                            results.append(f"[{action_type} error]: {str(step_error)}")

                    browser.close()

                return "\n\n".join(results) if results else "No output."

            except Exception as e:
                return f"Web keyboard interaction error: {str(e)}"

        self._registry["web_keyboard_interaction"] = ToolDefinition(
            name="web_keyboard_interaction",
            description=(
                "Interact with web pages using keyboard-only navigation (Tab, Enter, shortcuts). "
                "Designed for accessibility-style navigation and Mind2Web tasks where DOM selectors are unknown. "
                "Supports key presses, typing, shortcuts, focus inspection, and content extraction from the active element."
            ),
            func=_interact,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool(self, name: str, framework: str):
        """
        Return a framework-specific clone of the requested tool.

        Args:
            name:      One of "web_search", "web_browser", "wikipedia", "arxiv",
                       "python_repl", "requests_get", "beautifulsoup_scraper",
                       "pdf_reader", "datetime", "json_parser", "pubmed",
                       "youtube_transcript", "sympy", "web_interaction".
            framework: One of "langchain", "crewai", "smolagents", "llamaindex", "semantic_kernel".

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
            framework: One of "langchain", "crewai", "smolagents", "llamaindex", "semantic_kernel".

        Returns:
            List of all tool objects ready to be used by the specified framework.
            Includes: web_search, web_browser, wikipedia, arxiv, python_repl,
            requests_get, beautifulsoup_scraper, pdf_reader, datetime, json_parser,
            pubmed, youtube_transcript, sympy, web_interaction.
        """
        return [self.get_tool(name, framework) for name in self._registry]

    def get_tool_schema(self, name: str) -> dict:
        """
        Return an API-compatible tool schema for tool calling.
        """
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' not found in pool. Available: {list(self._registry)}")

        tool_def = self._registry[name]
        schema_map = {
            "web_search": ("Search the web using DuckDuckGo.", {"query": {"type": "string", "description": "Search query string."}}),
            "web_browser": ("Navigate a headless browser to a URL and return visible page text.", {"url": {"type": "string", "description": "URL of the page to browse."}}),
            "wikipedia": ("Look up a topic on Wikipedia and return a short summary.", {"query": {"type": "string", "description": "Topic to search on Wikipedia."}}),
            "arxiv": ("Search academic papers on arXiv.", {"query": {"type": "string", "description": "Search query for arXiv papers."}}),
            "python_repl": ("Execute a snippet of Python code in a sandboxed environment.", {"code": {"type": "string", "description": "Python code to execute."}}),
            "requests_get": ("Perform an HTTP GET request to a URL and return raw text.", {"url": {"type": "string", "description": "URL to fetch."}}),
            "beautifulsoup_scraper": ("Extract structured HTML content from a URL.", {"url": {"type": "string", "description": "URL to scrape."}}),
            "pdf_reader": ("Extract text from a PDF URL or local file path.", {"source": {"type": "string", "description": "PDF URL or local file path."}}),
            "datetime": ("Return the current UTC date and time.", {"input": {"type": "string", "description": "Optional input value (ignored)."}}),
            "json_parser": ("Parse or query a JSON string.", {"input": {"type": "string", "description": "JSON string or key::JSON payload."}}),
            "pubmed": ("Search PubMed for biomedical literature.", {"query": {"type": "string", "description": "Search query for PubMed."}}),
            "youtube_transcript": ("Retrieve a YouTube transcript from a URL or video ID.", {"input": {"type": "string", "description": "YouTube video URL or ID."}}),
            "sympy": ("Evaluate or simplify a symbolic math expression.", {"expression": {"type": "string", "description": "Mathematical expression to evaluate."}}),
            "web_interaction": ("Interact with a web page using structured browser actions.", {"input": {"type": "string", "description": "JSON string containing a URL and action sequence."}}),
            "web_keyboard_interaction": ("Interact with a web page using keyboard-driven browser actions.", {"input": {"type": "string", "description": "JSON string containing a URL and keyboard action sequence."}}),
        }

        description, properties = schema_map.get(
            name,
            (tool_def.description, {"input": {"type": "string", "description": "Tool input string."}})
        )

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            },
        }

    @property
    def available_tools(self) -> list:
        """List of tool names currently registered in the pool."""
        return list(self._registry.keys())

    def get_openai_schemas(self, names: list = None) -> list:
        """
        Return tools as OpenAI-compatible function-calling schema dicts.
        Compatible with both OpenAI API and HuggingFace chat_completion tools parameter.

        Args:
            names: List of tool names to include. Defaults to all registered tools.

        Returns:
            List of dicts in the format:
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Input query or argument."}
                        },
                        "required": ["query"]
                    }
                }
            }
        """
        names = names or list(self._registry.keys())
        schemas = []
        for name in names:
            if name not in self._registry:
                continue
            schema = self.get_tool_schema(name)
            if isinstance(schema, dict) and schema.get("type") == "function":
                schemas.append(schema)
            elif isinstance(schema, dict) and "name" in schema and "parameters" in schema:
                schemas.append({"type": "function", "function": schema})
        return schemas

# Singleton instance — import this directly
ToolPool = _ToolPool()