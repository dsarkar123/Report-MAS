import os
import pandas as pd
import asyncio
import json
from typing import Any, List

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import ActionReasoningStep, ObservationReasoningStep
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolOutput, ToolSelection
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

DOCUMENT_PATH = "./MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024.pdf"
INDEX_PERSIST_DIR = "./storage"

# --- LlamaIndex Setup ---
def setup_llama_index():
    """Initializes LlamaIndex settings and loads the data."""
    print("Setting up LlamaIndex...")
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    if not os.path.exists(INDEX_PERSIST_DIR):
        print(f"Loading document from: {DOCUMENT_PATH}")
        documents = SimpleDirectoryReader(input_files=[DOCUMENT_PATH]).load_data()
        print("Building index...")
        index = VectorStoreIndex.from_documents(documents)
        print(f"Persisting index to: {INDEX_PERSIST_DIR}")
        index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)
    else:
        print(f"Loading existing index from: {INDEX_PERSIST_DIR}")
        from llama_index.core import load_index_from_storage, StorageContext
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    print("Index setup complete.")
    return index

# --- Workflow Events ---
class PrepEvent(Event):
    pass

class InputEvent(Event):
    input: list[ChatMessage]

class StreamEvent(Event):
    delta: str

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]

class FunctionOutputEvent(Event):
    output: ToolOutput

# --- ReAct Agent Workflow ---
class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm or OpenAI()
        self.formatter = ReActChatFormatter.from_defaults(context=extra_context or "")
        self.output_parser = ReActOutputParser()

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        await ctx.store.set("sources", [])
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        user_input = ev.get("input")
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        await ctx.store.set("current_reasoning", [])
        await ctx.store.set("memory", memory)
        return PrepEvent()

    @step
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        memory = await ctx.store.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        llm_input = self.formatter.format(self.tools, chat_history, current_reasoning=current_reasoning)
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(self, ctx: Context, ev: InputEvent) -> ToolCallEvent | StopEvent:
        chat_history = ev.input
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        memory = await ctx.store.get("memory")

        response_gen = await self.llm.astream_chat(chat_history)
        full_response = None
        async for response in response_gen:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))
            if full_response is None:
                full_response = response
            else:
                full_response.delta = response.delta
                full_response.message.content += response.delta

        try:
            reasoning_step = self.output_parser.parse(full_response.message.content)
            current_reasoning.append(reasoning_step)
            if reasoning_step.is_done:
                memory.put(ChatMessage(role="assistant", content=reasoning_step.response))
                await ctx.store.set("memory", memory)
                await ctx.store.set("current_reasoning", current_reasoning)
                sources = await ctx.store.get("sources", default=[])
                return StopEvent(result={"response": reasoning_step.response, "sources": sources})
            elif isinstance(reasoning_step, ActionReasoningStep):
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=reasoning_step.action,
                            tool_kwargs=reasoning_step.action_input,
                        )
                    ]
                )
        except Exception as e:
            current_reasoning.append(ObservationReasoningStep(observation=f"There was an error in parsing my reasoning: {e}"))
            await ctx.store.set("current_reasoning", current_reasoning)

        return PrepEvent()

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        sources = await ctx.store.get("sources", default=[])

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                current_reasoning.append(ObservationReasoningStep(observation=f"Tool {tool_call.tool_name} does not exist"))
                continue
            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                current_reasoning.append(ObservationReasoningStep(observation=tool_output.content))
            except Exception as e:
                current_reasoning.append(ObservationReasoningStep(observation=f"Error calling tool {tool.metadata.get_name()}: {e}"))

        await ctx.store.set("sources", sources)
        await ctx.store.set("current_reasoning", current_reasoning)
        return PrepEvent()

def create_agent(index):
    """Creates a ReAct agent with a query engine tool."""
    print("Creating ReAct agent...")
    query_engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="mas_notice_758",
        description=(
            "Provides information from the MAS Notice 758 document. "
            "Use this tool to answer questions about business rules, action points, and amendments."
        ),
    )

    agent = ReActAgent(
        tools=[query_engine_tool],
        llm=Settings.llm,
        verbose=True,
        timeout=300,
    )
    print("ReAct agent created.")
    return agent

def save_to_excel(df, filename="report.xlsx"):
    """Saves a DataFrame to an Excel file."""
    print(f"Saving report to {filename}...")
    df.to_excel(filename, index=False)
    print("Report saved successfully.")

async def handle_query_2(agent):
    """Handles the second query to extract action points and business rules."""
    print("Handling Query 2: Extracting Action Points and Business Rules...")

    prompt = """\
    Analyze the entire MAS Notice 758 document and identify all distinct action points and business rules.
    For each identified item, provide the following details in a structured format:
    - Sr. No.
    - Section
    - Sub-section
    - Text (the relevant excerpt from the document)
    - Classification of Text (Information, Action Point, or Business Rule)
    - If it's an Action Point, specify:
        - Responsible Dept.
        - Compliance To be Done
        - Last Date of Compliance
    - If it's a Business Rule, provide:
        - Interpretation/Simplification

    Do not repeat the same action point. Your final response must be a JSON list of objects, where each object represents one item.
    Example format for an action point:
    [
        {
            "Sr. No.": 1,
            "Section": "3",
            "Sub-section": "3.1",
            "Text": "The bank must submit the revised credit risk policy by the deadline.",
            "Classification of Text": "Action Point",
            "Responsible Dept.": "Credit Risk Department",
            "Compliance To be Done": "Submit revised credit risk policy.",
            "Last Date of Compliance": "2024-12-31",
            "Interpretation/Simplification": ""
        }
    ]
    """

    print("Running agent...")
    handler = agent.run(input=prompt)
    response_obj = await handler
    response_text = response_obj['response']

    print("\n--- Agent Response ---")
    print(response_text)
    print("----------------------")

    try:
        # The agent might return markdown ```json ... ```, so let's clean it up.
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip()[7:-3].strip()

        data = json.loads(response_text)
        df = pd.DataFrame(data)

        required_columns = [
            "Sr. No.", "Section", "Sub-section", "Text", "Classification of Text",
            "Responsible Dept.", "Compliance To be Done", "Last Date of Compliance",
            "Interpretation/Simplification"
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = "" # Use empty string for missing columns

        save_to_excel(df, "MAS_Notice_758_Action_Points_and_Business_Rules.xlsx")

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Error parsing agent response: {e}")
        print("Saving the raw response to a text file for inspection.")
        with open("response_query_2.txt", "w") as f:
            f.write(response_text)

async def handle_query_1(agent):
    """Handles the first query to analyze amendments."""
    print("Handling Query 1: Amendment Analysis...")

    prompt = """\
    First, determine if the provided 'MAS Notice 758' is an amendment to a previous notification.
    If it is an amendment, please analyze the document and identify all the changes being introduced.
    Present the changes in a structured format with the following columns:
    - Sr. No.
    - Section
    - Sub-section
    - Original Text (with all amendments)
    - Relevant Text vide the Current Notification
    - Change (Highlight the change)

    If the original text is not explicitly mentioned in this document, please state that in the 'Original Text' column.
    Your final response must be a JSON list of objects, where each object represents one identified change.
    """

    print("Running agent for amendment analysis...")
    handler = agent.run(input=prompt)
    response_obj = await handler
    response_text = response_obj.get('response', '')

    print("\n--- Agent Response ---")
    print(response_text)
    print("----------------------")

    try:
        if response_text.strip().startswith("```json"):
            response_text = response_text.strip()[7:-3].strip()

        data = json.loads(response_text)
        df = pd.DataFrame(data)

        required_columns = [
            "Sr. No.", "Section", "Sub-section", "Original Text (with all amendments)",
            "Relevant Text vide the Current Notification", "Change (Highlight the change)"
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        save_to_excel(df, "MAS_Notice_758_Amendment_Analysis.xlsx")

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Error parsing agent response: {e}")
        print("Saving the raw response to a text file for inspection.")
        with open("response_query_1.txt", "w") as f:
            f.write(response_text)

async def main():
    """Main function to run the RAG system."""
    try:
        index = setup_llama_index()
        agent = create_agent(index)

        while True:
            choice = input("Which query would you like to run? (1 or 2, or 'exit' to quit): ")
            if choice == '1':
                await handle_query_1(agent)
                break
            elif choice == '2':
                await handle_query_2(agent)
                break
            elif choice.lower() == 'exit':
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 'exit'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())