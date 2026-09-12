import re
import sys
from pathlib import Path
from typing import Any, List, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# 1. Environment & Path Resolution (Root access for llm.py & Behaviour)
# -----------------------------------------------------------------------------
load_dotenv()

# Make root-level modules importable when this file is run directly.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Model provider from project root
from llm import get_llm

# Vocal output organ
from Organs.mouth import Mouth

# Persona & canned receipts
from Behaviour.persona import get_system_prompt, get_offline_receipt


# -----------------------------------------------------------------------------
# 2. Cognitive State Definition
# -----------------------------------------------------------------------------
class BrainState(TypedDict, total=False):
    user_prompt: str
    system_prompt: str
    messages: List[BaseMessage]
    selected_llm: Any
    raw_response: str
    final_response: str


# -----------------------------------------------------------------------------
# 3. Brain Organ & Hierarchical LangGraph State Machine
# -----------------------------------------------------------------------------
class Brain:
    def __init__(self, mouth: Mouth | None = None):
        """Initializes vocal connections and compiles the multi-graph hierarchy."""
        self.mouth: Mouth | None = mouth if mouth is not None else Mouth()
        self.graph = self._build_master_graph()

    # used in the output sanitization node to strip formatting symbols before TTS
    def _sanitize_for_tts(self, text: str) -> str:
        """Strips formatting symbols to prevent speech synthesis glitches."""
        clean = re.sub(r"[\*\_#\>\-`]", "", text)
        return " ".join(clean.split()).strip()


    # =========================================================================
    # GRAPH 1 NODES: INPUT QUERY PREPARATION & LLM SELECTION
    # =========================================================================

    # uses the get_system_prompt() from persona.py to construct the system prompt and dialogue messages
    # takes the user prompt from the BrainState and constructs a list of messages for the LLM to process
    def prepare_input_node(self, state: BrainState) -> dict:
        """Constructs persona directives and dialogue messages from raw speech."""
        sys_prompt = get_system_prompt()
        dialogue: List[BaseMessage] = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=state.get("user_prompt", "")),
        ]
        return {
            "system_prompt": sys_prompt,
            "messages": dialogue,
        }

    # uses the get_llm() function from llm.py to select the active LLM instance for processing
    # takes the BrainState and returns the llm to be used
    def select_llm_node(self, state: BrainState) -> dict:
        """Resolves and selects the active LLM waterfall instance for processing."""
        active_llm = get_llm()
        return {"selected_llm": active_llm}


    # =========================================================================
    # GRAPH 2 NODES: COGNITIVE PROCESSING VIA LLM INVOCATION
    # =========================================================================

    # uses the selected LLM from the BrainState to invoke the LLM and extract the raw output text
    # takes the BrainState and returns the raw output text from the LLM
    def processing(self, state: BrainState) -> dict:
        """Invokes the selected LLM and extracts solely the raw output text."""
        llm = state.get("selected_llm")
        if llm is None:
            llm = get_llm()

        try:
            response = llm.invoke(state.get("messages", []))
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict) and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
                content_text = " ".join(text_parts)
            else:
                content_text = str(content)
            return {"raw_response": content_text.strip() if content_text else ""}
        except Exception as e:
            print(f"[Brain Graph 2 - LLM Execution Error]: {e}")
            return {"raw_response": ""}


    # =========================================================================
    # GRAPH 3 NODES: SANITIZATION & DIRECT VOCAL MOTOR OUTPUT
    # =========================================================================

    # uses the get_offline_receipt() from persona.py to provide a fallback response if the LLM output is empty
    # uses the _sanitize_for_tts() method to clean the raw output text for speech synthesis
    # takes the BrainState and returns the final sanitized output text for TTS
    def sanitize_output_node(self, state: BrainState) -> dict:
        """Cleans residual formatting and prepares output text for speech."""
        raw = state.get("raw_response", "")
        if not raw:
            clean_reply = get_offline_receipt()
        else:
            clean_reply = self._sanitize_for_tts(raw)

        return {"final_response": clean_reply}

    # uses the speak() from mouth.py module to vocalize the final sanitized text
    # takes the BrainState and returns an empty dict
    def mouth_speak_node(self, state: BrainState) -> dict:
        """Hands the final sanitized text directly to mouth.py for TTS vocalization."""
        reply = state.get("final_response", "")
        if reply and self.mouth is not None:
            self.mouth.speak(reply)
        return {}


    # =========================================================================
    # SUBGRAPH BUILDERS
    # =========================================================================

    # builds the first subgraph that prepares the user input and selects the appropriate LLM for processing
    # returns a compiled StateGraph that can be invoked with the BrainState
    def ready_for_processing(self):
        """Graph 1: Ingests user input, formats persona directives, and picks the LLM."""
        workflow = StateGraph(BrainState)

        workflow.add_node("prepare_input", self.prepare_input_node)
        workflow.add_node("select_llm", self.select_llm_node)

        workflow.add_edge(START, "prepare_input")
        workflow.add_edge("prepare_input", "select_llm")
        workflow.add_edge("select_llm", END)

        return workflow.compile()

    # builds the second subgraph that invokes the selected LLM and extracts the raw output text
    # returns a compiled StateGraph that can be invoked with the BrainState
    def _build_processing_graph(self):
        """Graph 2: Executes llm.invoke and isolates the raw generated output."""
        workflow = StateGraph(BrainState)

        workflow.add_node("invoke_llm", self.processing)

        workflow.add_edge(START, "invoke_llm")
        workflow.add_edge("invoke_llm", END)

        return workflow.compile()

    # builds the third subgraph that sanitizes the raw output text and dispatches it to the mouth for TTS
    # returns a compiled StateGraph that can be invoked with the BrainState
    def ready_to_speak(self):
        """Graph 3: Sanitizes generated text and dispatches directly to mouth.speak."""
        workflow = StateGraph(BrainState)

        workflow.add_node("sanitize_output", self.sanitize_output_node)
        workflow.add_node("mouth_speak", self.mouth_speak_node)

        workflow.add_edge(START, "sanitize_output")
        workflow.add_edge("sanitize_output", "mouth_speak")
        workflow.add_edge("mouth_speak", END)

        return workflow.compile()


    # =========================================================================
    # MASTER GRAPH BUILDER
    # =========================================================================

    # builds the master graph that orchestrates the sequential flow of all three subgraphs: 
    # input preparation, LLM processing, and vocal output
    # returns a compiled StateGraph that can be invoked with the BrainState
    def _build_master_graph(self):
        """Graph 4: Master pipeline orchestrating the sequential flow of all 3 subgraphs."""
        graph_1 = self.ready_for_processing()
        graph_2 = self._build_processing_graph()
        graph_3 = self.ready_to_speak()

        master = StateGraph(BrainState)

        # Mount compiled subgraphs as discrete nodes
        master.add_node("input_preparation_graph", graph_1)
        master.add_node("processing_graph", graph_2)
        master.add_node("mouth_output_graph", graph_3)

        # Sequential flow across the subgraphs
        master.add_edge(START, "input_preparation_graph")
        master.add_edge("input_preparation_graph", "processing_graph")
        master.add_edge("processing_graph", "mouth_output_graph")
        master.add_edge("mouth_output_graph", END)

        return master.compile()

    # =========================================================================
    # SENSORY ENTRYPOINT
    # =========================================================================

    # invoked by ears.take_input() to run the complete master cognitive graph
    # takes the user prompt as input and returns the final vocalized response text
    def think(self, user_prompt: str) -> str:
        """Invoked by ears.take_input() to run the complete master cognitive graph."""

        print(f"[Brain]: Received user prompt: {user_prompt}", flush=True)
        print("[Brain]: Thinking...", flush=True)

        if not user_prompt or not user_prompt.strip():
            return ""

        initial_state: BrainState = {
            "user_prompt": user_prompt.strip(),
            "system_prompt": "",
            "messages": [],
            "selected_llm": None,
            "raw_response": "",
            "final_response": "",
        }

        # Run the master graph with the initial state and a specific run configuration
        run_config: RunnableConfig = {"run_name": "Atlas-Master-Cognitive-Turn"}
        result = self.graph.invoke(initial_state, config=run_config)
        return result.get("final_response", "")


if __name__ == "__main__":
    # Example usage
    brain = Brain()
    user_input = "Give me a brief summary of the latest advancements in AI research."
    brain.think(user_input)