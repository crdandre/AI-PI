import os, time
from dotenv import load_dotenv
import dspy
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama


def test_simple_dspy(prompt):
    """Simple test of section identifier with one model."""
    load_dotenv()

    # Initialize LLM
    lm = dspy.LM(
        'openrouter/anthropic/claude-3.5-haiku',
        # 'openrouter/google/gemini-2.0-flash-exp:free',
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        # temperature=0.01,
    )
    
    # Create and run identifier
    with dspy.context(lm=lm):
        respond = dspy.Predict("prompt->response")
        response = respond(prompt=prompt)

    
    # Print results
    print(response)


def test_simple_openrouter(prompt):
    """Simple test of section identifier with one model."""

    # Initialize LLM
    lm = OpenRouter(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )

    return lm.complete(prompt=prompt)


def test_simple_llamaindex_openai(prompt):
    return OpenAI(model="gpt-4o-mini").complete(prompt=prompt)


def test_simple_ollama(prompt):
    model = Ollama(model="llama3.1")
    responses = []
    for _ in range(3):
        start_time = time.time()
        response = model.complete(prompt=prompt)
        end_time = time.time()
        print(f"Response: {response}, Time: {end_time - start_time} seconds")
        responses.append((response, end_time - start_time))


if __name__ == "__main__":
    load_dotenv()
    prompt = "What is 2+2?"
    print(
        test_simple_dspy(prompt),
        # test_simple_openrouter(prompt),
        # test_simple_llamaindex_openai(prompt),
        # test_simple_ollama(prompt),
    )
