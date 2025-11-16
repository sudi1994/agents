import gradio as gr
from dotenv import load_dotenv
from research_manager_v2 import manager_agent
import httpx
from openai import AsyncOpenAI
from agents import set_default_openai_client, Runner

load_dotenv(override=True)
set_default_openai_client(AsyncOpenAI(http_client=httpx.AsyncClient(verify=False)))

# Single async function to handle a research query through the new LLM agent manager
async def run(query: str):
    result = await Runner.run(manager_agent, query)
    # All workflow and formatting is handled by the agent; output presented as markdown
    return result.final_output.markdown_report

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research (Agent-Based)")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")

    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)
