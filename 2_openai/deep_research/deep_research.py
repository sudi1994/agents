import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager
import httpx
from openai import AsyncOpenAI
from agents import set_default_openai_client


load_dotenv(override=True)
set_default_openai_client(AsyncOpenAI(http_client=httpx.AsyncClient(verify=False)))


async def run(query: str):
    async for chunk in ResearchManager().run(query):
        yield chunk


with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What topic would you like to research?")
    run_button = gr.Button("Run", variant="primary")
    report = gr.Markdown(label="Report")
    
    run_button.click(fn=run, inputs=query_textbox, outputs=report)
    query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

ui.launch(inbrowser=True)

