from pydantic import BaseModel, Field
from agents import Agent
from email_agent import email_agent

INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report which is lengthy and detailed.\n"
    "The Report should be in markdown format."
    "Aim for 5-10 pages of content, at least 1000 words then, Use your tool email_agent_tool  exactly once to send email to the user."
    "finally return the report as final output."
)


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")

    markdown_report: str = Field(description="The final report")

    follow_up_questions: list[str] = Field(description="Suggested topics to research further")

email_agent_tool = email_agent.as_tool(tool_name='email_agent_tool', tool_description='used to format the report and send email')
writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    tools=[email_agent_tool],
    handoff_description='Used to outline and prepare final report from the input details and finally send it via email',
    output_type=ReportData,
)