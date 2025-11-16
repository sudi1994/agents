import asyncio
from clarifier_agent import clarifier_agent, ClarifyingQuestions
from planner_agent import planner_agent, WebSearchPlan, WebSearchItem
from search_agent import search_agent
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from agents import Agent, Runner
from typing import List, Optional

# Instructions for the manager agent
manager_instructions = """
You are a research orchestrator agent. Your job is to answer a user-supplied research query by coordinating clarifications, planning, searching, synthesis, and finally emailing the report using the following tools:

1. clarifier_agent: Use this FIRST to generate 3 clarifying questions based on the initial query.
2. (Simulated) Gather clarifying answers: For each question, write a plausible answer mostly single line. Keep this step concise.
3. planner_agent: Use this NEXT, passing both the original query and the clarifying answers, to produce an effective web search plan (3 search queries, with reasons).
4. search_agent: For EACH of the search terms from the plan, use this tool to perform a web search and receive relevant findings.

At this point handoff results to writer_agent to write final report and send email and return the same report as final output

Workflow:
- Begin by calling clarifier_agent.
- For each clarifying question, simulate a reasonable answer (if user input isn't available).
- With the original query and ALL clarifying answers, plan the searches.
- Perform all planned searches and collect the results.
- handoff the results to writer_agent to write the final report and send an email.
- return the same final report as final output
Follow the above strictly, using ONLY the tools provided. Be thorough and explicit in your handoffs between steps.
"""
clarifier_agent = clarifier_agent.as_tool(tool_name='clarifier_agent' , tool_description='Generate 3 questions for the query')
planner_agent = planner_agent.as_tool(tool_name='planner_agent' , tool_description='Plan the searches')
search_agent = search_agent.as_tool(tool_name='search_agent' , tool_description='Perform relavent search')
# Construct the manager agent with all sub-agents as tools, including email_agent
manager_agent = Agent(
    name="ManagerAgent",
    instructions=manager_instructions,
    model="gpt-4o-mini",
    tools=[clarifier_agent, planner_agent, search_agent],
    handoffs=[writer_agent],
    output_type=ReportData, # Optionally specify if you want strict output
)

# Example usage (for orchestrated workflow; you can run this inside an async context):
async def run_manager_agent(query: str) -> str:
    """Run the manager agent with the specified user query. Returns the markdown research report."""
    result = await Runner.run(manager_agent, query)
    # The manager's instructions ask for a full research report in its final output.
    return str(result.final_output_as(ReportData))

# The file implements a tool-based manager agent and supports report emailing in the research workflow.
