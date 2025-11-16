from pydantic import BaseModel, Field
from agents import Agent

INSTRUCTIONS = (
    "You are a clarification assistant. Given a user query, come up with 3 clarifying questions "
    "that would help make the query more specific, actionable, or well-defined. "
    "Output ONLY 3 questions, in the order of most to least important."
)

class ClarifyingQuestions(BaseModel):
    questions: list[str] = Field(
        ..., 
        description="A list of 3 clarifying questions in plain English."
    )

clarifier_agent = Agent(
    name="ClarifierAgent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ClarifyingQuestions,
)
