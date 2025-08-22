from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model = model,
    model_provider=external_client,
    tracing_disabled=True
)

quizmaster_agent = Agent(
    name="QuizMaster Agent",
    instructions="""
You are QuizMaster Agent. Generate quizzes on ANY topic (Math, Science, History, Programming, etc.).

Guidelines:
1. Always generate ONE multiple-choice question at a time (4 options).
2. Wait for the user to answer before revealing the correct answer.
3. After showing the correct answer and explanation, generate the NEXT question.
4. After the final question, give total score + tips for improvement.
5. Keep questions concise, clear, and beginner-friendly unless level is specified.
"""
)

@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content=
        "👋 Hello! I'm your **AI QuizMaster Agent**.\n\n"
        "You can ask me for a quiz on **any subject or topic** — Math, Science, History, Technology, Coding, and more!\n\n"
        "✨ Here's what I can do for you:\n"
        "• Create short quizzes (3–5 questions)\n"
        "• Adjust difficulty (Beginner, Intermediate, Advanced)\n"
        "• Give you instant feedback & correct answers\n"
        "• Share tips on what to review if you make mistakes\n\n"
        "📌 Just type a topic like *'Fractions'*, *'Photosynthesis'*, or *'Python Loops'* and I'll generate a quiz for you!"
    ).send()
@cl.on_message
async def handle_message(message : cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    
    msg = cl.Message(content="")
    await msg.send()
    result = Runner.run_streamed(
        quizmaster_agent,
        input=history,
        run_config=config
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
            
    history.append({"role": "assistant", "content":result.final_output})
    cl.user_session.set("history", history)