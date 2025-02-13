import os
import json
import discord
from collections import deque
from dotenv import load_dotenv
from model import generate_response  # Import the generate_response function

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

# Memory (stores user conversations)
MEMORY_FILE = "memory.json"
memory = deque(maxlen=10)  # Keep last 10 conversations

# Define Arina's personality
PERSONALITY = """You are Arina, a cheerful and wise AI assistant. You love helping users and chatting with them.
You give **short, relevant, and friendly** answers. You remember past conversations and keep responses **on topic**.
You also learn from user interactions, remembering their preferences and past discussions."""

# Load memory from file
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return deque(json.load(f), maxlen=10)
    return deque(maxlen=10)

# Save memory to file
def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(list(memory), f)

memory = load_memory()  # Load past conversations

# Clear memory
def clear_memory():
    global memory
    memory = deque(maxlen=10)
    save_memory()

@client.event
async def on_ready():
    print(f"✅ Arina is online as {client.user}!")

@client.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == client.user:
        return
    
    print(f"📩 Received message from {message.author}: {message.content}")

    # Clear memory command
    if message.content == "!clear_memory":
        clear_memory()
        await message.channel.send("Memory cleared! 🧹")
        print("🧹 Memory cleared.")
        return
    
    # Check if Arina is mentioned
    if client.user.mentioned_in(message):
        user_input = message.content.replace(f"<@{client.user.id}>", "").strip()

        print(f"🔹 Processed User Input: {user_input}")

        if not user_input:
            response = "Yes? How can I help you? 😊"
        else:
            # Generate response using the model
            response = generate_response(user_input, list(memory))

            # Check if response is empty
            if not response:
                response = "I'm sorry, I didn't understand that. Can you please rephrase?"

            # Debugging final response
            print(f"💬 Arina's Response: {response}")

            # Store response in memory
            memory.append(f"User: {user_input}\nArina: {response}")
            save_memory()  # Save learning progress

        await message.channel.send(response)

client.run(TOKEN)
