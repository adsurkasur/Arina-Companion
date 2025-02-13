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

            # Debugging final response
            print(f"💬 Arina's Response: {response}")

            # Store response in memory
            memory.append(f"User: {user_input}\nArina: {response}")
            save_memory()  # Save learning progress

        await message.channel.send(response)
