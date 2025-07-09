from ollama import chat

def generate_user_queries(summary: str, model: str = "llama3") -> str:
    messages = [
        {"role": "system", "content": (
            "You are an assistant that generates user-style questions "
            "based on the provided summary "
            "the output should only be the questions."
        )},
        {"role": "user", "content": f"Summary:\n{summary}\n\nQuestions:"}
    ]

    response = chat(model=model, messages=messages)
    return response.message.content.split("\n")

# Example usage
if __name__ == "__main__":
    my_summary = (
        """
        This project focuses on designing a more inclusive, AI-enhanced digital space to make it easier for users to find and compare options, make decisions, communicate, grade, and evaluate work .<n>The primary users are university students, professors (instructors), and teaching assistants (TA).
        """
    )
    response = generate_user_queries(my_summary)
    print(response)
