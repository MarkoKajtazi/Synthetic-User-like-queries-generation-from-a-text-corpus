from ollama import chat

def generate_user_queries(summary: str, model: str = "llama3") -> str:
    messages = [
        {"role": "system", "content": (
            "You are an assistant that generates user-style questions "
            "based on the provided knowledge graph in json format "
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
       {
  "nodes": [
    {
      "id": "Professors",
      "type": "Faculty"
    },
    {
      "id": "Professors and TAs",
      "type": "Group"
    },
    {
      "id": "AI assistant",
      "type": "Tool"
    },
    {
      "id": "Modern tech",
      "type": "Technology"
    },
    {
      "id": "Students",
      "type": "Group"
    },
    {
      "id": "AI Design Assistant",
      "type": "Tool"
    },
    {
      "id": "LMS users",
      "type": "Group"
    },
    {
      "id": "LMS features",
      "type": "Functionality"
    },
    {
      "id": "Blackboard",
      "type": "Platform"
    },
    {
      "id": "LMS",
      "type": "Platform"
    }
  ],
  "relations": [
    {
      "source": "LMS",
      "target": "Professors",
      "type": "USED_BY"
    },
    {
      "source": "AI Design Assistant",
      "target": "Blackboard",
      "type": "PROVIDED_BY"
    },
    {
      "source": "LMS users",
      "target": "AI assistant",
      "type": "NEED_SUPPORT_FROM"
    },
    {
      "source": "Professors and TAs",
      "target": "LMS features",
      "type": "NEED_HELP_WITH"
    },
    {
      "source": "Students",
      "target": "AI assistant",
      "type": "NEED_SUPPORT_FROM"
    },
    {
      "source": "LMS",
      "target": "Modern tech",
      "type": "FALLS_SHORT_OF"
    }
  ]
}
"""
    )
    response = generate_user_queries(my_summary)
    print(response)
