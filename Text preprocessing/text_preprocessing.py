from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
import torch

def summarize(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=60,
        min_length=20,
        no_repeat_ngram_size=3,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    paragraphs = ["""
        University learning management systems (LMS) and grading platforms are central to modern higher education, but many are criticized for poor design and inefficiency. The integration of Artificial Intelligence (AI) offers a chance to transform these tools. AI-powered LMS platforms can personalize learning, automate tasks, and provide predictive insights. For example, AI can adapt content to each learner, auto-grade assignments, and highlight at-risk students early. Such capabilities promise a more efficient, engaging, and tailored learning experience than traditional systems. Despite these opportunities, current LMS and grading systems often fall short. Platforms like Blackboard and Gradescope have been called “administrative tools” that “fall short in enhancing student learning”. Students and faculty alike report frustrations: clunky navigation, information overload, and inefficient workflows. This project focuses on designing a more inclusive, AI-enhanced digital space to make it easier for users to find and compare options, make decisions, communicate, grade, and evaluate work.
    """, """
        The primary users are university students, professors (instructors), and teaching assistants (TAs). Each group interacts with LMS and grading tools in different ways. Students use LMS platforms to access course materials, submit assignments, check grades, and collaborate. They need intuitive navigation, timely feedback, and engaging learning experiences. Many juggle multiple courses and tasks, so clear organization and real-time updates (e.g. deadline reminders) are critical. Students also desire easier ways to communicate with peers and instructors online, as organic discussion and collaboration can be challenging in current tools.
    """]

    device = "cpu"
    model_name = "google/pegasus-cnn_dailymail"

    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.to(torch.device(device))

    print(summarize(" ".join(paragraphs), tokenizer, model))

if __name__ == "__main__":
    main()
