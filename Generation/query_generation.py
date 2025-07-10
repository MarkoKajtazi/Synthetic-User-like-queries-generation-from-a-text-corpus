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
These students have at least a couple of years of experience navigating the LMS for assignments, lecture materials, grades, and forums .<n>They typically juggle multiple courses, each with its own information from different platforms and rely on these platforms daily for updates and submissions.
This faculty member could be of any experience level (from a few years in teaching to a senior professor) but is familiar with using the LMS features for course delivery.<n>This sample includes one professor who teaches undergraduate courses and actively uses theLMS to manage class materials and assessments.
Many students feel overwhelmed when juggling multiple courses on different platforms .<n>Instructors feel drained by tedious grading, repetitive tasks make grading feel like a chore rather than an instructional exercise !<n>Students feel supported when they receive prompt, helpful feedback.
Students see multiple tabs and windows: one for the LMS, another for email, maybe a messaging app, etc.<n>Professors see piles of submissions in the queue and maybe spreadsheets they maintain outside the system to track grades, which indicates the platform isn’t serving their collaboration needs .
Professors use LMS to distribute content (syllabi, readings, lectures), communicate announcements, and manage grading .<n>Professors seek to save time on repetitive tasks and ensure fairness and consistency in assessments , many also want better analytics to understand student progress and adjust teaching strategies.
Teaching Assistants (TAs) often act as intermediaries, helping with grading, answering student questions, and handling administrative tasks .<n>They need streamlined grading tools (for homework, exams, etc.)<n> consistency and ease of use in the platform are important.
Students need intuitive navigation, timely feedback, and engaging learning experiences .<n>Many juggle multiple courses and tasks, so clear organization and real-time updates (e.g. deadline reminders) are critical <n>Students also desire easier ways to communicate with peers and instructors online, as organic discussion and collaboration can be challenging in current tools.
This project focuses on designing a more inclusive, AI-enhanced digital space to make it easier for users to find and compare options, make decisions, communicate, grade, and evaluate work .<n>Current LMS and grading systems often fall short. Platforms like Blackboard and Gradescope have been called ‘administrative tools’ that ‘fall short in enhancing student learning’
LMS users often think that the LMS is a necessary hub for their academic work, yet many believe it’s not leveraging modern tech to help them enough .<n> TAs and professors think about fairness and efficiency, for example, “How can I grade all these assignments consistently and in a timely manner?”
Professors and TAs spend significant time doing manual operations .<n>Instructors manually verify that an assignment is published correctly or that grades are calculated as intended, reflecting a lack of confidence in automation ,<n>Students keep their own spreadsheets, or to-do lists to track assignments, essentially doing the job the LMS should be doing in organizing their workload.
We admire Blackboard’s powerful capabilities and new AI initiatives (like goal-tracking and AI tutors)<n>But our design should avoid its complexity. Streamlining is essential – features are only useful if easily accessible.<n>AI features must be integrated in a seamless way (not adding more clicks)
Blackboard now includes an ‘AI Design Assistant’ to help instructors auto-generate course content like modules and assessment questions .<n>It has a comprehensive set of tools (assignments, quizzes, discussions, grade center) and supports complex administrative needs.
can answer questions about the course (e.g. "When is the next quiz?" or "Show me all assignments due next week") using semantic understanding.<n>This could include filtering by type (notes, assignments, discussions) and even searching within uploaded PDFs or videos transcripts.
AI crunches the data and helps instructors prioritize their interventions .<n>For example, AI could suggest students who haven’t logged in in a week. You might want to reach out to them. The data could then be used to suggest supplementary material.
This helps students compare approaches and learn from peers, guided by AI analysis .<n>It could then produce a summary for the class: “Here are 3 common errors in the latest assignment and 2 examples of creative solutions”<n>Instead of sending every minor update as a separate, it could instead send a notification every time a new item is added to the list of tasks.
The LMS could use AI to compile a daily or weekly digest highlighting what’s important for each user .<n>For example: “3 new forum posts in Calculus (2 from your group), 1 assignment feedback released in History (score: 85, with comments) upcoming deadline tomorrow for Lab report”
Key criteria included: Does this solve a real user pain point?<n>Is it technically feasible with AI today? And mainly, is it feasible to implement these features considering the deadline and the knowledge we have obtained until now as undergraduates .<n>After brainstorming, we evaluated them against our research findings and user needs.
It alerts instructors when questions remain unanswered and nudges students to form study groups, while summarizing long threads and auto-generating starter questions to boost engagement .<n>It also uses analytics to highlight what students are doing well and what they need to do to improve their grades, and how to get along with other students in a group.
The assistant would use the course data (syllabus, resources, posts) and perhaps external knowledge to answer .<n>Users (students or faculty) can ask it anything: “Where can I find my assignment feedback for last week?”
We aim to design a next-generation LMS/grading platform that is inclusive, supportive, and efficient.<n>With a human-centered approach and the power of AI, the future of digital learning space can truly empower its users and transform education.
The Engagement Hub would be a section of the LMS that aggregates discussion activity, prompts interactions, and provides insights .<n>An AI mentor could pop into a discussion with a Socratic question if the conversation is going in circles, guiding students to think deeper.
The Grading Studio would allow instructors to view all submissions for a question side-by-side (if desired) and see AI-generated grouping of answers .<n>For quantitative or programming assignments, the AI could auto-grade or flag anomalies (like potential plagiarism or if a student’s answer is unique)
Based on user feedback, we’d ensure its transparent (citing sources for answers, so students trust it) and context-aware (knows which course you’re in, etc.)<n>This assistant could reduce frustration significantly – instead of clicking through menus or waiting for an email reply, users get immediate support .
    """
    )
    response = generate_user_queries(my_summary)
    print(response)
