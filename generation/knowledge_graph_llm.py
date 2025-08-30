import asyncio
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.llms.ollama import Ollama
from generation.queries_generation import generate_user_queries

# 1) set up your LLM & transformer once
llm = Ollama(model="llama3", verbose=True)
graph_transformer = LLMGraphTransformer(llm=llm)


# 2) utility to extract a graph from one text chunk
async def _extract_one(text: str):
    docs = [Document(page_content=text)]
    # returns a list of GraphDocument, but we only ever
    # gave it one Document, so take [0]
    return (await graph_transformer.aconvert_to_graph_documents(docs))[0]


# 3) batch‐extract graphs for N chunks in parallel
async def extract_graphs(texts: list[str]):
    tasks = [_extract_one(t) for t in texts]
    return await asyncio.gather(*tasks)  # -> list[GraphDocument]


# 4) merge a list of GraphDocument into one
def merge_graph_documents(graph_docs):
    # simple stand‐in in case you can't import the real GraphDocument class
    class MergedGraph:
        def __init__(self, nodes, relationships):
            self.nodes = nodes
            self.relationships = relationships

    node_map = {}  # id -> node
    rel_set = set()  # (src, type, tgt)
    rel_list = []

    for doc in graph_docs:
        # merge nodes
        for n in doc.nodes:
            if n.id not in node_map:
                node_map[n.id] = n
            else:
                # optional: merge metadata dicts
                if getattr(n, "metadata", None):
                    existing = node_map[n.id]
                    existing.metadata = {
                        **(existing.metadata or {}),
                        **n.metadata
                    }
        # merge relationships
        for r in doc.relationships:
            key = (r.source.id, r.type, r.target.id)
            if key not in rel_set:
                rel_set.add(key)
                rel_list.append(r)

    # return a lightweight object with .nodes and .relationships
    return MergedGraph(list(node_map.values()), rel_list)


# 5) bring it all together
def build_and_merge(text_chunks: list[str]):
    graph_docs = asyncio.run(extract_graphs(text_chunks))
    merged = merge_graph_documents(graph_docs)
    return merged


# — example usage —
if __name__ == "__main__":
    chunks = [
        """
         University learning management systems (LMS) and grading platforms are central to 
         modern higher education, but many are criticized for poor design and inefficiency. The 
         integration of Artificial Intelligence (AI) offers a chance to transform these tools. AI-
         powered LMS platforms can personalize learning, automate tasks, and provide predictive 
         insights. For example, AI can adapt content to each learner, auto-grade assignments, and 
         highlight at-risk students early. Such capabilities promise a more efficient, engaging, and 
         tailored learning experience than traditional systems. Despite these opportunities, current 
         LMS and grading systems often fall short. Platforms like Blackboard and Gradescope 
         have been called “administrative tools” that “fall short in enhancing student learning”. 
         Students and faculty alike report frustrations: clunky navigation, information overload, and 
         inefficient workflows. This project focuses on designing a more inclusive, AI-enhanced 
         digital space to make it easier for users to find and compare options, make decisions, 
         communicate, grade, and evaluate work.""",
                """
        Students use LMS platforms to access course materials, submit assignments, check 
        grades, and collaborate. They need intuitive navigation, timely feedback, and engaging 
        learning experiences. Many juggle multiple courses and tasks, so clear organization and 
        real-time updates (e.g. deadline reminders) are critical. Students also desire easier ways 
        to communicate with peers and instructors online, as organic discussion and collaboration 
        can be challenging in current tools.""",

        """Professors use LMS to distribute content (syllabi, readings, lectures), communicate 
        announcements, and manage grading. They value efficient content organization and 
        reliable delivery of materials. Grading is a major effort. Professors seek to save time on 
        repetitive tasks and ensure fairness and consistency in assessments. Many also want 
        better analytics to understand student progress and adjust teaching strategies.
        Teaching Assistants (TAs) often act as intermediaries, helping with grading, answering 
        student questions, and handling administrative tasks. They need streamlined grading 
        tools (for homework, exams, etc.) and a clear overview of student submissions. TAs 
        frequently switch between student and instructor perspectives, so consistency and ease 
        of use in the platform are important.""",
        """3 undergraduate students – We include three undergraduate students who are in the 
        middle of their degree programs and use the LMS regularly for their courses. These 
        students are tech-savvy and have at least a couple of years of experience navigating the 
        LMS for assignments, lecture materials, grades, and forums. They typically juggle multiple 
        courses, each with its own information from different platforms, and they rely on these 
        platforms daily for updates and submissions.""",

        """One professor – our sample includes one professor who teaches undergraduate courses 
        and actively uses the LMS to manage class materials and assessments. This faculty 
        member could be of any experience level (from a few years in teaching to a senior 
        professor) but is familiar with using the LMS features for course delivery.
        Think: Users often think that the LMS is a necessary hub for their academic work, yet 
        many believe it’s not leveraging modern tech to help them enough. Students think about 
        staying on top of deadlines and worry “Am I missing any assignment updates?” TAs and 
        professors think about fairness and efficiency, for example, “How can I grade all these 
        assignments consistently and in a timely manner?” Both groups have an underlying 
        expectation that the technology should simplify their lives, not complicate it. There’s also 
        a concern about transparency – students want insight into the factors contributing to their 
        success, and instructors want to ensure students see the rationale behind grades.
        Feel: Emotions run high, especially frustration and stress. Many students feel 
        overwhelmed when juggling multiple courses on different platforms; as one put it, using 
        the LMS can feel like “too many clicks to find one thing”. Instructors feel drained by tedious 
        grading workflows; repetitive tasks make grading feel like a chore rather than an 
        instructional exercise. There is also frustration with poor communication tools – for 
        instance, students feel isolated when discussion boards are ghost towns or when their 
        question goes unanswered. On the positive side, students feel supported when they 
        receive prompt, helpful feedback. Notably, there is some hope and curiosity around AI – 
        a few students were excited by the idea that “an AI could give me quick feedback before 
        the professor even grades it,” while others feel skeptical or worried about AI fairness.
        See: Users see a fragmented digital environment. Students see multiple tabs and 
        windows: one for the LMS, another for email, maybe a messaging app, etc., because no 
        single tool covers all their needs. They see peers forming ad-hoc group chats outside the 
        official platform, which indicates the LMS isn’t serving their collaboration needs. 
        Professors see piles of submissions in the queue and maybe spreadsheets they maintain 
        outside the system to track grades. A surprising observation: some instructors still see 
        physical paper printouts or manual processes alongside the LMS (e.g. printing out 
        summaries of online quizzes) because they find it easier to review or compare. In terms 
        of content, users see a lot of text-heavy pages – long lists of resources or forum posts – 
        which can be daunting. They also see inconsistency; each professor’s course might look 
        different (navigation items, where to find things), causing confusion when students take 
        multiple courses. This visual and organizational inconsistency was a major pain point 
        mentioned in interviews. They also see opportunities: for example, TAs noticed patterns 
        (like the same question asked by many students) and thought “the system should catch 
        this.” Overall, what they see is potential for a more unified, smarter system versus the 
        siloed, cluttered interfaces they have now.""",

        """Do: Users take various actions to cope with or workaround current limitations. They 
        frequently search their email for keywords because it’s faster to find a message about an 
        assignment than to locate it in the LMS – a clear sign search in LMS is lacking. Students 
        also keep their own spreadsheets, or to-do lists to track assignments, essentially doing 
        the job the LMS should be doing in organizing their workload. Professors and TAs spend 
        significant time doing manual operations: exporting grades, copying feedback from one 
        place to another, or even writing the same comment on 30 submissions individually. On 
        the communication side, if a student has a question, what they do is often skip the LMS 
        messaging and send an email or a chat message, indicating the LMS is lacking the feature 
        of a built-in messaging feature. Instructors “do” a lot of double-checking – they manually 
        verify that an assignment is published correctly or that grades are calculated as intended, 
        reflecting a lack of confidence in automation.""",

        """Good features: It has a comprehensive set of tools (assignments, quizzes, discussions, 
        grade center) and supports complex administrative needs. Recent updates by Anthology 
        (Blackboard’s parent company) are integrating generative AI in novel ways. For example, 
        Blackboard now includes an “AI Design Assistant” to help instructors auto-generate 
        course content like modules and assessment questions.
        Key takeaway: We admire Blackboard’s powerful capabilities and new AI initiatives (like 
        goal-tracking and AI tutors), but our design should avoid its trap of complexity. 
        Streamlining workflows is essential – features are only useful if easily accessible. AI 
        features must be integrated in a seamless way (not adding more clicks). Blackboard’s 
        experience underscores the importance of user-centric design even in a feature-rich 
        system.
        can answer questions about the course (e.g. "When is the next quiz?" or "Show 
        me all assignments due next week") using semantic understanding. This could 
        include filtering by type (notes, assignments, discussions) and even searching 
        within uploaded PDFs or videos transcripts.""",

        """Personal AI Study Coach (Chatbot): An assistant students can chat with on the
        for an assignment and identifies common mistakes or exemplary answers. It could 
        then produce a summary for the class: “Here are 3 common errors in the latest 
        assignment and 2 examples of creative solutions.” This helps students compare 
        approaches and learn from peers, guided by AI analysis. """,

        """Smart Notification Digest: Instead of sending every minor update as a separate
        notification, the LMS could use AI to compile a daily or weekly digest highlighting 
        what’s important for each user. For example: “3 new forum posts in Calculus (2 
        from your group), 1 assignment feedback released in History (score: 85, with 
        comments), upcoming deadline tomorrow for Lab report.” This reduces noise and 
        helps users make decisions on what to address first.""",

        """Engagement Hub: An AI-powered section of the LMS that aggregates discussion
        activity and prompts interactions. It alerts instructors when questions remain 
        unanswered and nudges students to form study groups, while summarizing long 
        threads and auto-generating starter questions to boost engagement. """,

        """Decision Dashboard for Instructors: A dashboard that uses analytics to highlight
        things like “Which topic did students struggle with most on the exam? (with an AI-
        generated suggestion of supplementary material)” or “These 5 students haven’t 
        logged in for a week, you might want to reach out.” Essentially, AI crunches the 
        data and helps instructors prioritize their interventions.
        We generated dozens of ideas like these. After brainstorming, we evaluated them 
        against our research findings and user needs. Key criteria included: Does this solve a 
        real user pain point? Will users likely adopt this? Is it technically feasible with AI today? 
        Does it align with the goal of a more inclusive, efficient space? And mainly, is it feasible 
        to implement these features considering the deadline and the knowledge we have 
        obtained until now as undergraduates.
        several related concepts into one cohesive assistant accessible throughout the 
        platform. Users (students or faculty) can ask it anything: “Where can I find my 
        assignment feedback for last week?”, “What’s the average score on the midterm?”, 
        or even “Explain the concept of binary search from class, I’m stuck.” The assistant 
        would use the course data (syllabus, resources, posts) and perhaps external 
        knowledge to answer. We chose this as a top idea because it directly tackles
        finding info, decision support, and communication. It acts as a friendly front-end to 
        the complexity, aligning with the insight that users want powerful help but in a 
        simple way. Based on user feedback, we’d ensure its transparent (citing sources 
        for answers, so students trust it) and context-aware (knows which course you’re 
        in, etc.). This assistant could reduce frustration significantly – instead of clicking 
        through menus or waiting for an email reply, users get immediate support. It 
        essentially augments the LMS with a 24/7 virtual TA for everyone. """,

        """Intelligent Grading Studio – AI-augmented grading interface: This is aimed at
        professors and TAs to make grading faster and more consistent. The Grading 
        Studio would allow instructors to view all submissions for a question side-by-side 
        (if desired) and see AI-generated grouping of answers (e.g. 10 students made the 
        same mistake in question 2). It would also suggest feedback comments: for 
        instance, “These answers all missed mentioning X; suggest comment: ‘Remember 
        to include X in your answer.’” Instructors remain in control – they can accept, edit, 
        or ignore the suggestions – but it saves them typing the same comment repeatedly. 
        For quantitative or programming assignments, the AI could auto-grade or flag 
        anomalies (like potential plagiarism or if a student’s answer is unique). We selected 
        this idea because grading pain came up in interviews, and Gradescope’s success 
        validates the impact.""",

        """Engagement Hub – AI-driven communication and analytics center: This idea
        addresses the community and engagement aspect. The Engagement Hub would 
        be a section of the LMS that aggregates discussion activity, prompts interactions, 
        and provides insights. For example, it might highlight: “No one has answered 
        Alice’s question in her email for 2 days. (Click to answer or summon AI help)” for 
        instructors, or for students: “5 classmates are interested in forming a study group 
        for Project 1 – join them.” It could use AI to summarize long discussion threads to 
        encourage students to participate without reading everything, and even auto-
        generate starter questions if a forum is inactive (like an icebreaker related to 
        course content). Another feature: an AI mentor could pop into a discussion with a 
        Socratic question if the conversation is going in circles, guiding students to think 
        deeper. We chose this idea because it targets the less tangible but vital aspect of 
        engagement and communication.""",

        """In conclusion, this research-based design report captures an iterative journey: from 
        understanding user frustrations and needs, through analyzing current tools, to defining 
        key opportunities and ideating innovative solutions. By leveraging AI thoughtfully, we aim 
        to design a next-generation LMS/grading platform that is inclusive, supportive, and 
        efficient, where students, professors, and TAs can find information quickly, make informed 
        decisions, communicate effortlessly, and grade or receive feedback in a fair and timely 
        manner. With a human-centered approach and the power of AI, the future of digital 
        learning space can truly empower its users and transform education."""
        # …etc, split your huge text however you like…
    ]

    merged_graph = build_and_merge(chunks)

    # now you can visualize it:
    from pyvis.network import Network

    net = Network(height="800px", width="100%", directed=True)
    for n in merged_graph.nodes:
        net.add_node(n.id, label=n.id, title=n.type)
    for r in merged_graph.relationships:
        net.add_edge(r.source.id, r.target.id, label=r.type)
    net.save_graph("knowledge_graph.html")