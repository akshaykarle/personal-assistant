from rag import chain

if __name__ == "__main__":
    questions = [
        "What is 2+2?",
        "What are Sahaj core principles?",
        "What is our work from home policy?",
        "How does compensation work?",
    ]

    print(questions[0])
    response = chain.invoke(
        {
            "question": questions[0],
            "chat_history": [],
        }
    )
    print(response)

    follow_up_question = "And what if you add 2 more?"

    print(follow_up_question)
    response = chain.invoke(
        {
            "question": follow_up_question,
            "chat_history": [
                "What is 2+2?",
                "Hello! I'm here to help answer your questions. To answer your revised follow-up question, the result of adding 2 and 2 together is 4"

            ]
        }
    )
    print(response)
