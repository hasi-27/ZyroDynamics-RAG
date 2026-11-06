
from rag_pipeline import ask_bot

# Questions to test the Zyro Dynamics HR chatbot
test_questions = [
    "How many casual leaves do I get per year?",
    "Are laundry expenses reimbursable for a trip lasting exactly four nights?",
    "Who approves travel budgets above Rs. 50,000?"
]

# Run each question through the chatbot
for i, question in enumerate(test_questions, 1):

    print("\n" + "=" * 70)
    print(f"QUESTION {i}: {question}")

    result = ask_bot(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    if result["sources"]:
        for doc in result["sources"]:
            print(
                f"File: {doc.metadata.get('source')}"
                f" | Page: {doc.metadata.get('page')}"
            )
    else:
        print("No sources returned.")

    print("=" * 70)

print("\nAll test questions completed!")
