from chatbot.document_loader import load_document
from chatbot.vector_store import create_vector_store
from chatbot.chatbot import create_chatbot, ask_question
import os

def main():
    """
    Main function to run the chatbot.
    """
    # Get the path to the book
    book_path = "/Users/aletinto/Movies/gitRepos/TalkToMyDocsRag/src/book/peter_pan_book.txt"

    # Load the document
    print("Loading document...")
    documents = load_document(book_path)
    print("Document loaded.")

    # Create the vector store
    print("Creating vector store...")
    vector_store = create_vector_store(documents)
    print("Vector store created.")

    # Create the chatbot
    print("Creating chatbot...")
    qa_chain = create_chatbot(vector_store)
    print("Chatbot created. You can now ask questions.")

    # Ask a predefined question
    query = "who is peter pan?"
    print(f"You: {query}")
    answer = ask_question(qa_chain, query)
    print(f"Chatbot: {answer}")

if __name__ == "__main__":
    main()
