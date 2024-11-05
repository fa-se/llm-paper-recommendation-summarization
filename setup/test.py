import argparse
from datetime import datetime
from core import retrieval, summarization


def main(query):
    # Initialize the start date for the query
    start_date = datetime(2024, 1, 1)

    # Retrieve topics and publications for the query
    topics, publications_retrieved_for_topics = retrieval.initialize_for_query(query=query, start_date=start_date,
                                                                               limit=100, num_topics=10)

    # Get relevant works based on the query
    works = retrieval.get_relevant_works_for_query(query=query, n=5, start_date=start_date)

    # Display the top 5 retrieved works
    print("Top 5 relevant works:")
    for work in works[:5]:
        print(work)

    # Generate summaries for the top 3 retrieved works
    summaries = summarization.summarize_works_for_query(query, works[:3])
    print("\nSummaries for top 3 works:")
    for summarized_work in summaries:
        print(summarized_work)


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Retrieve and summarize works related to a research query.")
    parser.add_argument("query", type=str)

    # Parse arguments and run the main function with the given query
    args = parser.parse_args()
    main(args.query)
