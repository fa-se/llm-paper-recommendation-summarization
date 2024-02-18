import pyalex
from pyalex import Works, Work
from os import environ
pyalex.config.email = environ.get("OPENALEX_CONTACT_EMAIL")

def search_works(query: str):
    return (Works()
            .search(query)
            .sort(relevance_score="desc")
            .get())

def work_to_string(work: Work):
    authors = []

    for author in work['authorships'][:3]:
        authors.append(author['author']['display_name'])

    return f"'{work['title']}' by [{', '.join(authors)}] ({work['ids']['openalex']})"