from os import environ

import pyalex

pyalex.config.email = environ.get("OPENALEX_CONTACT_EMAIL")


class Work:
    def __init__(self, pyalex_work: pyalex.Work):
        self.title = pyalex_work['title']
        # First three authors for now
        self.authors = [author['author']['display_name'] for author in pyalex_work['authorships'][:3]]
        self.id = pyalex_work['ids']['openalex']
        # Depending on the query, pyalex may not return a relevance score
        self.relevance_score = pyalex_work.get('relevance_score', None)

    def __str__(self):
        return f"'{self.title}' by [{', '.join(self.authors)}] ({self.id}, relevance score: {self.relevance_score})"


def search_works(query: str) -> [Work]:
    pyalex_works = (pyalex.Works()
                    .search(query)
                    .sort(relevance_score="desc")
                    .get())

    return [Work(pyalex_work) for pyalex_work in pyalex_works]
