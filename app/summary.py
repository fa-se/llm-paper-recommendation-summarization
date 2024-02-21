from open_alex_interface import Work


def summarize_work(work: Work) -> float:
    # return OpenAlex's relevance score for now
    return work.abstract
