from open_alex_interface import Work


def rate_relevance(work: Work) -> float:
    # return OpenAlex's relevance score for now
    return work.oa_relevance_score
