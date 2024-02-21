import os

from openai import OpenAI

from open_alex_interface import Work

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
seed = 42  # for reproducibility


def summarize_work(work: Work, query: str) -> str:
    completion = client.chat.completions.create(
        # gpt-3.5-turbo is an alias and the model it resolves to might change over time. consider specifying an explicit model name for reproducibility
        # https://platform.openai.com/docs/models/gpt-3-5-turbo
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Your task is to write succinct summaries of scientific abstracts."
                           f"The summary should help the reader understand whether and how the work relates to their search query: '{query}'."
                           "The summary should be about 100 words long."
            },
            {"role": "user", "content": work.abstract}
        ],
        seed=seed
    )
    print(f"Used tokens: {completion.usage.total_tokens}")
    return completion.choices[0].message.content
