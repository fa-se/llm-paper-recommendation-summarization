import typer

from open_alex_interface import search_works

app = typer.Typer()


@app.command()
def search(query: str):
    works = search_works(query)

    for work in works:
        print(work)

if __name__ == "__main__":
    app()