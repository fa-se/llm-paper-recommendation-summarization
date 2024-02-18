import typer

import open_alex_interface

app = typer.Typer()


@app.command()
def search(query: str):
    works = open_alex_interface.search_works(query)

    for work in works:
        print(open_alex_interface.work_to_string(work))

if __name__ == "__main__":
    app()