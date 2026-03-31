# CTF Agent

Autonomous CTF solver that runs **three OpenRouter models in parallel** on whatever you put in a folder. **You do not need a YAML file** — optional `metadata.yml` is still supported if you want points/category metadata.


## Setup

1. Install: `pip install -e .`
2. Build the Docker sandbox: `docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .`
3. Create a `.env` file with:

   ```env
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

## Run

Put your challenge in a directory:

| What | Where |
|------|--------|
| Challenge statement | `challenge.txt`, `description.txt`, `README.md`, or `challenge.md` (first one found wins) |
| Hints | `hints.txt`, `hint.txt`, `hints.md`, or a `hints/` folder with text files |
| Attachments | Any files anywhere in the folder (or use a `distfiles/` subfolder — both work) |
| Remote service | Optional `connection.txt` with `nc ...` or a URL |

Then:

```bash
ctf-solve path/to/that/folder
```

If the folder is named `challenge` in the current directory, you can run:

```bash
ctf-solve
```

The agent always uses these three models together:

- `qwen/qwen3.6-plus-preview:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `stepfun/step-3.5-flash:free`

When a flag is found, it prints in the terminal. You submit it to the competition site yourself if there is one.

### Optional: many challenges at once

```bash
ctf-solve --watch path/to/parent
```

Each **subfolder** of `parent` is one challenge. A coordinator process manages them (Ctrl+C to stop). This is only needed if you are juggling multiple challenges; most of the time use a single folder and `ctf-solve` as above.

### Dry run

```bash
ctf-solve ./my-challenge --no-submit
```

## Optional `metadata.yml`

If you already have a `metadata.yml` (points, category, extra hints), it is merged in. Plain folders without YAML work fine.

## Requirements

- Python 3.13+
- Docker
- OpenRouter API key

## Acknowledgements

- [Pydantic AI OpenRouter](https://ai.pydantic.dev/models/openrouter/)
