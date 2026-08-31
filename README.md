# GOAKNN

This file describes the procedure which should be followed in order to set up this project from scratch, as well the command line interface which can be used to interact with the functionalities this project provides. It should be noted that the project requires a large amount of data to be consumed and stored. In its fully constructed state the database is over 2GB in size, and is stored on disk within the project's directory.

## Dependency Installation

This project was developed using Python 3.13.12, and is only verified to be working on this version. Earlier versions may be compatible but no guarantee is provided. In order to prevent pollution of the system-wide Python installation, it is recommended to create a virtual environment for the project.

```
python -m venv .venv && source .venv/bin/activate
```

The required dependencies are described in the [requirements.txt](requirements.txt) file, and can be installed using Pip.

```
pip install -r requirements.txt
```

### Prerequisite Files

There are two files which are strict prerequisites for this project.

- `go-basic.json` (_*71.2 MB*_) - The Gene Ontology release file available at [https://purl.obolibrary.org/obo/go/go-basic.json](https://purl.obolibrary.org/obo/go/go-basic.json).
- `goa_uniprot_all.gaf.gz` (_*11.6 GB*_) - The Gene Ontology Annotations archive available at [https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz](https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz).

These must be placed in the `data/source` directory, which is relative to the project root directory.

```
mkdir -p data/source
```

### NCBI E-Utilities Configuration

This project interacts with the NCBI E-Utilities API, which asks that users provide a project name and contact email address with each request. In order to conform to this requirement, the `NCBI_PROJECT_NAME` and `NCBI_PROJECT_EMAIL` environment variables should be populated with the appropriate values. Optionally, an API key can be obtained by creating an account with NCBI at [https://www.ncbi.nlm.nih.gov/account/](https://www.ncbi.nlm.nih.gov/account/) and be provided through the `NCBI_API_KEY` environment variable. Providing an API key increases the rate limit for requests from three per second to ten, decreasing the time required to aquire the publication abstracts required by this project.

### Runtime Components

Two components within the project are downloaded automatically at runtime: the NLTK stopwords list and the BiomedBERT model. No manual acquisition of these components is necessary, but it should be noted that an internet connection is required the first time these components are used.

## Command Line Interface

A basic command line interface is included in the project in order to build the database, index the corpus, and execute prediction on input text. The three commands detailed below should be run in the order they are defined, as the database must be progressively constructed before predictions can be made. It is important to note the `-m` flag used in these commands. The scripts in this project must be executed as modules in order to properly resolve dependencies.

```
python -m src.main [command] (options) (target)
```

### Building the Database

```
python -m src.main build ("annotations" | "terms" | "relations" | "citations")
```

This command parses source files and constructs the tables to store the information within them. Each of the four arguments corresponds to a specific table, and certain tables are dependent on each other. Running the command with the argument omitted will build all four tables sequentially, in the correct order.

| Table         | Source                   | Prerequisite  |
|---------------|--------------------------|---------------|
| `terms`       | `go-basic.json`          | None          |
| `relations`   | `go-basic.json`          | None          |
| `annotations` | `goa_uniprot_all.gaf.gz` | None          |
| `citations`   | NCBI E-Utilities         | `annotations` |

The `terms` and `relations` table require `go-basic.json` to exist in `data/source`, while `annotations` requires `goa_uniprot_all.gaf.gz`. The `citations` table is dependent on the contents of the `annotations` table. It should be noted that the three environment variables described above (`NCBI_PROJECT_NAME`, `NCBI_PROJECT_EMAIL`, `NCBI_API_KEY`) are only utilized upon building the `citations` table, and do not need to be defined in any other situation. A simple method to populate the necessary environment variables is by creating a `.env` file and sourcing it before execution.

Building the `citations` table may take a long time since over 200k publication abstracts are downloaded, and so it is recommended to obtain an API from NCBI to speed up this process. Building the `citations` table is a recoverable process, in that it will not download publication abstracts already present in the table. This means that if anything goes wrong, the process can safely be restarted by specifying `citations` in the argument.

#### Examples:

```
> python -m src.main build
> python -m src.main build terms relations
> source .env && python -m src.main build citations
```

### Indexing the Corpus

```
python -m src.main index --method (tfidf | bert)
```

This command performs the indexing of the corpus corresponding to the method provided using the `--method` flag. Two methods are implemented in this project: lexical TF-IDF vectorization and contextual BERT vectorization. If not method is provided, TF-IDF vectorization is selected by default.

Both of the indexes are stored seperately within the database, and the command has different behaviour for each index. Repeated execution of the `index` command with `--method tfidf` will overwrite the existing index. By contrast, the indexing using `--method bert` is a recoverable process, and will only vectorize the documents which are absent from the existing index. This behaviour is necessary as the model inference required to vectorize a document using BERT is time consuming and since each document vectorization is independent of the corpus, it should only ever be vectorized once.

#### Examples:

```
> python -m src.main index
> python -m src.main index --method bert
```

### Predicting Applicable Terms

```
python -m src.main query --method (tfidf | bert) [target]
```

Once the corpus has been indexed using a particular method, prediction can be carried out using that method with this command. As with the `index` command, if not method is specified, TF-IDF vectorization is selected by default. If text is supplied as input in the command line, it should be enclosed in quotation marks (`"`) to prevent parsing errors. Multiple targets can be provided, and predictions will be generated for each.

Output is returned in CSV format with the shape `(Target Index, GOID, Term Label, Score)` and can be redirected to a file. If targets are not provided as arguments, they will be read from `STDIN`. In this case, input must be terminated with the EOF character and target text must be delineated by two consecutive new-line characters (`\n\n`).

#### Examples:

```
> python -m src.main query "Test String 1" "Test String 2"
> python -m src.main query --method bert < input.txt > output.csv
```

## Running the Analysis

```
python -m tests.[hyperparameters | historical] [tfidf | bert] [exact | hierarchical]
```

There are a total of two analysis scripts which are not bundled in the core command line utility: `historical.py` and `hyperparameters.py`. These must also be executed as modules, and take two arguments: the vectorization method and the evaluation metric. These scripts output raw results in JSON format to the `data/cache` directory, which is automatically created when either script runs.

```
python -m tests.util.[visualize | summarize]
```

In order to generate the visualizations and summarized results, there are an additional two scripts which read the raw JSON results: `visualize.py` and `summarize.py`. Both place results in the `data/cache` directory, and the latter also prints to the console. It should be noted that these scripts require all eight possible argument combinations of the `historical.py` and `hyperparameters` scripts to be run. An exception will be thrown if a raw JSON result file is missing.

## Data Usage Disclaimers

The contents of the database are partially aquired and derived from the NCBI E-Utilities service and as such, the disclaimer and copyright notice available at [https://www.ncbi.nlm.nih.gov/About/disclaimer.html](https://www.ncbi.nlm.nih.gov/About/disclaimer.html) applies. The US Library of Medicine does not claim copyright over the abstracts available from PubMed, but journal publishers or authors can. Gene Ontology data from the 2026-06-15 release (DOI:10.5281) is made available under the terms of the CC BY 4.0 license. Gene Ontology Annotations Uniprot archive from the 2026-06-17 release is made available under the terms of the CC BY 4.0 license. This project makes no attempt to distribute the database, and does not recommend users of this project to do so.
