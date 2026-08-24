# Research Report

## The Gene Ontology

The Gene Antology is hierarchical vocabulary aimed at categorizing and standardizing biological terminology, particulary with regard to the properties of gene products. [1] Structuraly, it is a set of Directed Acyclic Graphs (DAGs), with specificity increasing proportionally with distance from the root node. A node represents a term, with relations represented by edges of the graph. [2]

The hierarchy is not strict, in the sense that a child term may have more than one parent.

### The Three Aspects

Within the ontology, there are three seperate hierarchies, officially called "aspects". These are seperate root nodes of a DAG, representing different elements of a gene process.

- Molecular Function
- Cellular Component
- Biological Process

The Molecular Function aspect represents an activity performed by a gene product, which is a biochemical action. These can involve any number of gene products. With more complex processes, a single gene product may not be enough to descbe the process.

The Cellular Components aspect refers to the location in which a molecular function takes place, for example the plasma membrane. There are various subcategories within this aspect, such as virion components, which are seperate due to the non-cellular nature of virus organisms.

The Biological Procesess aspect is a description of the overarching "program" used to describe a set of molecular actions. An example of such a process would be DNA repair.

These aspects are disjoint, in the sense that no term which appears in one will appear in another. This means that certain relations cannot be attributed to terms in seperate aspects. [1]

### Relations and Structure

There are four key relations between terms within the ontology, each of which can be represented as an edge in the DAG. These relationships can potentially be used for grouping. The following definitions use parent term $P$ and child term $C$. The $\vDash$ operator is used to depict the specific relation and can be substituted with the name of the relation.

- `is a` : $C \subseteq P$
- `part of` : $P \Leftrightarrow C$
- `has part` : $P \Rightarrow C \land \neg (C \Rightarrow P)$
- `regulates` : $(P, C \Rightarrow P \vDash C) \land \neg (C \Rightarrow P \vDash C)$

The `is a` relation represents a subtyping of a term, in the sense that a child term is subtype of it's parent. This relation can be used for grouping by parent type, such that $A \coloneq \forall C \subseteq P$.

The `part of` relation defines a part-whole relationship between terms, such that the presence of either parent or child implies the presence of the other. This relation can also be used for grouping terms.

The `has part` relation also defines a part-whole relationship between terms, but the presence of the child term does not imply the presence of the parent. All $P$ have $C$ but not all $C$ are part of $P$. This relation cannot be used for grouping, since not all $C$ are part of $P$.

The `regulates` relation is slightly different in that it describes situations in which the parent $P$ directly affects the child $C$, if both are present. But this relation does not imply that $P$ always regulates $C$. Using this relation for grouping is not always correct, but within certain contexts, it is viable. [2]



### Term Elements

Each term within the ontology has a set of properties which describe the term, and serve to relate it to other terms within the ontology. Importantly, each term has a unqiue identifier, starting with `GO:`.

The term also must contain it's name, which aspect it belongs to, a definition, and the relationships it holds to other terms.

There are also a number of optional properties which may be included, such as synonyms, obsoletion status, and comments.

With regards to the overall structure of the DAG, the term object is the node, with the relationships of a term describing the connecting edges. [1]


```
accession: string # Unique identifier of the term.
name: string # Plaintext name of the term.
ontology: 'molecular_function' | 'cellular_component' | 'biological_process' # Aspect the term belongs to.
definition: string # Definition of the term.
relationships: Relationship[]
```

The above is a basic depiction of the term structure.

### Uses and Annotations

The functional purpose of the Gene Ontology is to homogenate the various terms used to describe gene products and the processes they play a role in. It aims to prevent the sprawl of vocabulary used by different research groups and laboratories.


_*The Gene Ontology (GO) provides a framework and set of concepts for describing the functions of gene products from all organisms.*_ [3]

In practice, the ontology is used in a process known as annotation; which is essentially the tagging of biomedical text based on the terms used or alluded to within the text. This process allows for easier automated lookup and digestion of research, even in cases where the terminology is not directly referenced or utilized. [4]

This is a largely manual process, requiring significant oversight and a deep knowledge base. A number of computational methods have been developed in order to speed up the process by seeding relavent annotations for a paper to be reviewed by experts, but ultimately, neither the human nor the computer are guarenteed to be correct in every case. As such, the central goal of a computational annotation algorithm is to provide a baseline from which the expert annotator can work from to create a full annotation set, which can then be refined for accuracy over time.

## GOA Dataset Overview

The Gene Ontology Annotations (GOA) project, provided from the European Bioinformatics Institute (EBI) is a comprehensive collection of GO annotations for gene protien databases such as UniProtKB. Using a mix of manual and computed (electronic) annotations, the project uses a set of pipelines to automatically produce large quantities of annotations which are released on a monthly basis. [5] Using UniProtKB as the standard information base provides benefits in that a number of annotations can be attributed to lesser studied species, which may require larger quantities of automated annotations.

Each gene product within the dataset is attributed to the most applicable and precise term, allowing comparisons between the terminlogy appearing in biomedical texts and the synonymous GO terms.

The GOA project assembles it's datasets from both manual and automated sources, both of which are cross-referenced from first and third party sources. While the growth of the project is accelerating in terms of the quantity of annotations, the dataset only represents a small fraction of the available literature. [6]

## GOCat Implementation

The GOCat project, developed by the University of Geneva, was a program which automatically assigns applicable GO terms to general text input, utilizing a KNN algorithm under the hood to provide a ranked list. The machine learning approach resulted in a significant increase in accuracy over dictionary based approaches. [6]

The similarity metric for the KNN algorithm was based on semantic meaning of text, rather than a quantifiable metric such as the number of matching words. This provided a more nuanced term output, and this higher accuracy results.

### GOA in GOCat

The GOCat project used several releases of the GOA database, and by associating the GO terms to biomedical papers, the model could be trained on a dataset consisting of paper samples with GO term labels. In order to benchmark the model, the researchers used the database releases over multiple years and assessed the accuracy for a given year by using that years release as a testing set.

### The Terrier Platform

In order to index the vast quantity of papers and articles Gobeill et al. used the Terrier Platform, and open source project which provides a framework for document retrieval systems. Within the use case of the project, the platform provided a distance/similarity metric for the KNN algorithm, returning the most similar articles to the input text. The platform provides a highly configurable indexer and a set of information retrieval tools, which together provide the functionality used within GOCat.

When a document is indexed, it can be normalized in a customizable pattern including tokenization, stop word removal, or stemming. The GOCat project used Porter Stemming, which strips suffixes from words. [8]

#### BM25 for Matching

The platform itself can utilize a number of weighing models for information retrieval [7], but the GOCat project used the BM25 algorithm. [6] This algorithm provides a score for a document based on the term frequency (TF), inverse document frequency (IDF) and document length (N). The formula describing the score

$SCORE(q, d) = \sum_{t \in q}IDF(t)*TF(t, d)$

utilizes normalized TF and IDF values based on the documents length. [9] Using this algorithm, the terrier platform can provide a ranked list of most applicable documents within the indexed store.

#### Matching Terms

In order to provide a list of ranked terms, the GOCat project assembled the list of matching documents and the GOA annotations associated with them. By weighing each term by the similarity score of the document it appeared in, and then taking the sum of the scores across each appearing term, a ranked list can be created, which is the list of most applicable GO terms.

# Project Plan

The high-level overview of the project is to replace the Terrier Platform used in GOCat with a text embeddings model, such as PubMedBERT or BioSentVec. As such, the hypothesis is that text embedding models are capable of capturing even greater semantic/contextual information about the training data and arbitrary text input. While more computationally intense, this should not be a significant factor since once a sample has been ingested, it is ultimately represented as a single $p$ dimensional vector.

The first step would be to aquire and pre-process the training paper abstracts, which can be downloaded from PubMed using a web query API. [10] These papers will then be split into training/testing sets when necessary.

In order to map each sample to a singular vector, an aggregation strategy will be required. This will be finalized during the implementation stage of the project, but may involve several strategies such as taking the mean of all vectors. This strategy may be different depending on the model, since PubMedBert uses word-level embeddings while BioSentVec uses sentence-level embeddings. Once vectorized, each sample will be stored alongside the annotations from GOA, both of which compose the finalized index of samples.

In order to compare test samples to the index, it will be vectorized following the same process as the training set, and then compared to each sample using cosine similarity. The top $k$ samples will be collected and the annotations will be aggregated. The annotation quantity will be weighed against the similarity, and then the list will be returned as output. This process will be functionally identical to GOCat, with the only major change being the similarity score being derived differently.

Testing will also be functionally identical to GOCat, where the data set will be split into training/testing based on year, and the overall accuracy will be calculated.

Once these components have been implemented, a command line interface or web ui will be a simple addition to just call the top-level function, and return its output.

## Required Components

1. Data Sets
    a. Gene Ontology
    b. Gene Ontology Annotations
2. Text Embedding Models
    a. BioSentVec
    b. PubMedBERT
3. Algorithmic Implementations
    a. Cosine Vector Similarity
    b. Embedding Vector Aggregation
4. User Facing Implementation
    a. Command Line Tool
    b. Browser-based UI

## Risk Factors

1. **Device Failure** _*(Low, Low)*_ - Documents stored in OneDrive, code stored in GitLab.
2. **Unplanned/Underestimated Programatic Complexity** _*(Medium, Medium)*_ - More research or optimization may be required.
3. **Unrealistic Cost** _*(Low, High)*_ - If the models take an unrealistic amount of time/resources to infer embeddings, it may be necessary to purchase compute from a cloud provider. This will not be a significant risk, as training a new model is not within the scope of the project.

# Refrences

1. https://www.geneontology.org/docs/ontology-documentation/
2. https://www.geneontology.org/docs/ontology-relations/
3. https://www.ebi.ac.uk/ols4/ontologies/go
4. https://pmc.ncbi.nlm.nih.gov/articles/PMC2367625/
5. https://pmc.ncbi.nlm.nih.gov/articles/PMC2686469/
6. Gobeill et al. (2013)
7. https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_retrieval.md
8. https://www.geeksforgeeks.org/nlp/porter-stemmer-technique-in-natural-language-processing/
9. https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/
10. https://pubmed.ncbi.nlm.nih.gov/download/