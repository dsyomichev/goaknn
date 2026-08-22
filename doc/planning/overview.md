### Overarching Goal

To build a tool that will identify similarities between text and the descriptions in the Gene Ontology.


### Background

Molecular Biology is now generating colossal amounts of data. In particular, there are a variety of technologies that can scan entire genomes or transcriptomes and determine the likelihood of their association with a particular Biological process. However such results are very noisy and suffer from a "large p, small n" effect, namely that the ratio of possible degrees of freedom, p to the number of independent observations is much greater than 1 and hence suffer from very large false positive rates.

On the other hand, the very extensive body of biomedical literature could be used to filter out spurious genes from the data before undergoing analysis. Tools such as GOCat attempted to mine the literature to relate Gene Ontology terms to specific search terms representing Biological processes. This specific tool is no longer functional and an alternative needs to be provided.

This project will require the student to develop an equivalent of GOCat.

### Early Deliverables

1. A report on Gene Ontologies.
2. A report on the tool GOCat explaining the methods.
3. A report on the data set required to build the tool.

### Final Deliverables

1. The development of a stand alone tool to implement the equivalent of GOCat.
2. An API to query the tool.
3. Rigourous testing of the tool.

### Suggested Extensions

- Using more advanced methods beyond kNN for the implementation.
- Using tools such as LLMs to carry out the same task.
- An API to query the tool.

### Readings

1. **Managing the data deluge: data-driven GO category assignment improves while complexity of functional annotation increases**. _*Gobeill J Pasche E Vishnyakova D Ruch P*_ (DOI 10.1093/database/bat041)
2. **A survey on annotation tools for the biomedical literature.** _*Neves M Leser U*_ (DOI 10.1093/bib/bbs084)
3. **The GOA database in 2009--an integrated Gene Ontology Annotation resource.** _*Barrell D Dimmer E Huntley R Binns D O'Donovan C Apweiler R*_ (dx.doi.org/10.1093/nar/gkn803)

### Prerequisites
1. Understanding of machine learning methods such as kNN.
2. The student must have a good understanding of R and/or Python.