# AI Companion for Clinicians 
AI-Companion that assist clinicians in diagnosis contextualization, critical analysis of heterogeneous docs, and reporting automation to improve efficiency.

## Key Components of Deep Clinician Assistant ##

# Handles extraction of complex document structures:
1) It uses the SOTA data extraction tools to extract the data by analyzing the document structure first, and then extract with data while ensuring it remains grounded in its context.
2) It ensures the context-aware parsing of the content, ensures that extracted data remains grounded in details like placement location on doc, type of content, link to the surrounding content etc.

# Stores multi-modal content in its true form:
1) It stores the multi-modal data like tables and images in jpg format and equations in coded format while ensuring that their related content like table caption or image caption also remains intact with it. 
2) Given the inefficiencies in the approach of converting multi-modal content into plain text, these images of multi-modal content were passed to LLM with respective prompts defined for each type of content, and generated text description for it.
3) While passing the images of multi-modal content, the dynamic mechanism to get surrounding text along with caption was also used to pass contextual information in prompt to ensure LLM generates description in the full context.

# Compilation of chunks for Vector database:
1) Using the specifics (metadata) of the text extracted, the chunk was compiled with complete information required to push data to Milvus database. It contains chunk-id, chunk content, and Metadata of chunk includes details like data type, page, headings etc
2) Vector embeddings of chunks were generated and data was pushed to the Milvus Vector database. Meta data schema associated with embeddings of chunk text was also defined. Indexing technique "HNSW" was used for efficient retrieval.
3) Chunks that contain textual description of multi-modal content were stored in a separate database and textual chunks were stored separately. The reason behind this was just to ensure that text specifically related to multi-modal content remains grounded to their main content. 

# Compilation of entities & relationships in Neo4j:
1) Tested prompt was provided to generate entities and relationships between extracted entities for every chunk separately, and got structured output from LLM for further processing. 
2) One parent entity for every chunk text was created which represents the theme of chunk text, and all the entities discussed in the chunk text were also created as child entities.
3) Belongs_to relationships between parent and child entities were created.
4) Cypher queries for the entities and relationships generation were pushed to generate knowledge graphs in Neo4j.
5) Knowledge graphs for entities extracted from text chunks and entities extracted from multi-modal text description in chunks were separately generated so that query identification can route the search to content of relevant data type during retrieval. 
