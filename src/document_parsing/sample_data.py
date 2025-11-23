





                                            #### textual_chunk_payload_prep final output ####     
{'page_no.': 0, 'index_on_page': 1, 'raw_content': 'RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK', 'content_type': 'title', 'doc_id': '12e54263-17eb-4436-97cc-c15b81a2fb21', 'chunk_id': '12e54263-chunk-1', 'document_title': 'RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK'}                                 


                                            ####  Context Extraction Module Results   ####
Multi_model_chunks_with_llm_description = [{'page_no.': 6, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\14861407bef7f68b31affea722351d2101dcf3994e2fdd92f50524cbefb3411b.jpg', 'content_type': 'table', 'contextual_text': ['Effective multimodal question answering requires preserving rich visual semantics while maintaining coherent grounding across heterogeneous knowledge sources. Simple text-only approaches lose crucial visual information, while naive multimodal methods struggle with coherent cross-modal integration. Our synthesis stage addresses these challenges by systematically combining retrieved multimodal knowledge into comprehensive, evidence-grounded responses.', '- (i) Building Textual Context. Given the top-ranked retrieval candidates  , we construct a structured textual context. We concatenate textual representations of all retrieved components, includ', 'Table 1: Statistics of Experimental Datasets.', 'ing entity summaries, relationship descriptions, and chunk contents. The concatenation incorporates appropriate delimiters to indicate modality types and hierarchical origins. This approach ensures the language model can effectively parse and reason over heterogeneous knowledge components.'], 'Table_description': 'The table presents comparative statistics for two datasets: DocBench and MMLongBench. For each dataset, the table shows the following data points: the total number of documents, the average number of pages per document, the average number of tokens per document, the number of document types, and the total number of questions associated. DocBench includes 229 documents with an average of 66 pages and 46,377 tokens per document, spanning 5 document types and a total of 1,102 questions. MMLongBench includes 135 documents with a lower average of 47.5 pages and 21,214 tokens per document, but more document types (7) and nearly the same number of questions (1,082). The data is organized to highlight direct, column-wise comparisons in document count, size, diversity, and question complexity between the two datasets.'}, {'page_no.': 7, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\f137776e8af0d7ce8bd6e318be2769a719ac6f4b1fb06f3adb4a40b32fbf9acc.jpg', 'content_type': 'table', 'contextual_text': ['Baselines. We compare RAG-Anything against the following methods for performance evaluation:', 'Experimental Settings. In our experiments, we implement all baselines using GPT-4o-mini as the backbone LLM. Documents are parsed using MinerU (Wang et al., 2024) to extract text, images, tables, and equations for downstream RAG processing. For the retrieval pipeline, we employ the text-embedding-3-large model with 3072-dimensional embeddings. We use the bge-reranker-v2-m3 model for reranking. For graph-based RAG methods, we enforce a combined entity-and-relation token limit of 20,000 tokens and a chunk token limit of 12,000 tokens.', 'Table 2: Accuracy (%) on DocBench Dataset. Performance results with best scores highlighted in dark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance (Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types are categorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.).'], 'Table_description': 'The table presents the accuracy percentages of four methods—GPT-4o-mini, LightRAG, MMGraphRAG, and RAGAnything—on the DocBench dataset, broken down by five domain categories (Academia, Finance, Government, Legal Documents, News), three query types (Text-only, Multimodal, Unanswerable), and an Overall score. Each cell shows the percentage accuracy for a specific method and category. The best-performing value in each column is highlighted in dark blue, while the second-best is light blue. For instance, MMGraphRAG achieves the highest accuracy in the Academia and Government domains (64.3% and 64.9% respectively), while RAGAnything performs best in Finance (67.0%) and overall (63.4%). The Text-only type achieves the same top score (85.0%) for both LightRAG and RAGAnything. Unanswerable accuracy peaks with MMGraphRAG (60.5%). These values allow direct comparison of method performance across multiple domains and query types, revealing the strengths of each approach in particular contexts.'}, {'page_no.': 7, 'index_on_page': 4, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\57705f0111dadaea4f4a06fd5b20d4c30089c4619ce0e95c4a9ba9af82a04625.jpg', 'content_type': 'table', 'contextual_text': ['Table 2: Accuracy (%) on DocBench Dataset. Performance results with best scores highlighted in dark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance (Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types are categorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.).', 'Table 3: Accuracy (%) on MMLongBench across different domains and overall performance. Best results are highlighted in dark blue and second-best in light blue.. Domain categories include Research Reports/Introductions (Res.), Tutorials/Workshops (Tut.), Academic Papers (Acad.), Guidebooks (Guid.), Brochures (Broch.), Administration/Industry Files (Admin.), and Financial Reports (Fin.).', 'Outputs are constrained to a one-sentence format. For the baseline GPT-4o-mini in our QA scenario, documents are concatenated into image form with a maximum of 50 pages per document, rendered at 144 dpi. Finally, all query results are evaluated for accuracy by GPT-4o-mini.'], 'Table_description': "The table presents accuracy percentages of four different methods (GPT-4o-mini, LightRAG, MMGraphRAG, and RAGAnything) when evaluated on the MMLongBench dataset across various document domains. The columns represent domains: Research Reports/Introductions (Res.), Tutorials/Workshops (Tut.), Academic Papers (Acad.), Guidebooks (Guid.), Brochures (Broch.), Administration/Industry Files (Admin.), and Financial Reports (Fin.), with an Overall average. Each method's performance in each domain is listed, and the best and second-best results per domain and overall are visually highlighted. RAGAnything displays the highest overall accuracy (42.8), outperforming other methods in most domain categories, while GPT-4o-mini generally shows the lowest accuracy."}, {'page_no.': 8, 'index_on_page': 7, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\12f755ba148b16e967d1e4628d11a09039e7a1dfd51629985b340dbb3b7bc223.jpg', 'content_type': 'table', 'contextual_text': ['Enhanced Long-Context Performance. RAG-Anything demonstrates superior performance on long-context documents. The framework excels where relevant evidence is dispersed across multiple modalities and sections. It achieves the best results in information-dense domains such as Research Reports and Financial Reports on MMLongBench. These improvements stem from the structured context injection mechanism. This mechanism integrates dual-graph construction for cross-page entity alignment. It combines semantic retrieval with structural navigation. The framework also employs modality-aware processing for efficient context window utilization. Unlike baselines that cannot uniformly process diverse modalities, RAG-Anything effectively captures scattered multimodal evidence. Its cross-modal hybrid retrieval architecture combines structural knowledge navigation with semantic similarity matching. This enables the framework to leverage both explicit relationships and implicit semantic connections across modalities.', "To systematically evaluate model performance across varying document lengths, we conducted comprehensive experiments on both datasets. As illustrated in Figure 2, RAG-Anything and MMGraphRAG exhibit comparable performance on shorter documents. However, RAG-Anything's advantages become increasingly pronounced as document length grows. On DocBench, the performance gap expands dramatically to over 13 points for documents exceeding 100 pages (68.2% vs.", 'Table 4: Ablation study results on DocBench. The "Chunk-only" variant bypasses dual-graph construction and relies solely on traditional chunk-based retrieval, while "w/o Reranker" eliminates cross-modal reranking but preserves the core graph-based architecture.', '54.6% for 101-200 pages; 68.8% vs. 55.0% for 200+ pages). On MMLongBench, RAG-Anything demonstrates consistent improvements across all length categories, achieving accuracy gains of 3.4 points for 11-50 pages, 9.3 points for 51-100 pages, and 7.9 points for 101-200 pages. These findings confirm that our dual-graph construction and cross-modal hybrid retrieval mechanism is particularly effective for long-document reasoning tasks.'], 'Table_description': "The table compares the performance of three different methods (Chunk-only, w/o Reranker, RAGAnything) across multiple evaluation domains and data types, as well as their overall scores. Under the 'Domains' columns (Aca., Fin., Gov., Law., News), the reported values indicate domain-specific performance for each method. For 'Types' (Txt., Mm., Una.), the table shows how each method performs on different data types: text, multimodal, and unanswerable. The 'Overall' column provides an aggregate performance score for each method. RAGAnything consistently achieves the highest scores across most domains and types, especially in the 'Fin.' (67.0), 'Gov.' (61.5), 'Txt.' (85.0), and overall (63.4) categories, compared to the other two methods. The Chunk-only method has the lowest scores overall, particularly in the 'Una.' type (43.5). Performance improvements are visually highlighted with darker blue shades for higher values."}, {'page_no.': 12, 'index_on_page': 5, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\33981691987ca5749049465db763f9c0e43875229ba6d96c9745466747165c90.jpg', 'content_type': 'table', 'contextual_text': ["This appendix provides comprehensive supporting materials for our experimental evaluation and implementation details. Section A.1 presents detailed dataset statistics for the DocBench and MMLongBench multi-modal benchmarks, including document type distributions and complexity metrics. Section A.2 showcases additional case studies that demonstrate RAG-Anything's structure-aware capabilities across diverse multimodal content understanding tasks. Section A.3 documents the complete set of multimodal analysis prompts for vision, table, and equation processing that enable context-aware interpretation. Section A.4 provides the standardized accuracy evaluation prompt used for consistent response assessment across all experimental conditions.", 'A.1 DATASET CHARACTERISTICS AND STATISTICS', 'Table 5: Document type distribution and statistics for the DocBench benchmark.'], 'Table_description': 'The table presents statistics related to document types in the DocBench benchmark. It categorizes data into five document types: Academic (Acad.), Financial (Fin.), Government (Gov.), Legal (Law.), and News. For each type, the table lists the number of documents (# Docs), the number of questions (# Questions), and the average number of pages per document (Avg. Pages). Academic documents are represented by 49 documents with 303 questions and an average of 11 pages per document. Financial documents contain 40 documents, 288 questions, and have the highest average page count at 192. Government documents have 44 entries with 148 questions and an average of 69 pages. Legal documents are listed with 46 entries, 191 questions, and an average of 58 pages per document. News documents, with 50 entries and 172 questions, have the lowest average length at just 1 page.'}, {'page_no.': 12, 'index_on_page': 7, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\968b2751f1362050c4c4adf30dfc6ad94c9d6eab5836a3535a145576ab107877.jpg', 'content_type': 'table', 'contextual_text': ['Table 5: Document type distribution and statistics for the DocBench benchmark.', 'Table 6: Document type distribution and statistics for the MMLongBench benchmark.', 'Tables 5 and 6 present the distribution of document types across the DocBench and MMLong-Bench benchmarks. • DocBench encompasses medium- to long-length documents spanning various domains, including legal, governmental, and financial files. Financial reports represent the most extensive category, averaging 192 pages per document, while the News category consists of concise single-page newspapers. • MMLongBench demonstrates a broader spectrum of document types and lengths. Research reports, tutorials, and academic papers maintain moderate lengths of 35–58 pages on average, while guidebooks extend to approximately 78 pages. Brochures and administrative files remain relatively compact, whereas financial reports again emerge as the longest category.'], 'Table_description': 'The table provides detailed statistics for different document types in the MMLongBench benchmark. It categorizes documents into Research (Res.), Tutorial (Tut.), Academic (Acad.), Guidebook (Guid.), Brochure (Broch.), Administrative (Admin.), and Financial (Fin.) types. For each category, the table lists the total number of documents (# Docs), total number of questions (# Questions), and the average number of pages (Avg. Pages) per document. Research documents are the most numerous (34), while administrative and financial documents have the fewest (10 and 11 respectively). Tutorial documents, though fewer in number, have one of the highest average page counts (58), surpassed only by financial (87) and guidebooks (78). Financial documents stand out with the highest average page length (87 pages). The distribution of questions is highest for research documents (292) and lowest for administrative ones (81). There is significant variability across types in both quantity and average length.'}]




                                            ####  Information Processor Generator Output  ####

("The table presents the accuracy percentages achieved by four different methods—GPT-4o-mini, LightRAG, MMGraphRAG, and RAGAnything—on the DocBench dataset, organized by domain (Academia, Finance, Government, Law, News), by document type (Text-only, Multimodal, Unanswerable), and with an Overall column. Each cell shows the method's accuracy in that specific setting. The best-performing results for each column are highlighted in dark blue, and the second-best in light blue. For example, MMGraphRAG achieves the highest accuracy in Academia (64.3%) and Government (64.9%), while RAGAnything leads in Finance (67.0%), Multimodal (76.3%), and overall performance (63.4%). Among document types, the highest performance on Text-only documents is shared by LightRAG and RAGAnything (85.0%). There are notable variations among methods depending on domain and type, with the RAGAnything method yielding the highest overall average accuracy.", [{'entity_name': 'DocBench Method Accuracy Table', 'entity_type': 'table', 'related_entities': 'Methods (GPT-4o-mini, LightRAG, MMGraphRAG, RAGAnything), Domains (Academia, Finance, Government, Law, News), Types (Text-only, Multimodal, Unanswerable), and Overall accuracy values. The table details the relationships between methods and their accuracy scores across different domains and document types.', 'entity_summary': 'This table summarizes the accuracy scores (%) of multiple retrieval-augmented generation (RAG) and baseline methods evaluated on the DocBench dataset across various domain categories and document types. It captures method-wise performance breakdown and highlights the best and second-best results, establishing comparative relationships between methods and content characteristics.'}], {'content_description': "The table presents the accuracy percentages achieved by four different methods—GPT-4o-mini, LightRAG, MMGraphRAG, and RAGAnything—on the DocBench dataset, organized by domain (Academia, Finance, Government, Law, News), by document type (Text-only, Multimodal, Unanswerable), and with an Overall column. Each cell shows the method's accuracy in that specific setting. The best-performing results for each column are highlighted in dark blue, and the second-best in light blue. For example, MMGraphRAG achieves the highest accuracy in Academia (64.3%) and Government (64.9%), while RAGAnything leads in Finance (67.0%), Multimodal (76.3%), and overall performance (63.4%). Among document types, the highest performance on Text-only documents is shared by LightRAG and RAGAnything (85.0%). There are notable variations among methods depending on domain and type, with the RAGAnything method yielding the highest overall average accuracy.", 'entity_summary': [{'entity_name': 'DocBench Method Accuracy Table', 'entity_type': 'table', 'related_entities': 'Methods (GPT-4o-mini, LightRAG, MMGraphRAG, RAGAnything), Domains (Academia, Finance, Government, Law, News), Types (Text-only, Multimodal, Unanswerable), and Overall accuracy values. The table details the relationships between methods and their accuracy scores across different domains and document types.', 'entity_summary': 'This table summarizes the accuracy scores (%) of multiple retrieval-augmented generation (RAG) and baseline methods evaluated on the DocBench dataset across various domain categories and document types. It captures method-wise performance breakdown and highlights the best and second-best results, establishing comparative relationships between methods and content characteristics.'}]})

                                            ####  Multi-model unit for context extraction ####

current_multimodel_unit = [{'page_no.': 7, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\f137776e8af0d7ce8bd6e318be2769a719ac6f4b1fb06f3adb4a40b32fbf9acc.jpg', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 2, 'table_caption': 'Table 2: Accuracy (%) on DocBench Dataset. Performance results with best scores highlighted in dark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance (Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types are categorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.).', 'content_type': 'table'}]


                                            ####  Multi-model knowledge units ####

multi_model_knowledge_units = [{'page_no.': 6, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\14861407bef7f68b31affea722351d2101dcf3994e2fdd92f50524cbefb3411b.jpg', 'content_type': 'table'}, {'page_no.': 6, 'index_on_page': 2, 'table_caption': 'Table 1: Statistics of Experimental Datasets.', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\f137776e8af0d7ce8bd6e318be2769a719ac6f4b1fb06f3adb4a40b32fbf9acc.jpg', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 2, 'table_caption': 'Table 2: Accuracy (%) on DocBench Dataset. Performance results with best scores highlighted in dark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance (Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types are categorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.).', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 4, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\57705f0111dadaea4f4a06fd5b20d4c30089c4619ce0e95c4a9ba9af82a04625.jpg', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 4, 'table_caption': 'Table 3: Accuracy (%) on MMLongBench across different domains and overall performance. Best results are highlighted in dark blue and second-best in light blue.. Domain categories include Research Reports/Introductions (Res.), Tutorials/Workshops (Tut.), Academic Papers (Acad.), Guidebooks (Guid.), Brochures (Broch.), Administration/Industry Files (Admin.), and Financial Reports (Fin.).', 'content_type': 'table'}, {'page_no.': 8, 'index_on_page': 7, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\12f755ba148b16e967d1e4628d11a09039e7a1dfd51629985b340dbb3b7bc223.jpg', 'content_type': 'table'}, {'page_no.': 8, 'index_on_page': 7, 'table_caption': 'Table 4: Ablation study results on DocBench. The "Chunk-only" variant bypasses dual-graph construction and relies solely on traditional chunk-based retrieval, while "w/o Reranker" eliminates cross-modal reranking but preserves the core graph-based architecture.', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 5, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\33981691987ca5749049465db763f9c0e43875229ba6d96c9745466747165c90.jpg', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 5, 'table_caption': 'Table 5: Document type distribution and statistics for the DocBench benchmark.', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 7, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\968b2751f1362050c4c4adf30dfc6ad94c9d6eab5836a3535a145576ab107877.jpg', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 7, 'table_caption': 'Table 6: Document type distribution and statistics for the MMLongBench benchmark.', 'content_type': 'table'}]


                                            ####  Textual Knowledge units  ####
textual_knowledge_units = [{'page_no.': 0, 'index_on_page': 1, 'raw_content': 'RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK', 'content_type': 'title'}, {'page_no.': 0, 'index_on_page': 2, 'raw_content': 'Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang*', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 3, 'raw_content': 'The University of Hong Kong', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 4, 'raw_content': 'zrguol01@hku.hk xubinrencs@gmail.com chaohuang75@gmail.com', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 5, 'raw_content': 'ABSTRACT', 'content_type': 'title'}, {'page_no.': 0, 'index_on_page': 6, 'raw_content': 'Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding Large Language Models beyond their static training limitations. However, a critical misalignment exists between current RAG capabilities and real-world information environments. Modern knowledge repositories are inherently multimodal, containing rich combinations of textual content, visual elements, structured tables, and mathematical expressions. Yet existing RAG frameworks are limited to textual content, creating fundamental gaps when processing multimodal documents. We present RAG-Anything, a unified framework that enables comprehensive knowledge retrieval across all modalities. Our approach reconceptualizes multimodal content as interconnected knowledge entities rather than isolated data types. The framework introduces dual-graph construction to capture both cross-modal relationships and textual semantics within a unified representation. We develop cross-modal hybrid retrieval that combines structural knowledge navigation with semantic matching. This enables effective reasoning over heterogeneous content where relevant evidence spans multiple modalities. RAG-Anything demonstrates superior performance on challenging multimodal benchmarks, achieving significant improvements over state-of-the-art methods. Performance gains become particularly pronounced on long documents where traditional approaches fail. Our framework establishes a new paradigm for multimodal knowledge access, eliminating the architectural fragmentation that constrains current systems. Our framework is open-sourced at: https://github.com/HKUDS/RAG-Anything.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 7, 'raw_content': '1 INTRODUCTION', 'content_type': 'title'}, {'page_no.': 0, 'index_on_page': 8, 'raw_content': 'Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding the knowledge boundaries of Large Language Models (LLM) beyond their static training limitations Zhang et al. (2025). By enabling dynamic retrieval and incorporation of external knowledge during inference, RAG systems transform static language models into adaptive, knowledge-aware systems. This capability has proven essential for applications requiring up-to-date information, domain-specific knowledge, or factual grounding that extends beyond pre-training corpora.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 9, 'raw_content': 'However, existing RAG frameworks focus exclusively on text-only knowledge while neglecting the rich multimodal information present in real-world documents. This limitation fundamentally misaligns with how information exists in authentic environments. Real-world knowledge repositories are inherently heterogeneous and multimodal Abootorabi et al. (2025). They contain rich combinations of textual content, visual elements, structured tables, and mathematical expressions across diverse document formats. This textual assumption forces existing RAG systems to either discard non-textual information entirely or flatten complex multimodal content into inadequate textual approximations.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 10, 'raw_content': 'The consequences of this limitation become particularly severe in document-intensive domains where multimodal content carries essential meaning. Academic research, financial analysis, and technical documentation represent prime examples of knowledge-rich environments. These domains fundamentally depend on visual and structured information. Critical insights are often encoded exclusively in non-textual formats. Such formats resist meaningful conversion to plain text.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 11, 'raw_content': 'The consequences of this limitation become particularly severe in knowledge-intensive domains where multimodal content carries essential meaning. Three representative scenarios illustrate the critical', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 1, 'raw_content': 'need for multimodal RAG capabilities. In Scientific Research, experimental results are primarily communicated through plots, diagrams, and statistical visualizations. These contain core discoveries that remain invisible to text-only systems. Financial Analysis relies heavily on market charts, correlation matrices, and performance tables. Investment insights are encoded in visual patterns rather than textual descriptions. Additionally, Medical Literature Analysis depends on radiological images, diagnostic charts, and clinical data tables. These contain life-critical information essential for accurate diagnosis and treatment decisions. Current RAG frameworks systematically exclude these vital knowledge sources across all three scenarios. This creates fundamental gaps that render them inadequate for real-world applications requiring comprehensive information understanding. Therefore, multimodal RAG emerges as a critical advancement. It is necessary to bridge these knowledge gaps and enable truly comprehensive intelligence across all modalities of human knowledge representation.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 2, 'raw_content': 'Addressing multimodal RAG presents three fundamental technical challenges that demand principled solutions. This makes it significantly more complex than traditional text-only approaches. The naive solution of converting all multimodal content to textual descriptions introduces severe information loss. Visual elements such as charts, diagrams, and spatial layouts contain semantic richness that cannot be adequately captured through text alone. These inherent limitations necessitate the design of effective technical components. Such components must be specifically designed to handle multimodal complexity and preserve the full spectrum of information contained within diverse content types.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 3, 'raw_content': 'Technical Challenges.   First, the unified multimodal representation challenge requires seamlessly integrating diverse information types. The system must preserve their unique characteristics and cross-modal relationships. This demands advanced multimodal encoders that can capture both intra-modal and inter-modal dependencies without losing essential visual semantics.   Second, the structure-aware decomposition challenge demands intelligent parsing of complex layouts. The system must maintain spatial and hierarchical relationships crucial for understanding. This requires specialized layout-aware parsing modules that can interpret document structure and preserve contextual positioning of multimodal elements.   Third, the cross-modal retrieval challenge necessitates sophisticated mechanisms that can navigate between different modalities. These mechanisms must reason over their interconnections during retrieval. This calls for cross-modal alignment systems capable of understanding semantic correspondences across text, images, and structured data. These challenges are amplified in long-context scenarios. Relevant evidence is dispersed across multiple modalities and sections, requiring coordinated reasoning across heterogeneous information sources.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 4, 'raw_content': 'Our Contributions. To address these challenges, we introduce RAG-Anything, a unified framework that fundamentally reimagines multimodal knowledge representation and retrieval. Our approach employs a dual-graph construction strategy that elegantly bridges the gap between cross-modal understanding and fine-grained textual semantics. Rather than forcing diverse modalities into text-centric pipelines, RAG-Anything constructs complementary knowledge graphs that preserve both multimodal contextual relationships and detailed textual knowledge. This design enables seamless integration of visual elements, structured data, and mathematical expressions within a unified retrieval framework. The system maintains semantic integrity across modalities while ensuring efficient cross-modal reasoning capabilities throughout the process.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 5, 'raw_content': 'Our cross-modal hybrid retrieval mechanism strategically combines structural knowledge navigation with semantic similarity matching. This architecture addresses the fundamental limitation of existing approaches that rely solely on embedding-based retrieval or keyword matching. RAG-Anything leverages explicit graph relationships to capture multi-hop reasoning patterns. It simultaneously employs dense vector representations to identify semantically relevant content that lacks direct structural connections. The framework introduces modality-aware query processing and cross-modal alignment systems. These enable textual queries to effectively access visual and structured information. This unified approach eliminates the architectural fragmentation that plagues current multimodal RAG systems. It delivers superior performance particularly on long-context documents where relevant evidence spans multiple modalities and document sections.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 6, 'raw_content': 'Experimental Validation. To validate the effectiveness of our proposed approach, we conduct comprehensive experiments on two challenging multimodal benchmarks: DocBench and MMLongBench. Our evaluation demonstrates that RAG-Anything achieves superior performance across diverse domains. The framework represents substantial improvements over state-of-the-art baselines. Notably, our performance gains become increasingly significant as content length increases. We observe particularly pronounced advantages on long-context materials. This validates our core hypothesis', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 1, 'raw_content': 'that dual-graph construction and cross-modal hybrid retrieval are essential for handling complex multimodal materials. Our ablation studies reveal that graph-based knowledge representation provides the primary performance gains. Traditional chunk-based approaches fail to capture the structural relationships critical for multimodal reasoning. Case studies further demonstrate that our framework excels at precise localization within complex layouts. The system effectively disambiguates similar terminology and navigates multi-panel visualizations through structure-aware retrieval mechanisms.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 2, 'raw_content': '2 THE RAG-ANYTHING FRAMEWORK', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 3, 'raw_content': '2.1 PRELIMINARY', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 4, 'raw_content': 'Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for dynamically expanding the knowledge boundaries of LLMs. While LLMs demonstrate exceptional reasoning capabilities, their knowledge remains static and bounded by training data cutoffs. This creates an ever-widening gap with the rapidly evolving information landscape. RAG systems address this critical limitation by enabling LLMs to retrieve and incorporate external knowledge sources during inference. This transforms them from static repositories into adaptive, knowledge-aware systems.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 5, 'raw_content': 'The Multimodal Reality: Beyond Text-Only RAG. Current RAG systems face a critical limitation that severely restricts their real-world deployment. Existing frameworks operate under the restrictive assumption that knowledge corpus consists exclusively of plain textual documents. This assumption fundamentally misaligns with how information exists in authentic environments. Real-world knowledge repositories are inherently heterogeneous and multimodal, containing rich combinations of textual content, visual elements, structured data, and mathematical expressions. These diverse knowledge sources span multiple document formats and presentation mediums, from research papers and technical slides to web pages and interactive documents.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 6, 'raw_content': '2.1.1 MOTIVATING RAG-ANYTHING', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 7, 'raw_content': 'This multimodal reality introduces fundamental technical challenges that expose the inadequacy of current text-only RAG approaches. Effective multimodal RAG requires unified indexing strategies that can handle disparate data types, cross-modal retrieval mechanisms that preserve semantic relationships across modalities, and sophisticated synthesis techniques that can coherently integrate diverse information sources. These challenges demand a fundamentally different architectural approach rather than incremental improvements to existing systems.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 8, 'raw_content': 'The RAG-Anything framework introduces a unified approach for retrieving and processing knowledge from heterogeneous multimodal information sources. Our system addresses the fundamental challenge of handling diverse data modalities and document formats within a retrieval pipeline. The framework comprises three core components: universal indexing for multimodal knowledge, cross-modal adaptive retrieval, and knowledge-enhanced response generation. This integrated design enables effective knowledge utilization across modalities while maintaining computational efficiency.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 9, 'raw_content': '2.2 UNIVERSAL REPRESENTATION FOR HETEROGENEOUS KNOWLEDGE', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 10, 'raw_content': 'A key requirement for universal knowledge access is the ability to represent heterogeneous multimodal content in a unified, retrieval-oriented abstraction. Unlike existing pipelines that simply parse documents into text segments, RAG-Anything introduces Multimodal Knowledge Unification. This process decomposes raw inputs into atomic knowledge units while preserving their structural context and semantic alignment. For instance, RAG-Anything ensures that figures remain grounded in their captions, equations remain linked to surrounding definitions, and tables stay connected to explanatory narratives. This transforms heterogeneous files into a coherent substrate for cross-modal retrieval.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 11, 'raw_content': 'Formally, each knowledge source   (e.g., a web page) is decomposed into atomic content units:', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 13, 'raw_content': 'where each unit   consists of a modality type   and its corresponding raw content  . The content   represents the extracted information from the original knowledge source, processed in a modality-aware manner to preserve semantic integrity.', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 3, 'raw_content': 'To ensure high-fidelity extraction, RAG-Anything leverages specialized parsers for different content types. Text is segmented into coherent paragraphs or list items. Figures are extracted with associated metadata such as captions and cross-references. Tables are parsed into structured cells with headers and values. Mathematical expressions are converted into symbolic representations. The resulting   preserves both content and structural context within the source. This provides a faithful, modality-consistent representation. The decomposition abstracts diverse file formats into atomic units while maintaining their hierarchical order and contextual relationships. This canonicalization enables uniform processing, indexing, and retrieval of multimodal content within our framework.', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 4, 'raw_content': '2.2.1 DUAL-GRAPH CONSTRUCTION FOR MULTIMODAL KNOWLEDGE', 'content_type': 'title'}, {'page_no.': 3, 'index_on_page': 5, 'raw_content': 'While multimodal knowledge unification provides a uniform abstraction across modalities, directly constructing a single unified graph often risks overlooking modality-specific structural signals. The proposed RAG-Anything addresses this challenge through a dual-graph construction strategy. The system first builds a cross-modal knowledge graph that faithfully grounds non-textual modalities within their contextual environment. It then constructs a text-based knowledge graph using established text-centric extraction pipelines. These complementary graphs are merged through entity alignment. This design ensures accurate cross-modal grounding and comprehensive coverage of textual semantics, enabling richer knowledge representation and robust retrieval.', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 6, 'raw_content': "- Cross-Modal Knowledge Graph: Non-textual content like images, tables, and equations contains rich semantic information that traditional text-only approaches often overlook. To preserve this knowledge, RAG-Anything constructs a multimodal knowledge graph where non-text atomic units are transformed into structured graph entities. RAG-Anything leverages multimodal large language models to derive two complementary textual representations from each atomic content unit. The first is a detailed description   optimized for cross-modal retrieval. The second is an entity summary   containing key attributes such as entity name, type, and description for graph construction. The generation process is context-aware, processing each unit with its local neighborhood  , where   controls the contextual window size. This ensures representations accurately reflect each unit's role within the broader document structure.", 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 7, 'raw_content': 'Building on these textual representations, RAG-Anything constructs the graph structure using nontext units as anchor points. For each non-text unit  , the graph extraction routine   processes its description   to identify fine-grained entities and relations:', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 9, 'raw_content': 'where   and   denote the sets of intra-chunk entities and their relations, respectively. Each atomic non-text unit is associated with a multimodal entity node   that serves as an anchor for', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 1, 'raw_content': 'its intra-chunk entities through explicit belongs_to edges:', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 4, 'raw_content': 'This construction preserves modality-specific grounding while ensuring non-textual content is contextualized by its textual neighborhood. This enables reliable cross-modal retrieval and reasoning.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 5, 'raw_content': "- Text-based Knowledge Graph: For text modality chunks, we construct a traditional text-based knowledge graph following established methodologies similar to LightRAG (Guo et al., 2024) and GraphRAG (Edge et al., 2024). The extraction process operates directly on textual content   where   text, leveraging named entity recognition and relation extraction techniques to identify entities and their semantic relationships. Given the rich semantic information inherent in textual content, multimodal context integration is not required for this component. The resulting text-based knowledge graph captures explicit knowledge and semantic connections present in textual portions of documents, complementing the multimodal graph's cross-modal grounding capabilities.", 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 6, 'raw_content': '2.2.2 GRAPH FUSION AND INDEX CREATION', 'content_type': 'title'}, {'page_no.': 4, 'index_on_page': 7, 'raw_content': 'The separate cross-modal and text-based knowledge graphs capture complementary aspects of document semantics. Integrating them creates a unified representation leveraging visual-textual associations and fine-grained textual relationships for enhanced retrieval.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 12, 'raw_content': 'where   denotes the embedding function tailored for each component type. Together, the unified knowledge graph   and the embedding table   constitute the complete retrieval index  . This provides both structural knowledge representation and dense vector space for efficient cross-modal similarity search during the subsequent retrieval stage.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 13, 'raw_content': '2.3 CROSS-MODAL HYBRID RETRIEVAL', 'content_type': 'title'}, {'page_no.': 4, 'index_on_page': 14, 'raw_content': 'The retrieval stage operates on the index   to identify relevant knowledge components for a given user query. Traditional RAG methods face significant limitations when dealing with multimodal documents. They typically rely on semantic similarity within single modalities and fail to capture the rich interconnections between visual, mathematical, tabular, and textual elements. To address these challenges, our framework introduces a cross-modal hybrid retrieval mechanism. This mechanism leverages structural knowledge and semantic representations across heterogeneous modalities.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 15, 'raw_content': 'Modality-Aware Query Encoding. Given a user query  , we first perform modality-aware query analysis to extract lexical cues and potential modality preferences embedded within the query. For instance, queries containing terms such as "figure," "chart," "table," or "equation" provide explicit signals about the expected modality of relevant information. We then compute a unified text embedding   using the same encoder employed during indexing, ensuring consistency between', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 1, 'raw_content': 'query and knowledge representations. This embedding-based approach enables cross-modal retrieval capabilities where textual queries can effectively access multimodal content through their shared representations, maintaining retrieval consistency while preserving cross-modal accessibility.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 2, 'raw_content': 'Hybrid Knowledge Retrieval Architecture. Recognizing that knowledge relevance manifests through both explicit structural connections and implicit semantic relationships, we design a hybrid retrieval architecture that strategically combines two complementary mechanisms.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 3, 'raw_content': '- (i) Structural Knowledge Navigation. This mechanism addresses the challenge of capturing explicit relationships and multi-hop reasoning patterns. Traditional keyword-based retrieval often fails to identify knowledge connected through intermediate entities or cross-modal relationships. To overcome this limitation, we exploit the structural properties encoded within our unified knowledge graph G. We employ keyword matching and entity recognition to locate relevant graph components. The retrieval process begins with exact entity matching against query terms.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 4, 'raw_content': 'We then perform strategic neighborhood expansion to include related entities and relationships within a specified hop distance. This structural approach proves particularly effective at uncovering high-level semantic connections and entity-relation patterns that span multiple modalities. It capitalizes on the rich cross-modal linkages established in our multimodal knowledge graph. The structural navigation yields candidate set   containing relevant entities, relationships, and their associated content chunks that provide comprehensive contextual information.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 5, 'raw_content': '- (ii) Semantic Similarity Matching. This mechanism addresses the challenge of identifying semantically relevant knowledge that lacks explicit structural connections. While structural navigation excels at following explicit relationships, it may miss relevant content that is semantically related but not directly connected in the graph topology. To bridge this gap, we conduct dense vector similarity search between the query embedding   and all components stored in embedding table  .', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 6, 'raw_content': 'This approach encompasses atomic content chunks across all modalities, graph entities, and relationship representations, enabling fine-grained semantic matching that can surface relevant knowledge even when traditional lexical or structural signals are absent. The learned embedding space captures nuanced semantic relationships and contextual similarities that complement the explicit structural signals from the navigation mechanism. This retrieval pathway returns the top-k most semantically similar chunks   ranked by cosine similarity scores, ensuring comprehensive coverage of both structurally and semantically relevant knowledge.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 7, 'raw_content': 'Candidate Pool Unification. Both retrieval pathways may return overlapping candidates with differing relevance signals. This necessitates a principled approach to unify and rank results. Retrieval candidates from both pathways are unified into a comprehensive candidate pool:  . Simply merging candidates would ignore distinct evidence each pathway provides. It would fail to account for redundancy between retrieved content.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 11, 'raw_content': '2.4 FROM RETRIEVAL TO SYNTHESIS', 'content_type': 'title'}, {'page_no.': 5, 'index_on_page': 12, 'raw_content': 'Effective multimodal question answering requires preserving rich visual semantics while maintaining coherent grounding across heterogeneous knowledge sources. Simple text-only approaches lose crucial visual information, while naive multimodal methods struggle with coherent cross-modal integration. Our synthesis stage addresses these challenges by systematically combining retrieved multimodal knowledge into comprehensive, evidence-grounded responses.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 13, 'raw_content': '- (i) Building Textual Context. Given the top-ranked retrieval candidates  , we construct a structured textual context. We concatenate textual representations of all retrieved components, includ', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 3, 'raw_content': 'ing entity summaries, relationship descriptions, and chunk contents. The concatenation incorporates appropriate delimiters to indicate modality types and hierarchical origins. This approach ensures the language model can effectively parse and reason over heterogeneous knowledge components.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 4, 'raw_content': '- (ii) Recovering Visual Content. For multimodal chunks corresponding to visual artifacts, we perform dereferencing to recover original visual content, creating  . This design maintains consistency with our unified embedding strategy. Textual proxies enable efficient retrieval while authentic visual content provides rich semantics necessary for sophisticated reasoning during synthesis.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 5, 'raw_content': 'The synthesis process jointly conditions on both the assembled comprehensive textual context and dereferenced visual artifacts using a vision-language model:', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 7, 'raw_content': 'where the VLM integrates information from query, textual context, and visual content. This unified conditioning enables sophisticated visual interpretation while maintaining grounding in retrieved evidence. The resulting responses are both visually informed and factually grounded.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 8, 'raw_content': '3 EVALUATION', 'content_type': 'title'}, {'page_no.': 6, 'index_on_page': 9, 'raw_content': '3.1 EXPERIMENTAL SETTINGS', 'content_type': 'title'}, {'page_no.': 6, 'index_on_page': 10, 'raw_content': 'Evaluation Datasets. We conduct comprehensive evaluations on two challenging multimodal Document Question Answering (DQA) benchmarks that reflect real-world complexity and diversity. DocBench (Zou et al., 2024) provides a rigorous testbed with 229 multimodal documents spanning five critical domains: Academia, Finance, Government, Laws, and News. The dataset includes 1,102 expert-crafted question-answer pairs. These documents are notably extensive, averaging 66 pages and approximately 46,377 tokens, which presents substantial challenges for long-context understanding.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 11, 'raw_content': 'MMLongBench (Ma et al., 2024) complements this evaluation by focusing specifically on long-context multimodal document comprehension. It features 135 documents across 7 diverse document types with 1,082 expert-annotated questions. Together, these benchmarks provide comprehensive coverage of the multimodal document understanding challenges that RAG-Anything aims to address. They ensure our evaluation captures both breadth across domains and depth in document complexity. Detailed dataset statistics and characteristics are provided in Appendix A.1.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 12, 'raw_content': 'Baselines. We compare RAG-Anything against the following methods for performance evaluation:', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 17, 'raw_content': 'Experimental Settings. In our experiments, we implement all baselines using GPT-4o-mini as the backbone LLM. Documents are parsed using MinerU (Wang et al., 2024) to extract text, images, tables, and equations for downstream RAG processing. For the retrieval pipeline, we employ the text-embedding-3-large model with 3072-dimensional embeddings. We use the bge-reranker-v2-m3 model for reranking. For graph-based RAG methods, we enforce a combined entity-and-relation token limit of 20,000 tokens and a chunk token limit of 12,000 tokens.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 5, 'raw_content': 'Outputs are constrained to a one-sentence format. For the baseline GPT-4o-mini in our QA scenario, documents are concatenated into image form with a maximum of 50 pages per document, rendered at 144 dpi. Finally, all query results are evaluated for accuracy by GPT-4o-mini.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 6, 'raw_content': '3.2 PERFORMANCE COMPARISON', 'content_type': 'title'}, {'page_no.': 7, 'index_on_page': 7, 'raw_content': 'Superior Performance and Cross-Domain Generalization. RAG-Anything demonstrates superior overall performance over baselines through its unified multimodal framework. Unlike LightRAG, which is restricted to text-only content processing, RAG-Anything treats text, images, tables, and equations as first-class entities. MMGraphRAG only adds basic image processing while treating tables and equations as plain text, missing crucial structural information. RAG-Anything introduces a comprehensive dual-graph construction strategy that preserves structural relationships across all modalities. This unified approach enables superior performance across both evaluation benchmarks.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 8, 'raw_content': 'Enhanced Long-Context Performance. RAG-Anything demonstrates superior performance on long-context documents. The framework excels where relevant evidence is dispersed across multiple modalities and sections. It achieves the best results in information-dense domains such as Research Reports and Financial Reports on MMLongBench. These improvements stem from the structured context injection mechanism. This mechanism integrates dual-graph construction for cross-page entity alignment. It combines semantic retrieval with structural navigation. The framework also employs modality-aware processing for efficient context window utilization. Unlike baselines that cannot uniformly process diverse modalities, RAG-Anything effectively captures scattered multimodal evidence. Its cross-modal hybrid retrieval architecture combines structural knowledge navigation with semantic similarity matching. This enables the framework to leverage both explicit relationships and implicit semantic connections across modalities.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 9, 'raw_content': "To systematically evaluate model performance across varying document lengths, we conducted comprehensive experiments on both datasets. As illustrated in Figure 2, RAG-Anything and MMGraphRAG exhibit comparable performance on shorter documents. However, RAG-Anything's advantages become increasingly pronounced as document length grows. On DocBench, the performance gap expands dramatically to over 13 points for documents exceeding 100 pages (68.2% vs.", 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 8, 'raw_content': '54.6% for 101-200 pages; 68.8% vs. 55.0% for 200+ pages). On MMLongBench, RAG-Anything demonstrates consistent improvements across all length categories, achieving accuracy gains of 3.4 points for 11-50 pages, 9.3 points for 51-100 pages, and 7.9 points for 101-200 pages. These findings confirm that our dual-graph construction and cross-modal hybrid retrieval mechanism is particularly effective for long-document reasoning tasks.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 9, 'raw_content': '3.3 ARCHITECTURAL VALIDATION WITH ABLATION STUDIES', 'content_type': 'title'}, {'page_no.': 8, 'index_on_page': 10, 'raw_content': 'To isolate and quantify the contributions of key architectural components in RAG-Anything, we conducted systematic ablation studies examining two critical design choices. Given that our approach fundamentally differs from existing methods through dual-graph construction and hybrid retrieval, we specifically evaluated: i) Chunk-only, which bypasses graph construction entirely and relies solely on traditional chunk-based retrieval, and ii) w/o Reranker, which eliminates the cross-modal reranking component while preserving the core graph-based architecture.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 11, 'raw_content': 'As demonstrated in Table 4, the results validate our architectural design through striking performance variations.   Graph Construction is Essential. The chunk-only variant achieves merely   accuracy with substantial cross-domain drops. This demonstrates that traditional chunking fails to capture structural and cross-modal relationships essential for multimodal documents.   Reranking Provides Marginal Gains. Removing the reranker yields only a modest decline to  , while the full model achieves   accuracy. This indicates that cross-modal reranking provides valuable refinement, but primary gains stem from our graph-based retrieval and cross-modal integration.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 12, 'raw_content': '3.4 CASE STUDIES', 'content_type': 'title'}, {'page_no.': 8, 'index_on_page': 13, 'raw_content': 'Multimodal documents contain rich structural information within each modality. Understanding these intra-modal structures is crucial for accurate reasoning. We analyze two representative cases from DocBench to demonstrate how RAG-Anything leverages these structures. These cases highlight a key limitation of existing methods. Baselines either rely on superficial textual cues or flatten complex visual elements into plain text. In contrast, RAG-Anything builds modality-aware graphs that preserve essential relationships (e.g., table header   cell   unit edges; panel   caption   axis edges). This enables precise reasoning over complex document layouts.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 14, 'raw_content': '- Case 1: Multi-panel Figure Interpretation. This case examines a common scenario in academic literature. Researchers often need to compare results across different experimental conditions. These results are typically presented in multi-panel visualizations. Figure 3 shows a challenging t-SNE', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 7, 'raw_content': 'visualization with multiple subpanels. The query requires distinguishing between two related but distinct panels. RAG-Anything constructs a visual layout graph where panels, axis titles, legends, and captions become nodes. Key edges encode semantic relationships. Panels contain specific plots. Captions provide contextual information. Subfigures relate hierarchically. This structure guides the retriever to focus on the style-space panel for comparing cluster separation patterns. The system avoids confusion from the adjacent content space panel. This panel shows less clear distinctions.', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 14, 'raw_content': '- Case 2: Financial Table Navigation. This case addresses a common challenge in financial document analysis. Analysts must extract specific metrics from tables with similar terminology and multiple time periods. Figure 4 shows this scenario. The query involves resolving ambiguous financial terms and selecting the correct column for a specified year.', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 15, 'raw_content': 'RAG-Anything transforms the financial report table into a structured graph. Each row header, column header (year), data cell, and unit becomes a node. The edges capture key relationships: row-of, column-of, header-applies-to, and unit-of. This structure enables precise navigation. The retriever focuses on the row "Wages and salaries" and the column for "2020". It directs attention to the target cell (26,778 million). The system successfully disambiguates nearby entries like "Share-based payments." Competing methods treat tables as linear text. They often confuse numerical spans and years. This leads to significantly inaccurate answers. RAG-Anything explicitly models relationships within the table. It achieves precise selection and numeric grounding. This ensures accurate responses.', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 16, 'raw_content': "- Key Insights. Both cases demonstrate how RAG-Anything's structure-aware design delivers targeted advantages. Our approach transforms documents into explicit graph representations. These graphs capture intra-modal relationships that traditional methods miss. In figures, connections between panels, captions, and axes enable panel-level comparisons. This goes beyond keyword matching. In tables, row-column-unit graphs ensure accurate identification through modeling.", 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 17, 'raw_content': "This structure-aware retrieval design reduces confusion from repeated terminology and complex layouts. Traditional RAG systems struggle with these scenarios due to lack of structural understanding. Even MMGraphRAG fails here because it only considers image modality entities. It ignores other modality entities like table cells, row headers, and column headers. RAG-Anything's comprehensive graph representation captures all modality-specific entities and their relationships. This enables precise, modality-specific grounding that leads to consistent improvements in document Q&A tasks requiring fine-grained localization. Additional cases are available in Appendix A.2.", 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 18, 'raw_content': '4 RELATED WORK', 'content_type': 'title'}, {'page_no.': 9, 'index_on_page': 19, 'raw_content': '- Graph-Enhanced Retrieval-Augmented Generation. Large language models struggle with long-context inputs and multi-hop queries, failing to precisely locate dispersed evidence (Zhang et al.,', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 1, 'raw_content': '2025). Graph structures address this limitation by introducing explicit relational modeling, improving both retrieval efficiency and reasoning accuracy (Bei et al., 2025).', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 2, 'raw_content': "Since GraphRAG (Edge et al., 2024), research has evolved along two complementary directions. First, graph construction approaches optimize structures for retrieval efficiency, ranging from LightRAG's (Guo et al., 2024) sparsified indices to neural models like GNN-RAG (Mavromatis & Karypis, 2024) and memory-augmented variants like HippoRAG (Jimenez Gutierrez et al., 2024). Second, knowledge aggregation approaches integrate information for multi-level reasoning through hierarchical methods like RAPTOR (Sarthi et al., 2024) and ArchRAG (Wang et al., 2025). Despite these advances, existing systems remain text-centric with homogeneous inputs. This limits their applicability to multimodal documents and constrains robust reasoning over heterogeneous content. RAG-Anything addresses this gap by extending GraphRAG to all modalities.", 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 3, 'raw_content': '- Multimodal Retrieval-Augmented Generation. Multimodal RAG represents a natural evolution from text-based RAG systems, addressing the need to integrate external knowledge from diverse data modalities for comprehensive response generation (Abootorabi et al., 2025). However, current approaches are fundamentally constrained by their reliance on modality-specific architectures. Existing methods demonstrate these constraints across domains: VideoRAG (Ren et al., 2025) employs dual-channel architectures for video understanding while MM-VID (Lin et al., 2023) converts videos to text, losing visual information; VisRAG (Yu et al., 2025) preserves document layouts as images but misses granular relationships; MMGraphRAG (Wan & Yu, 2025) links scene graphs with textual representations but suffers from structural blindness—treating tables and formulas as plain text without proper entity extraction, losing structural information for reasoning.', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 4, 'raw_content': 'The fundamental problem underlying these limitations is architectural fragmentation. Current systems require specialized processing pipelines for each modality. This creates poor generalizability as new modalities demand custom architectures and fusion mechanisms. Such fragmentation introduces cross-modal alignment difficulties, modality biases, and information bottlenecks. These issues systematically compromise system performance and scalability. RAG-Anything addresses this fragmentation through a unified graph-based framework. Our approach processes all modalities with consistent structured modeling. This eliminates architectural constraints while preserving multimodal information integrity. The result is seamless cross-modal reasoning across heterogeneous content.', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 5, 'raw_content': '5 CONCLUSION', 'content_type': 'title'}, {'page_no.': 10, 'index_on_page': 6, 'raw_content': 'RAG-Anything introduces a paradigm shift in multimodal retrieval through its unified graph-based framework. Our core technical innovation is the dual-graph construction strategy that seamlessly integrates cross-modal and text-based knowledge graphs. Rather than forcing diverse modalities into text-centric pipelines that lose critical structural information, our approach fundamentally reconceptualizes multimodal content as interconnected knowledge entities with rich semantic relationships. The hybrid retrieval mechanism strategically combines structural navigation with semantic matching, enabling precise reasoning over complex document layouts. Comprehensive evaluation demonstrates superior performance on long-context documents, particularly those exceeding 100 pages where traditional methods fail. This work establishes a new foundation for multimodal RAG systems that can handle the heterogeneous nature of diverse information landscapes.', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 7, 'raw_content': 'Our analysis in Appendix A.5 reveals critical challenges facing current multimodal RAG systems. Two fundamental issues emerge through systematic failure case examination. First, systems exhibit text-centric retrieval bias, preferentially accessing textual sources even when queries explicitly require visual information. Second, rigid spatial processing patterns fail to adapt to non-standard document layouts. These limitations manifest in cross-modal misalignment scenarios and structurally ambiguous tables. The findings highlight the need for adaptive spatial reasoning and layout-aware parsing mechanisms to handle real-world multimodal document complexity.', 'content_type': 'text'}, {'page_no.': 11, 'index_on_page': 1, 'raw_content': 'REFERENCES', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 1, 'raw_content': 'A APPENDIX', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 2, 'raw_content': "This appendix provides comprehensive supporting materials for our experimental evaluation and implementation details. Section A.1 presents detailed dataset statistics for the DocBench and MMLongBench multi-modal benchmarks, including document type distributions and complexity metrics. Section A.2 showcases additional case studies that demonstrate RAG-Anything's structure-aware capabilities across diverse multimodal content understanding tasks. Section A.3 documents the complete set of multimodal analysis prompts for vision, table, and equation processing that enable context-aware interpretation. Section A.4 provides the standardized accuracy evaluation prompt used for consistent response assessment across all experimental conditions.", 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 3, 'raw_content': 'A.1 DATASET CHARACTERISTICS AND STATISTICS', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 8, 'raw_content': 'Tables 5 and 6 present the distribution of document types across the DocBench and MMLong-Bench benchmarks. • DocBench encompasses medium- to long-length documents spanning various domains, including legal, governmental, and financial files. Financial reports represent the most extensive category, averaging 192 pages per document, while the News category consists of concise single-page newspapers. • MMLongBench demonstrates a broader spectrum of document types and lengths. Research reports, tutorials, and academic papers maintain moderate lengths of 35–58 pages on average, while guidebooks extend to approximately 78 pages. Brochures and administrative files remain relatively compact, whereas financial reports again emerge as the longest category.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 9, 'raw_content': 'Collectively, these two benchmarks provide comprehensive coverage ranging from brief news articles to extensive technical and financial documentation. This establishes diverse and challenging evaluation contexts for multimodal document understanding tasks.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 10, 'raw_content': 'A.2 ADDITIONAL CASE STUDIES', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 12, 'raw_content': 'GPT-4o-mini: The removal of the dual coattention mechanism from the GCAN sub-model resulted in the lowest accuracy for Twitter15.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 13, 'raw_content': 'MMGraphRAG: The removal of the source tweet from the GCAN model resulted in the lowest accuracy for Twitter15, as indicated by significant accuracy drops in the ablation analysis in Figure 4.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 14, 'raw_content': 'LightRAG: Removing the source tweet from the GCAN model resulted in the lowest accuracy for Twitter15, as indicated by a significant drop in performance when using the sub-model that excluded both source tweet embeddings and dual co-attention.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 15, 'raw_content': 'RAG-Anything(Correct): The removal of the source tweet embeddings and dual co-attention, indicated as model configuration "S-A," resulted in the lowest accuracy for Twitter15.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 17, 'raw_content': '- Demonstrating Visual Reasoning Capabilities. Figure 5 illustrates how RAG-Anything handles complex visual reasoning tasks involving chart interpretation. The query asks which GCAN sub-model component removal yields the lowest accuracy on Twitter15. Traditional approaches struggle', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 1, 'raw_content': 'with spatial relationships between visual elements. RAG-Anything addresses this challenge by constructing a structured graph representation of the bar plot. Bars, axis labels, and legends become interconnected nodes. These are linked by semantic relations such as bar-of and label-applies-to.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 2, 'raw_content': 'This graph-based approach enables precise alignment between visual and textual elements. The system correctly identifies the bar labeled "-S-A" (removing source tweet embeddings and dual co-attention) and its corresponding accuracy value as the lowest performer. Baseline methods that flatten visual information often misinterpret spatial relationships. They frequently conflate nearby components. RAG-Anything\'s structured representation preserves critical visual-textual associations. This leads to accurate query resolution and proper attribution of performance drops to "-S-A".', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 9, 'raw_content': "- Handling Complex Tabular Structures. Figure 6 showcases RAG-Anything's ability to navigate intricate tabular data where structural disambiguation is crucial. The query seeks the model combination achieving the highest AUPRC value for the Evidence Inference dataset—a task complicated by repeated row labels across multiple datasets within the same table. This scenario highlights a fundamental limitation of conventional approaches that struggle with structural ambiguity in data.", 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 10, 'raw_content': 'RAG-Anything overcomes this by parsing the table into a comprehensive relational graph where headers and data cells become nodes connected through explicit row-of and column-of relationships. This structured representation enables the system to correctly isolate the Evidence Inference dataset context and identify "GloVe + LSTM - Attention" with a score of 0.506 as the optimal configuration. By explicitly preserving hierarchical table constraints that other methods often collapse or misinterpret, RAG-Anything ensures reliable reasoning across complex multi-dataset tabular structures.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 11, 'raw_content': 'A.3 CONTEXT-AWARE MULTIMODAL PROMPTING', 'content_type': 'title'}, {'page_no.': 13, 'index_on_page': 12, 'raw_content': 'These three prompts orchestrate structured, context-aware multimodal analysis with JSON-formatted outputs. They systematically guide the model to extract comprehensive descriptions of visual, tabular, and mathematical content while maintaining explicit alignment with surrounding information.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 13, 'raw_content': 'Vision Analysis Prompt. Figure 7 orchestrates comprehensive image-context integration. The prompt directs the model to systematically capture compositional elements, object relationships, visual attributes, stylistic features, dynamic actions, and technical components (e.g., charts), while establishing explicit connections to accompanying text. This approach transcends superficial description, enabling contextually-grounded interpretations that enhance knowledge retrieval and substantiation.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 14, 'raw_content': 'Table Analysis Prompt. Figure 8 structures systematic tabular content decomposition across multiple analytical dimensions: structural organization, column semantics, critical values, statistical patterns, and contextual relevance. Through precise terminology and numerical accuracy requirements, the prompt eliminates ambiguous generalizations and ensures faithful preservation of key indicators while maintaining coherent alignment with surrounding discourse.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 15, 'raw_content': 'Equation Analysis Prompt. Figure 9 prioritizes semantic interpretation over syntactic restatement of mathematical expressions. The prompt instructs comprehensive analysis of variable definitions, operational logic, theoretical foundations, inter-formula relationships, and practical applications. This methodology ensures mathematical content becomes integral to broader argumentative frameworks, supporting enhanced retrieval accuracy, analytical traceability, and reasoning coherence.', 'content_type': 'text'}, {'page_no.': 14, 'index_on_page': 3, 'raw_content': 'Figure 7: Vision analysis prompt for context-aware image interpretation and knowledge extraction.', 'content_type': 'text'}, {'page_no.': 14, 'index_on_page': 6, 'raw_content': 'Figure 8: Table analysis prompt for structured content decomposition and semantic understanding.', 'content_type': 'text'}, {'page_no.': 15, 'index_on_page': 3, 'raw_content': 'Figure 9: Equation analysis prompt for mathematical expression interpretation and integration.', 'content_type': 'text'}, {'page_no.': 15, 'index_on_page': 6, 'raw_content': 'Figure 10: Accuracy evaluation prompt for consistent factual assessment across question types.', 'content_type': 'text'}, {'page_no.': 15, 'index_on_page': 7, 'raw_content': 'A.4 ACCURACY EVALUATION PROMPT DESIGN', 'content_type': 'title'}, {'page_no.': 15, 'index_on_page': 8, 'raw_content': 'Figure 10 presents the standardized prompt specifically designed for systematic factual accuracy assessment of generated responses across multiple domains. The prompt establishes explicit evaluation criteria that prioritize content correctness over stylistic considerations, producing binary accuracy', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 1, 'raw_content': 'classifications accompanied by concise analytical justifications. All accuracy evaluations throughout our comprehensive experimental framework were conducted using GPT-4o-mini, ensuring consistent and reliable assessment standards across diverse question categories and specialized domains.', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 2, 'raw_content': 'A.5 CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG', 'content_type': 'title'}, {'page_no.': 16, 'index_on_page': 3, 'raw_content': 'While current multimodal RAG systems demonstrate promising capabilities, their limitations emerge most clearly through systematic analysis of failure cases. Understanding where and why these systems break down is crucial for advancing the field beyond current performance plateaus. Examining failure patterns helps identify fundamental architectural bottlenecks and design principles for more robust multimodal systems. Our investigation reveals two critical failure patterns exposing deeper systemic issues in multimodal RAG architectures. These patterns are not merely edge cases but reflect fundamental challenges in cross-modal information integration and structural reasoning:', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 7, 'raw_content': 'These failure modes illuminate key insights about current multimodal AI. They provide concrete directions for architectural innovations that could substantially improve system robustness.', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 20, 'raw_content': 'Case 1: Cross-Modal Misalignment. Figure 11 presents a particularly revealing failure scenario where all evaluated methods consistently produce incorrect answers despite having access to the necessary information. This universal failure across different architectures suggests fundamental limitations in how current systems handle noisy, heterogeneous multimodal data—a critical challenge as real-world applications inevitably involve imperfect, inconsistent information sources. The failure exposes two interconnected systemic issues that compound each other:', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 21, 'raw_content': 'Issue 1: Retrieval Bias Toward Text. Current RAG systems demonstrate pronounced bias toward textual passages. This occurs particularly when visual content lacks exact keyword matches. The bias persists even when queries contain explicit instructions to prioritize visual sources. This reveals a fundamental weakness in cross-modal attention mechanisms.', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 22, 'raw_content': 'The retrieved textual information, while topically related, often operates at a different granularity level than visual content. Images may contain precise, structured data such as specific numerical values,', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 1, 'raw_content': 'detailed diagrams, or exact spatial relationships. Corresponding text typically provides general, conceptual descriptions. This semantic misalignment introduces noise that actively misleads the reasoning process. The system attempts to reconcile incompatible levels of detail and specificity.', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 2, 'raw_content': 'Issue 2: Rigid Spatial Processing Patterns. Current visual processing models exhibit fundamental rigidity in spatial interpretation. Most systems default to sequential scanning patterns—top-to-bottom and left-to-right—that mirror natural reading conventions. While effective for simple text documents, this approach creates systematic failures with structurally complex real-world content. Many documents require non-conventional processing strategies. Tables demand column-wise interpretation, technical diagrams follow specific directional flows, and scientific figures embed critical information in unexpectedly positioned annotations. These structural variations are prevalent in professional documents, making adaptive spatial reasoning essential.', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 3, 'raw_content': "In the observed failure case, the correct answer required integrating visual elements in reverse order from the model's default processing sequence. The system's inability to recognize and adapt to this structural requirement led to systematic misinterpretation. This represents a fundamental architectural limitation where spatial reasoning remains static regardless of document context or query intent. When spatial processing patterns are misaligned with document structure, the extracted information becomes not merely incomplete but actively misleading. This structural noise compounds other processing errors and can lead to confident but entirely incorrect conclusions.", 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 4, 'raw_content': 'Case 2: Structural Noise in Ambiguous Table Layouts. As shown in Figure 12, all methods failed when confronted with a structurally ambiguous table. The primary failure stems from the table\'s confusing design: the GEM row lacks dedicated cell boundaries, and the "Joint" and "Slot" columns merge without clear separation. These structural irregularities create parsing ambiguities that systematically mislead extraction algorithms. This failure pattern reveals a critical vulnerability in current RAG systems. When table structures deviate from standard formatting conventions—through merged cells, unclear boundaries, or non-standard layouts—extraction methods consistently misinterpret cell relationships and conflate distinct data values. This exposes the brittleness of current approaches when faced with real-world document variations that deviate from clean, structured formats.', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 5, 'raw_content': 'The case highlights two essential directions for enhancing robustness. RAG systems require layout-aware parsing mechanisms that can recognize and adapt to structural irregularities rather than imposing rigid formatting assumptions. Additionally, integrating visual processing capabilities could significantly improve noise resilience, as visual models can leverage spatial relationships and contextual design cues that are lost in purely structural representations.', 'content_type': 'text'}]                            
                                            
                                            ####  Combined Knowledge Units  ####

combined_knowledge_units = [{'page_no.': 0, 'index_on_page': 1, 'raw_content': 'RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK', 'content_type': 'title'}, {'page_no.': 0, 'index_on_page': 2, 'raw_content': 'Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang*', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 3, 'raw_content': 'The University of Hong Kong', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 4, 'raw_content': 'zrguol01@hku.hk xubinrencs@gmail.com chaohuang75@gmail.com', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 5, 'raw_content': 'ABSTRACT', 'content_type': 'title'}, {'page_no.': 0, 'index_on_page': 6, 'raw_content': 'Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding Large Language Models beyond their static training limitations. However, a critical misalignment exists between current RAG capabilities and real-world information environments. Modern knowledge repositories are inherently multimodal, containing rich combinations of textual content, visual elements, structured tables, and mathematical expressions. Yet existing RAG frameworks are limited to textual content, creating fundamental gaps when processing multimodal documents. We present RAG-Anything, a unified framework that enables comprehensive knowledge retrieval across all modalities. Our approach reconceptualizes multimodal content as interconnected knowledge entities rather than isolated data types. The framework introduces dual-graph construction to capture both cross-modal relationships and textual semantics within a unified representation. We develop cross-modal hybrid retrieval that combines structural knowledge navigation with semantic matching. This enables effective reasoning over heterogeneous content where relevant evidence spans multiple modalities. RAG-Anything demonstrates superior performance on challenging multimodal benchmarks, achieving significant improvements over state-of-the-art methods. Performance gains become particularly pronounced on long documents where traditional approaches fail. Our framework establishes a new paradigm for multimodal knowledge access, eliminating the architectural fragmentation that constrains current systems. Our framework is open-sourced at: https://github.com/HKUDS/RAG-Anything.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 7, 'raw_content': '1 INTRODUCTION', 'content_type': 'title'}, {'page_no.': 0, 'index_on_page': 8, 'raw_content': 'Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding the knowledge boundaries of Large Language Models (LLM) beyond their static training limitations Zhang et al. (2025). By enabling dynamic retrieval and incorporation of external knowledge during inference, RAG systems transform static language models into adaptive, knowledge-aware systems. This capability has proven essential for applications requiring up-to-date information, domain-specific knowledge, or factual grounding that extends beyond pre-training corpora.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 9, 'raw_content': 'However, existing RAG frameworks focus exclusively on text-only knowledge while neglecting the rich multimodal information present in real-world documents. This limitation fundamentally misaligns with how information exists in authentic environments. Real-world knowledge repositories are inherently heterogeneous and multimodal Abootorabi et al. (2025). They contain rich combinations of textual content, visual elements, structured tables, and mathematical expressions across diverse document formats. This textual assumption forces existing RAG systems to either discard non-textual information entirely or flatten complex multimodal content into inadequate textual approximations.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 10, 'raw_content': 'The consequences of this limitation become particularly severe in document-intensive domains where multimodal content carries essential meaning. Academic research, financial analysis, and technical documentation represent prime examples of knowledge-rich environments. These domains fundamentally depend on visual and structured information. Critical insights are often encoded exclusively in non-textual formats. Such formats resist meaningful conversion to plain text.', 'content_type': 'text'}, {'page_no.': 0, 'index_on_page': 11, 'raw_content': 'The consequences of this limitation become particularly severe in knowledge-intensive domains where multimodal content carries essential meaning. Three representative scenarios illustrate the critical', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 1, 'raw_content': 'need for multimodal RAG capabilities. In Scientific Research, experimental results are primarily communicated through plots, diagrams, and statistical visualizations. These contain core discoveries that remain invisible to text-only systems. Financial Analysis relies heavily on market charts, correlation matrices, and performance tables. Investment insights are encoded in visual patterns rather than textual descriptions. Additionally, Medical Literature Analysis depends on radiological images, diagnostic charts, and clinical data tables. These contain life-critical information essential for accurate diagnosis and treatment decisions. Current RAG frameworks systematically exclude these vital knowledge sources across all three scenarios. This creates fundamental gaps that render them inadequate for real-world applications requiring comprehensive information understanding. Therefore, multimodal RAG emerges as a critical advancement. It is necessary to bridge these knowledge gaps and enable truly comprehensive intelligence across all modalities of human knowledge representation.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 2, 'raw_content': 'Addressing multimodal RAG presents three fundamental technical challenges that demand principled solutions. This makes it significantly more complex than traditional text-only approaches. The naive solution of converting all multimodal content to textual descriptions introduces severe information loss. Visual elements such as charts, diagrams, and spatial layouts contain semantic richness that cannot be adequately captured through text alone. These inherent limitations necessitate the design of effective technical components. Such components must be specifically designed to handle multimodal complexity and preserve the full spectrum of information contained within diverse content types.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 3, 'raw_content': 'Technical Challenges.   First, the unified multimodal representation challenge requires seamlessly integrating diverse information types. The system must preserve their unique characteristics and cross-modal relationships. This demands advanced multimodal encoders that can capture both intra-modal and inter-modal dependencies without losing essential visual semantics.   Second, the structure-aware decomposition challenge demands intelligent parsing of complex layouts. The system must maintain spatial and hierarchical relationships crucial for understanding. This requires specialized layout-aware parsing modules that can interpret document structure and preserve contextual positioning of multimodal elements.   Third, the cross-modal retrieval challenge necessitates sophisticated mechanisms that can navigate between different modalities. These mechanisms must reason over their interconnections during retrieval. This calls for cross-modal alignment systems capable of understanding semantic correspondences across text, images, and structured data. These challenges are amplified in long-context scenarios. Relevant evidence is dispersed across multiple modalities and sections, requiring coordinated reasoning across heterogeneous information sources.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 4, 'raw_content': 'Our Contributions. To address these challenges, we introduce RAG-Anything, a unified framework that fundamentally reimagines multimodal knowledge representation and retrieval. Our approach employs a dual-graph construction strategy that elegantly bridges the gap between cross-modal understanding and fine-grained textual semantics. Rather than forcing diverse modalities into text-centric pipelines, RAG-Anything constructs complementary knowledge graphs that preserve both multimodal contextual relationships and detailed textual knowledge. This design enables seamless integration of visual elements, structured data, and mathematical expressions within a unified retrieval framework. The system maintains semantic integrity across modalities while ensuring efficient cross-modal reasoning capabilities throughout the process.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 5, 'raw_content': 'Our cross-modal hybrid retrieval mechanism strategically combines structural knowledge navigation with semantic similarity matching. This architecture addresses the fundamental limitation of existing approaches that rely solely on embedding-based retrieval or keyword matching. RAG-Anything leverages explicit graph relationships to capture multi-hop reasoning patterns. It simultaneously employs dense vector representations to identify semantically relevant content that lacks direct structural connections. The framework introduces modality-aware query processing and cross-modal alignment systems. These enable textual queries to effectively access visual and structured information. This unified approach eliminates the architectural fragmentation that plagues current multimodal RAG systems. It delivers superior performance particularly on long-context documents where relevant evidence spans multiple modalities and document sections.', 'content_type': 'text'}, {'page_no.': 1, 'index_on_page': 6, 'raw_content': 'Experimental Validation. To validate the effectiveness of our proposed approach, we conduct comprehensive experiments on two challenging multimodal benchmarks: DocBench and MMLongBench. Our evaluation demonstrates that RAG-Anything achieves superior performance across diverse domains. The framework represents substantial improvements over state-of-the-art baselines. Notably, our performance gains become increasingly significant as content length increases. We observe particularly pronounced advantages on long-context materials. This validates our core hypothesis', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 1, 'raw_content': 'that dual-graph construction and cross-modal hybrid retrieval are essential for handling complex multimodal materials. Our ablation studies reveal that graph-based knowledge representation provides the primary performance gains. Traditional chunk-based approaches fail to capture the structural relationships critical for multimodal reasoning. Case studies further demonstrate that our framework excels at precise localization within complex layouts. The system effectively disambiguates similar terminology and navigates multi-panel visualizations through structure-aware retrieval mechanisms.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 2, 'raw_content': '2 THE RAG-ANYTHING FRAMEWORK', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 3, 'raw_content': '2.1 PRELIMINARY', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 4, 'raw_content': 'Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for dynamically expanding the knowledge boundaries of LLMs. While LLMs demonstrate exceptional reasoning capabilities, their knowledge remains static and bounded by training data cutoffs. This creates an ever-widening gap with the rapidly evolving information landscape. RAG systems address this critical limitation by enabling LLMs to retrieve and incorporate external knowledge sources during inference. This transforms them from static repositories into adaptive, knowledge-aware systems.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 5, 'raw_content': 'The Multimodal Reality: Beyond Text-Only RAG. Current RAG systems face a critical limitation that severely restricts their real-world deployment. Existing frameworks operate under the restrictive assumption that knowledge corpus consists exclusively of plain textual documents. This assumption fundamentally misaligns with how information exists in authentic environments. Real-world knowledge repositories are inherently heterogeneous and multimodal, containing rich combinations of textual content, visual elements, structured data, and mathematical expressions. These diverse knowledge sources span multiple document formats and presentation mediums, from research papers and technical slides to web pages and interactive documents.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 6, 'raw_content': '2.1.1 MOTIVATING RAG-ANYTHING', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 7, 'raw_content': 'This multimodal reality introduces fundamental technical challenges that expose the inadequacy of current text-only RAG approaches. Effective multimodal RAG requires unified indexing strategies that can handle disparate data types, cross-modal retrieval mechanisms that preserve semantic relationships across modalities, and sophisticated synthesis techniques that can coherently integrate diverse information sources. These challenges demand a fundamentally different architectural approach rather than incremental improvements to existing systems.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 8, 'raw_content': 'The RAG-Anything framework introduces a unified approach for retrieving and processing knowledge from heterogeneous multimodal information sources. Our system addresses the fundamental challenge of handling diverse data modalities and document formats within a retrieval pipeline. The framework comprises three core components: universal indexing for multimodal knowledge, cross-modal adaptive retrieval, and knowledge-enhanced response generation. This integrated design enables effective knowledge utilization across modalities while maintaining computational efficiency.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 9, 'raw_content': '2.2 UNIVERSAL REPRESENTATION FOR HETEROGENEOUS KNOWLEDGE', 'content_type': 'title'}, {'page_no.': 2, 'index_on_page': 10, 'raw_content': 'A key requirement for universal knowledge access is the ability to represent heterogeneous multimodal content in a unified, retrieval-oriented abstraction. Unlike existing pipelines that simply parse documents into text segments, RAG-Anything introduces Multimodal Knowledge Unification. This process decomposes raw inputs into atomic knowledge units while preserving their structural context and semantic alignment. For instance, RAG-Anything ensures that figures remain grounded in their captions, equations remain linked to surrounding definitions, and tables stay connected to explanatory narratives. This transforms heterogeneous files into a coherent substrate for cross-modal retrieval.', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 11, 'raw_content': 'Formally, each knowledge source   (e.g., a web page) is decomposed into atomic content units:', 'content_type': 'text'}, {'page_no.': 2, 'index_on_page': 13, 'raw_content': 'where each unit   consists of a modality type   and its corresponding raw content  . The content   represents the extracted information from the original knowledge source, processed in a modality-aware manner to preserve semantic integrity.', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 3, 'raw_content': 'To ensure high-fidelity extraction, RAG-Anything leverages specialized parsers for different content types. Text is segmented into coherent paragraphs or list items. Figures are extracted with associated metadata such as captions and cross-references. Tables are parsed into structured cells with headers and values. Mathematical expressions are converted into symbolic representations. The resulting   preserves both content and structural context within the source. This provides a faithful, modality-consistent representation. The decomposition abstracts diverse file formats into atomic units while maintaining their hierarchical order and contextual relationships. This canonicalization enables uniform processing, indexing, and retrieval of multimodal content within our framework.', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 4, 'raw_content': '2.2.1 DUAL-GRAPH CONSTRUCTION FOR MULTIMODAL KNOWLEDGE', 'content_type': 'title'}, {'page_no.': 3, 'index_on_page': 5, 'raw_content': 'While multimodal knowledge unification provides a uniform abstraction across modalities, directly constructing a single unified graph often risks overlooking modality-specific structural signals. The proposed RAG-Anything addresses this challenge through a dual-graph construction strategy. The system first builds a cross-modal knowledge graph that faithfully grounds non-textual modalities within their contextual environment. It then constructs a text-based knowledge graph using established text-centric extraction pipelines. These complementary graphs are merged through entity alignment. This design ensures accurate cross-modal grounding and comprehensive coverage of textual semantics, enabling richer knowledge representation and robust retrieval.', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 6, 'raw_content': "- Cross-Modal Knowledge Graph: Non-textual content like images, tables, and equations contains rich semantic information that traditional text-only approaches often overlook. To preserve this knowledge, RAG-Anything constructs a multimodal knowledge graph where non-text atomic units are transformed into structured graph entities. RAG-Anything leverages multimodal large language models to derive two complementary textual representations from each atomic content unit. The first is a detailed description   optimized for cross-modal retrieval. The second is an entity summary   containing key attributes such as entity name, type, and description for graph construction. The generation process is context-aware, processing each unit with its local neighborhood  , where   controls the contextual window size. This ensures representations accurately reflect each unit's role within the broader document structure.", 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 7, 'raw_content': 'Building on these textual representations, RAG-Anything constructs the graph structure using nontext units as anchor points. For each non-text unit  , the graph extraction routine   processes its description   to identify fine-grained entities and relations:', 'content_type': 'text'}, {'page_no.': 3, 'index_on_page': 9, 'raw_content': 'where   and   denote the sets of intra-chunk entities and their relations, respectively. Each atomic non-text unit is associated with a multimodal entity node   that serves as an anchor for', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 1, 'raw_content': 'its intra-chunk entities through explicit belongs_to edges:', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 4, 'raw_content': 'This construction preserves modality-specific grounding while ensuring non-textual content is contextualized by its textual neighborhood. This enables reliable cross-modal retrieval and reasoning.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 5, 'raw_content': "- Text-based Knowledge Graph: For text modality chunks, we construct a traditional text-based knowledge graph following established methodologies similar to LightRAG (Guo et al., 2024) and GraphRAG (Edge et al., 2024). The extraction process operates directly on textual content   where   text, leveraging named entity recognition and relation extraction techniques to identify entities and their semantic relationships. Given the rich semantic information inherent in textual content, multimodal context integration is not required for this component. The resulting text-based knowledge graph captures explicit knowledge and semantic connections present in textual portions of documents, complementing the multimodal graph's cross-modal grounding capabilities.", 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 6, 'raw_content': '2.2.2 GRAPH FUSION AND INDEX CREATION', 'content_type': 'title'}, {'page_no.': 4, 'index_on_page': 7, 'raw_content': 'The separate cross-modal and text-based knowledge graphs capture complementary aspects of document semantics. Integrating them creates a unified representation leveraging visual-textual associations and fine-grained textual relationships for enhanced retrieval.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 12, 'raw_content': 'where   denotes the embedding function tailored for each component type. Together, the unified knowledge graph   and the embedding table   constitute the complete retrieval index  . This provides both structural knowledge representation and dense vector space for efficient cross-modal similarity search during the subsequent retrieval stage.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 13, 'raw_content': '2.3 CROSS-MODAL HYBRID RETRIEVAL', 'content_type': 'title'}, {'page_no.': 4, 'index_on_page': 14, 'raw_content': 'The retrieval stage operates on the index   to identify relevant knowledge components for a given user query. Traditional RAG methods face significant limitations when dealing with multimodal documents. They typically rely on semantic similarity within single modalities and fail to capture the rich interconnections between visual, mathematical, tabular, and textual elements. To address these challenges, our framework introduces a cross-modal hybrid retrieval mechanism. This mechanism leverages structural knowledge and semantic representations across heterogeneous modalities.', 'content_type': 'text'}, {'page_no.': 4, 'index_on_page': 15, 'raw_content': 'Modality-Aware Query Encoding. Given a user query  , we first perform modality-aware query analysis to extract lexical cues and potential modality preferences embedded within the query. For instance, queries containing terms such as "figure," "chart," "table," or "equation" provide explicit signals about the expected modality of relevant information. We then compute a unified text embedding   using the same encoder employed during indexing, ensuring consistency between', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 1, 'raw_content': 'query and knowledge representations. This embedding-based approach enables cross-modal retrieval capabilities where textual queries can effectively access multimodal content through their shared representations, maintaining retrieval consistency while preserving cross-modal accessibility.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 2, 'raw_content': 'Hybrid Knowledge Retrieval Architecture. Recognizing that knowledge relevance manifests through both explicit structural connections and implicit semantic relationships, we design a hybrid retrieval architecture that strategically combines two complementary mechanisms.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 3, 'raw_content': '- (i) Structural Knowledge Navigation. This mechanism addresses the challenge of capturing explicit relationships and multi-hop reasoning patterns. Traditional keyword-based retrieval often fails to identify knowledge connected through intermediate entities or cross-modal relationships. To overcome this limitation, we exploit the structural properties encoded within our unified knowledge graph G. We employ keyword matching and entity recognition to locate relevant graph components. The retrieval process begins with exact entity matching against query terms.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 4, 'raw_content': 'We then perform strategic neighborhood expansion to include related entities and relationships within a specified hop distance. This structural approach proves particularly effective at uncovering high-level semantic connections and entity-relation patterns that span multiple modalities. It capitalizes on the rich cross-modal linkages established in our multimodal knowledge graph. The structural navigation yields candidate set   containing relevant entities, relationships, and their associated content chunks that provide comprehensive contextual information.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 5, 'raw_content': '- (ii) Semantic Similarity Matching. This mechanism addresses the challenge of identifying semantically relevant knowledge that lacks explicit structural connections. While structural navigation excels at following explicit relationships, it may miss relevant content that is semantically related but not directly connected in the graph topology. To bridge this gap, we conduct dense vector similarity search between the query embedding   and all components stored in embedding table  .', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 6, 'raw_content': 'This approach encompasses atomic content chunks across all modalities, graph entities, and relationship representations, enabling fine-grained semantic matching that can surface relevant knowledge even when traditional lexical or structural signals are absent. The learned embedding space captures nuanced semantic relationships and contextual similarities that complement the explicit structural signals from the navigation mechanism. This retrieval pathway returns the top-k most semantically similar chunks   ranked by cosine similarity scores, ensuring comprehensive coverage of both structurally and semantically relevant knowledge.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 7, 'raw_content': 'Candidate Pool Unification. Both retrieval pathways may return overlapping candidates with differing relevance signals. This necessitates a principled approach to unify and rank results. Retrieval candidates from both pathways are unified into a comprehensive candidate pool:  . Simply merging candidates would ignore distinct evidence each pathway provides. It would fail to account for redundancy between retrieved content.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 11, 'raw_content': '2.4 FROM RETRIEVAL TO SYNTHESIS', 'content_type': 'title'}, {'page_no.': 5, 'index_on_page': 12, 'raw_content': 'Effective multimodal question answering requires preserving rich visual semantics while maintaining coherent grounding across heterogeneous knowledge sources. Simple text-only approaches lose crucial visual information, while naive multimodal methods struggle with coherent cross-modal integration. Our synthesis stage addresses these challenges by systematically combining retrieved multimodal knowledge into comprehensive, evidence-grounded responses.', 'content_type': 'text'}, {'page_no.': 5, 'index_on_page': 13, 'raw_content': '- (i) Building Textual Context. Given the top-ranked retrieval candidates  , we construct a structured textual context. We concatenate textual representations of all retrieved components, includ', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\14861407bef7f68b31affea722351d2101dcf3994e2fdd92f50524cbefb3411b.jpg', 'content_type': 'table'}, {'page_no.': 6, 'index_on_page': 2, 'table_caption': 'Table 1: Statistics of Experimental Datasets.', 'content_type': 'table'}, {'page_no.': 6, 'index_on_page': 3, 'raw_content': 'ing entity summaries, relationship descriptions, and chunk contents. The concatenation incorporates appropriate delimiters to indicate modality types and hierarchical origins. This approach ensures the language model can effectively parse and reason over heterogeneous knowledge components.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 4, 'raw_content': '- (ii) Recovering Visual Content. For multimodal chunks corresponding to visual artifacts, we perform dereferencing to recover original visual content, creating  . This design maintains consistency with our unified embedding strategy. Textual proxies enable efficient retrieval while authentic visual content provides rich semantics necessary for sophisticated reasoning during synthesis.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 5, 'raw_content': 'The synthesis process jointly conditions on both the assembled comprehensive textual context and dereferenced visual artifacts using a vision-language model:', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 7, 'raw_content': 'where the VLM integrates information from query, textual context, and visual content. This unified conditioning enables sophisticated visual interpretation while maintaining grounding in retrieved evidence. The resulting responses are both visually informed and factually grounded.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 8, 'raw_content': '3 EVALUATION', 'content_type': 'title'}, {'page_no.': 6, 'index_on_page': 9, 'raw_content': '3.1 EXPERIMENTAL SETTINGS', 'content_type': 'title'}, {'page_no.': 6, 'index_on_page': 10, 'raw_content': 'Evaluation Datasets. We conduct comprehensive evaluations on two challenging multimodal Document Question Answering (DQA) benchmarks that reflect real-world complexity and diversity. DocBench (Zou et al., 2024) provides a rigorous testbed with 229 multimodal documents spanning five critical domains: Academia, Finance, Government, Laws, and News. The dataset includes 1,102 expert-crafted question-answer pairs. These documents are notably extensive, averaging 66 pages and approximately 46,377 tokens, which presents substantial challenges for long-context understanding.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 11, 'raw_content': 'MMLongBench (Ma et al., 2024) complements this evaluation by focusing specifically on long-context multimodal document comprehension. It features 135 documents across 7 diverse document types with 1,082 expert-annotated questions. Together, these benchmarks provide comprehensive coverage of the multimodal document understanding challenges that RAG-Anything aims to address. They ensure our evaluation captures both breadth across domains and depth in document complexity. Detailed dataset statistics and characteristics are provided in Appendix A.1.', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 12, 'raw_content': 'Baselines. We compare RAG-Anything against the following methods for performance evaluation:', 'content_type': 'text'}, {'page_no.': 6, 'index_on_page': 17, 'raw_content': 'Experimental Settings. In our experiments, we implement all baselines using GPT-4o-mini as the backbone LLM. Documents are parsed using MinerU (Wang et al., 2024) to extract text, images, tables, and equations for downstream RAG processing. For the retrieval pipeline, we employ the text-embedding-3-large model with 3072-dimensional embeddings. We use the bge-reranker-v2-m3 model for reranking. For graph-based RAG methods, we enforce a combined entity-and-relation token limit of 20,000 tokens and a chunk token limit of 12,000 tokens.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 2, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\f137776e8af0d7ce8bd6e318be2769a719ac6f4b1fb06f3adb4a40b32fbf9acc.jpg', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 2, 'table_caption': 'Table 2: Accuracy (%) on DocBench Dataset. Performance results with best scores highlighted in dark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance (Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types are categorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.).', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 4, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\57705f0111dadaea4f4a06fd5b20d4c30089c4619ce0e95c4a9ba9af82a04625.jpg', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 4, 'table_caption': 'Table 3: Accuracy (%) on MMLongBench across different domains and overall performance. Best results are highlighted in dark blue and second-best in light blue.. Domain categories include Research Reports/Introductions (Res.), Tutorials/Workshops (Tut.), Academic Papers (Acad.), Guidebooks (Guid.), Brochures (Broch.), Administration/Industry Files (Admin.), and Financial Reports (Fin.).', 'content_type': 'table'}, {'page_no.': 7, 'index_on_page': 5, 'raw_content': 'Outputs are constrained to a one-sentence format. For the baseline GPT-4o-mini in our QA scenario, documents are concatenated into image form with a maximum of 50 pages per document, rendered at 144 dpi. Finally, all query results are evaluated for accuracy by GPT-4o-mini.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 6, 'raw_content': '3.2 PERFORMANCE COMPARISON', 'content_type': 'title'}, {'page_no.': 7, 'index_on_page': 7, 'raw_content': 'Superior Performance and Cross-Domain Generalization. RAG-Anything demonstrates superior overall performance over baselines through its unified multimodal framework. Unlike LightRAG, which is restricted to text-only content processing, RAG-Anything treats text, images, tables, and equations as first-class entities. MMGraphRAG only adds basic image processing while treating tables and equations as plain text, missing crucial structural information. RAG-Anything introduces a comprehensive dual-graph construction strategy that preserves structural relationships across all modalities. This unified approach enables superior performance across both evaluation benchmarks.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 8, 'raw_content': 'Enhanced Long-Context Performance. RAG-Anything demonstrates superior performance on long-context documents. The framework excels where relevant evidence is dispersed across multiple modalities and sections. It achieves the best results in information-dense domains such as Research Reports and Financial Reports on MMLongBench. These improvements stem from the structured context injection mechanism. This mechanism integrates dual-graph construction for cross-page entity alignment. It combines semantic retrieval with structural navigation. The framework also employs modality-aware processing for efficient context window utilization. Unlike baselines that cannot uniformly process diverse modalities, RAG-Anything effectively captures scattered multimodal evidence. Its cross-modal hybrid retrieval architecture combines structural knowledge navigation with semantic similarity matching. This enables the framework to leverage both explicit relationships and implicit semantic connections across modalities.', 'content_type': 'text'}, {'page_no.': 7, 'index_on_page': 9, 'raw_content': "To systematically evaluate model performance across varying document lengths, we conducted comprehensive experiments on both datasets. As illustrated in Figure 2, RAG-Anything and MMGraphRAG exhibit comparable performance on shorter documents. However, RAG-Anything's advantages become increasingly pronounced as document length grows. On DocBench, the performance gap expands dramatically to over 13 points for documents exceeding 100 pages (68.2% vs.", 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 7, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\12f755ba148b16e967d1e4628d11a09039e7a1dfd51629985b340dbb3b7bc223.jpg', 'content_type': 'table'}, {'page_no.': 8, 'index_on_page': 7, 'table_caption': 'Table 4: Ablation study results on DocBench. The "Chunk-only" variant bypasses dual-graph construction and relies solely on traditional chunk-based retrieval, while "w/o Reranker" eliminates cross-modal reranking but preserves the core graph-based architecture.', 'content_type': 'table'}, {'page_no.': 8, 'index_on_page': 8, 'raw_content': '54.6% for 101-200 pages; 68.8% vs. 55.0% for 200+ pages). On MMLongBench, RAG-Anything demonstrates consistent improvements across all length categories, achieving accuracy gains of 3.4 points for 11-50 pages, 9.3 points for 51-100 pages, and 7.9 points for 101-200 pages. These findings confirm that our dual-graph construction and cross-modal hybrid retrieval mechanism is particularly effective for long-document reasoning tasks.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 9, 'raw_content': '3.3 ARCHITECTURAL VALIDATION WITH ABLATION STUDIES', 'content_type': 'title'}, {'page_no.': 8, 'index_on_page': 10, 'raw_content': 'To isolate and quantify the contributions of key architectural components in RAG-Anything, we conducted systematic ablation studies examining two critical design choices. Given that our approach fundamentally differs from existing methods through dual-graph construction and hybrid retrieval, we specifically evaluated: i) Chunk-only, which bypasses graph construction entirely and relies solely on traditional chunk-based retrieval, and ii) w/o Reranker, which eliminates the cross-modal reranking component while preserving the core graph-based architecture.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 11, 'raw_content': 'As demonstrated in Table 4, the results validate our architectural design through striking performance variations.   Graph Construction is Essential. The chunk-only variant achieves merely   accuracy with substantial cross-domain drops. This demonstrates that traditional chunking fails to capture structural and cross-modal relationships essential for multimodal documents.   Reranking Provides Marginal Gains. Removing the reranker yields only a modest decline to  , while the full model achieves   accuracy. This indicates that cross-modal reranking provides valuable refinement, but primary gains stem from our graph-based retrieval and cross-modal integration.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 12, 'raw_content': '3.4 CASE STUDIES', 'content_type': 'title'}, {'page_no.': 8, 'index_on_page': 13, 'raw_content': 'Multimodal documents contain rich structural information within each modality. Understanding these intra-modal structures is crucial for accurate reasoning. We analyze two representative cases from DocBench to demonstrate how RAG-Anything leverages these structures. These cases highlight a key limitation of existing methods. Baselines either rely on superficial textual cues or flatten complex visual elements into plain text. In contrast, RAG-Anything builds modality-aware graphs that preserve essential relationships (e.g., table header   cell   unit edges; panel   caption   axis edges). This enables precise reasoning over complex document layouts.', 'content_type': 'text'}, {'page_no.': 8, 'index_on_page': 14, 'raw_content': '- Case 1: Multi-panel Figure Interpretation. This case examines a common scenario in academic literature. Researchers often need to compare results across different experimental conditions. These results are typically presented in multi-panel visualizations. Figure 3 shows a challenging t-SNE', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 7, 'raw_content': 'visualization with multiple subpanels. The query requires distinguishing between two related but distinct panels. RAG-Anything constructs a visual layout graph where panels, axis titles, legends, and captions become nodes. Key edges encode semantic relationships. Panels contain specific plots. Captions provide contextual information. Subfigures relate hierarchically. This structure guides the retriever to focus on the style-space panel for comparing cluster separation patterns. The system avoids confusion from the adjacent content space panel. This panel shows less clear distinctions.', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 14, 'raw_content': '- Case 2: Financial Table Navigation. This case addresses a common challenge in financial document analysis. Analysts must extract specific metrics from tables with similar terminology and multiple time periods. Figure 4 shows this scenario. The query involves resolving ambiguous financial terms and selecting the correct column for a specified year.', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 15, 'raw_content': 'RAG-Anything transforms the financial report table into a structured graph. Each row header, column header (year), data cell, and unit becomes a node. The edges capture key relationships: row-of, column-of, header-applies-to, and unit-of. This structure enables precise navigation. The retriever focuses on the row "Wages and salaries" and the column for "2020". It directs attention to the target cell (26,778 million). The system successfully disambiguates nearby entries like "Share-based payments." Competing methods treat tables as linear text. They often confuse numerical spans and years. This leads to significantly inaccurate answers. RAG-Anything explicitly models relationships within the table. It achieves precise selection and numeric grounding. This ensures accurate responses.', 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 16, 'raw_content': "- Key Insights. Both cases demonstrate how RAG-Anything's structure-aware design delivers targeted advantages. Our approach transforms documents into explicit graph representations. These graphs capture intra-modal relationships that traditional methods miss. In figures, connections between panels, captions, and axes enable panel-level comparisons. This goes beyond keyword matching. In tables, row-column-unit graphs ensure accurate identification through modeling.", 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 17, 'raw_content': "This structure-aware retrieval design reduces confusion from repeated terminology and complex layouts. Traditional RAG systems struggle with these scenarios due to lack of structural understanding. Even MMGraphRAG fails here because it only considers image modality entities. It ignores other modality entities like table cells, row headers, and column headers. RAG-Anything's comprehensive graph representation captures all modality-specific entities and their relationships. This enables precise, modality-specific grounding that leads to consistent improvements in document Q&A tasks requiring fine-grained localization. Additional cases are available in Appendix A.2.", 'content_type': 'text'}, {'page_no.': 9, 'index_on_page': 18, 'raw_content': '4 RELATED WORK', 'content_type': 'title'}, {'page_no.': 9, 'index_on_page': 19, 'raw_content': '- Graph-Enhanced Retrieval-Augmented Generation. Large language models struggle with long-context inputs and multi-hop queries, failing to precisely locate dispersed evidence (Zhang et al.,', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 1, 'raw_content': '2025). Graph structures address this limitation by introducing explicit relational modeling, improving both retrieval efficiency and reasoning accuracy (Bei et al., 2025).', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 2, 'raw_content': "Since GraphRAG (Edge et al., 2024), research has evolved along two complementary directions. First, graph construction approaches optimize structures for retrieval efficiency, ranging from LightRAG's (Guo et al., 2024) sparsified indices to neural models like GNN-RAG (Mavromatis & Karypis, 2024) and memory-augmented variants like HippoRAG (Jimenez Gutierrez et al., 2024). Second, knowledge aggregation approaches integrate information for multi-level reasoning through hierarchical methods like RAPTOR (Sarthi et al., 2024) and ArchRAG (Wang et al., 2025). Despite these advances, existing systems remain text-centric with homogeneous inputs. This limits their applicability to multimodal documents and constrains robust reasoning over heterogeneous content. RAG-Anything addresses this gap by extending GraphRAG to all modalities.", 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 3, 'raw_content': '- Multimodal Retrieval-Augmented Generation. Multimodal RAG represents a natural evolution from text-based RAG systems, addressing the need to integrate external knowledge from diverse data modalities for comprehensive response generation (Abootorabi et al., 2025). However, current approaches are fundamentally constrained by their reliance on modality-specific architectures. Existing methods demonstrate these constraints across domains: VideoRAG (Ren et al., 2025) employs dual-channel architectures for video understanding while MM-VID (Lin et al., 2023) converts videos to text, losing visual information; VisRAG (Yu et al., 2025) preserves document layouts as images but misses granular relationships; MMGraphRAG (Wan & Yu, 2025) links scene graphs with textual representations but suffers from structural blindness—treating tables and formulas as plain text without proper entity extraction, losing structural information for reasoning.', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 4, 'raw_content': 'The fundamental problem underlying these limitations is architectural fragmentation. Current systems require specialized processing pipelines for each modality. This creates poor generalizability as new modalities demand custom architectures and fusion mechanisms. Such fragmentation introduces cross-modal alignment difficulties, modality biases, and information bottlenecks. These issues systematically compromise system performance and scalability. RAG-Anything addresses this fragmentation through a unified graph-based framework. Our approach processes all modalities with consistent structured modeling. This eliminates architectural constraints while preserving multimodal information integrity. The result is seamless cross-modal reasoning across heterogeneous content.', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 5, 'raw_content': '5 CONCLUSION', 'content_type': 'title'}, {'page_no.': 10, 'index_on_page': 6, 'raw_content': 'RAG-Anything introduces a paradigm shift in multimodal retrieval through its unified graph-based framework. Our core technical innovation is the dual-graph construction strategy that seamlessly integrates cross-modal and text-based knowledge graphs. Rather than forcing diverse modalities into text-centric pipelines that lose critical structural information, our approach fundamentally reconceptualizes multimodal content as interconnected knowledge entities with rich semantic relationships. The hybrid retrieval mechanism strategically combines structural navigation with semantic matching, enabling precise reasoning over complex document layouts. Comprehensive evaluation demonstrates superior performance on long-context documents, particularly those exceeding 100 pages where traditional methods fail. This work establishes a new foundation for multimodal RAG systems that can handle the heterogeneous nature of diverse information landscapes.', 'content_type': 'text'}, {'page_no.': 10, 'index_on_page': 7, 'raw_content': 'Our analysis in Appendix A.5 reveals critical challenges facing current multimodal RAG systems. Two fundamental issues emerge through systematic failure case examination. First, systems exhibit text-centric retrieval bias, preferentially accessing textual sources even when queries explicitly require visual information. Second, rigid spatial processing patterns fail to adapt to non-standard document layouts. These limitations manifest in cross-modal misalignment scenarios and structurally ambiguous tables. The findings highlight the need for adaptive spatial reasoning and layout-aware parsing mechanisms to handle real-world multimodal document complexity.', 'content_type': 'text'}, {'page_no.': 11, 'index_on_page': 1, 'raw_content': 'REFERENCES', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 1, 'raw_content': 'A APPENDIX', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 2, 'raw_content': "This appendix provides comprehensive supporting materials for our experimental evaluation and implementation details. Section A.1 presents detailed dataset statistics for the DocBench and MMLongBench multi-modal benchmarks, including document type distributions and complexity metrics. Section A.2 showcases additional case studies that demonstrate RAG-Anything's structure-aware capabilities across diverse multimodal content understanding tasks. Section A.3 documents the complete set of multimodal analysis prompts for vision, table, and equation processing that enable context-aware interpretation. Section A.4 provides the standardized accuracy evaluation prompt used for consistent response assessment across all experimental conditions.", 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 3, 'raw_content': 'A.1 DATASET CHARACTERISTICS AND STATISTICS', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 5, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\33981691987ca5749049465db763f9c0e43875229ba6d96c9745466747165c90.jpg', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 5, 'table_caption': 'Table 5: Document type distribution and statistics for the DocBench benchmark.', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 7, 'table_image_path': 'C:\\Users\\Hp\\MinerU\\RAG_for_Anything.pdf-5fed3e90-5818-479c-aeda-c1079ca562df\\images\\968b2751f1362050c4c4adf30dfc6ad94c9d6eab5836a3535a145576ab107877.jpg', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 7, 'table_caption': 'Table 6: Document type distribution and statistics for the MMLongBench benchmark.', 'content_type': 'table'}, {'page_no.': 12, 'index_on_page': 8, 'raw_content': 'Tables 5 and 6 present the distribution of document types across the DocBench and MMLong-Bench benchmarks. • DocBench encompasses medium- to long-length documents spanning various domains, including legal, governmental, and financial files. Financial reports represent the most extensive category, averaging 192 pages per document, while the News category consists of concise single-page newspapers. • MMLongBench demonstrates a broader spectrum of document types and lengths. Research reports, tutorials, and academic papers maintain moderate lengths of 35–58 pages on average, while guidebooks extend to approximately 78 pages. Brochures and administrative files remain relatively compact, whereas financial reports again emerge as the longest category.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 9, 'raw_content': 'Collectively, these two benchmarks provide comprehensive coverage ranging from brief news articles to extensive technical and financial documentation. This establishes diverse and challenging evaluation contexts for multimodal document understanding tasks.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 10, 'raw_content': 'A.2 ADDITIONAL CASE STUDIES', 'content_type': 'title'}, {'page_no.': 12, 'index_on_page': 12, 'raw_content': 'GPT-4o-mini: The removal of the dual coattention mechanism from the GCAN sub-model resulted in the lowest accuracy for Twitter15.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 13, 'raw_content': 'MMGraphRAG: The removal of the source tweet from the GCAN model resulted in the lowest accuracy for Twitter15, as indicated by significant accuracy drops in the ablation analysis in Figure 4.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 14, 'raw_content': 'LightRAG: Removing the source tweet from the GCAN model resulted in the lowest accuracy for Twitter15, as indicated by a significant drop in performance when using the sub-model that excluded both source tweet embeddings and dual co-attention.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 15, 'raw_content': 'RAG-Anything(Correct): The removal of the source tweet embeddings and dual co-attention, indicated as model configuration "S-A," resulted in the lowest accuracy for Twitter15.', 'content_type': 'text'}, {'page_no.': 12, 'index_on_page': 17, 'raw_content': '- Demonstrating Visual Reasoning Capabilities. Figure 5 illustrates how RAG-Anything handles complex visual reasoning tasks involving chart interpretation. The query asks which GCAN sub-model component removal yields the lowest accuracy on Twitter15. Traditional approaches struggle', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 1, 'raw_content': 'with spatial relationships between visual elements. RAG-Anything addresses this challenge by constructing a structured graph representation of the bar plot. Bars, axis labels, and legends become interconnected nodes. These are linked by semantic relations such as bar-of and label-applies-to.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 2, 'raw_content': 'This graph-based approach enables precise alignment between visual and textual elements. The system correctly identifies the bar labeled "-S-A" (removing source tweet embeddings and dual co-attention) and its corresponding accuracy value as the lowest performer. Baseline methods that flatten visual information often misinterpret spatial relationships. They frequently conflate nearby components. RAG-Anything\'s structured representation preserves critical visual-textual associations. This leads to accurate query resolution and proper attribution of performance drops to "-S-A".', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 9, 'raw_content': "- Handling Complex Tabular Structures. Figure 6 showcases RAG-Anything's ability to navigate intricate tabular data where structural disambiguation is crucial. The query seeks the model combination achieving the highest AUPRC value for the Evidence Inference dataset—a task complicated by repeated row labels across multiple datasets within the same table. This scenario highlights a fundamental limitation of conventional approaches that struggle with structural ambiguity in data.", 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 10, 'raw_content': 'RAG-Anything overcomes this by parsing the table into a comprehensive relational graph where headers and data cells become nodes connected through explicit row-of and column-of relationships. This structured representation enables the system to correctly isolate the Evidence Inference dataset context and identify "GloVe + LSTM - Attention" with a score of 0.506 as the optimal configuration. By explicitly preserving hierarchical table constraints that other methods often collapse or misinterpret, RAG-Anything ensures reliable reasoning across complex multi-dataset tabular structures.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 11, 'raw_content': 'A.3 CONTEXT-AWARE MULTIMODAL PROMPTING', 'content_type': 'title'}, {'page_no.': 13, 'index_on_page': 12, 'raw_content': 'These three prompts orchestrate structured, context-aware multimodal analysis with JSON-formatted outputs. They systematically guide the model to extract comprehensive descriptions of visual, tabular, and mathematical content while maintaining explicit alignment with surrounding information.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 13, 'raw_content': 'Vision Analysis Prompt. Figure 7 orchestrates comprehensive image-context integration. The prompt directs the model to systematically capture compositional elements, object relationships, visual attributes, stylistic features, dynamic actions, and technical components (e.g., charts), while establishing explicit connections to accompanying text. This approach transcends superficial description, enabling contextually-grounded interpretations that enhance knowledge retrieval and substantiation.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 14, 'raw_content': 'Table Analysis Prompt. Figure 8 structures systematic tabular content decomposition across multiple analytical dimensions: structural organization, column semantics, critical values, statistical patterns, and contextual relevance. Through precise terminology and numerical accuracy requirements, the prompt eliminates ambiguous generalizations and ensures faithful preservation of key indicators while maintaining coherent alignment with surrounding discourse.', 'content_type': 'text'}, {'page_no.': 13, 'index_on_page': 15, 'raw_content': 'Equation Analysis Prompt. Figure 9 prioritizes semantic interpretation over syntactic restatement of mathematical expressions. The prompt instructs comprehensive analysis of variable definitions, operational logic, theoretical foundations, inter-formula relationships, and practical applications. This methodology ensures mathematical content becomes integral to broader argumentative frameworks, supporting enhanced retrieval accuracy, analytical traceability, and reasoning coherence.', 'content_type': 'text'}, {'page_no.': 14, 'index_on_page': 3, 'raw_content': 'Figure 7: Vision analysis prompt for context-aware image interpretation and knowledge extraction.', 'content_type': 'text'}, {'page_no.': 14, 'index_on_page': 6, 'raw_content': 'Figure 8: Table analysis prompt for structured content decomposition and semantic understanding.', 'content_type': 'text'}, {'page_no.': 15, 'index_on_page': 3, 'raw_content': 'Figure 9: Equation analysis prompt for mathematical expression interpretation and integration.', 'content_type': 'text'}, {'page_no.': 15, 'index_on_page': 6, 'raw_content': 'Figure 10: Accuracy evaluation prompt for consistent factual assessment across question types.', 'content_type': 'text'}, {'page_no.': 15, 'index_on_page': 7, 'raw_content': 'A.4 ACCURACY EVALUATION PROMPT DESIGN', 'content_type': 'title'}, {'page_no.': 15, 'index_on_page': 8, 'raw_content': 'Figure 10 presents the standardized prompt specifically designed for systematic factual accuracy assessment of generated responses across multiple domains. The prompt establishes explicit evaluation criteria that prioritize content correctness over stylistic considerations, producing binary accuracy', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 1, 'raw_content': 'classifications accompanied by concise analytical justifications. All accuracy evaluations throughout our comprehensive experimental framework were conducted using GPT-4o-mini, ensuring consistent and reliable assessment standards across diverse question categories and specialized domains.', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 2, 'raw_content': 'A.5 CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG', 'content_type': 'title'}, {'page_no.': 16, 'index_on_page': 3, 'raw_content': 'While current multimodal RAG systems demonstrate promising capabilities, their limitations emerge most clearly through systematic analysis of failure cases. Understanding where and why these systems break down is crucial for advancing the field beyond current performance plateaus. Examining failure patterns helps identify fundamental architectural bottlenecks and design principles for more robust multimodal systems. Our investigation reveals two critical failure patterns exposing deeper systemic issues in multimodal RAG architectures. These patterns are not merely edge cases but reflect fundamental challenges in cross-modal information integration and structural reasoning:', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 7, 'raw_content': 'These failure modes illuminate key insights about current multimodal AI. They provide concrete directions for architectural innovations that could substantially improve system robustness.', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 20, 'raw_content': 'Case 1: Cross-Modal Misalignment. Figure 11 presents a particularly revealing failure scenario where all evaluated methods consistently produce incorrect answers despite having access to the necessary information. This universal failure across different architectures suggests fundamental limitations in how current systems handle noisy, heterogeneous multimodal data—a critical challenge as real-world applications inevitably involve imperfect, inconsistent information sources. The failure exposes two interconnected systemic issues that compound each other:', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 21, 'raw_content': 'Issue 1: Retrieval Bias Toward Text. Current RAG systems demonstrate pronounced bias toward textual passages. This occurs particularly when visual content lacks exact keyword matches. The bias persists even when queries contain explicit instructions to prioritize visual sources. This reveals a fundamental weakness in cross-modal attention mechanisms.', 'content_type': 'text'}, {'page_no.': 16, 'index_on_page': 22, 'raw_content': 'The retrieved textual information, while topically related, often operates at a different granularity level than visual content. Images may contain precise, structured data such as specific numerical values,', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 1, 'raw_content': 'detailed diagrams, or exact spatial relationships. Corresponding text typically provides general, conceptual descriptions. This semantic misalignment introduces noise that actively misleads the reasoning process. The system attempts to reconcile incompatible levels of detail and specificity.', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 2, 'raw_content': 'Issue 2: Rigid Spatial Processing Patterns. Current visual processing models exhibit fundamental rigidity in spatial interpretation. Most systems default to sequential scanning patterns—top-to-bottom and left-to-right—that mirror natural reading conventions. While effective for simple text documents, this approach creates systematic failures with structurally complex real-world content. Many documents require non-conventional processing strategies. Tables demand column-wise interpretation, technical diagrams follow specific directional flows, and scientific figures embed critical information in unexpectedly positioned annotations. These structural variations are prevalent in professional documents, making adaptive spatial reasoning essential.', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 3, 'raw_content': "In the observed failure case, the correct answer required integrating visual elements in reverse order from the model's default processing sequence. The system's inability to recognize and adapt to this structural requirement led to systematic misinterpretation. This represents a fundamental architectural limitation where spatial reasoning remains static regardless of document context or query intent. When spatial processing patterns are misaligned with document structure, the extracted information becomes not merely incomplete but actively misleading. This structural noise compounds other processing errors and can lead to confident but entirely incorrect conclusions.", 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 4, 'raw_content': 'Case 2: Structural Noise in Ambiguous Table Layouts. As shown in Figure 12, all methods failed when confronted with a structurally ambiguous table. The primary failure stems from the table\'s confusing design: the GEM row lacks dedicated cell boundaries, and the "Joint" and "Slot" columns merge without clear separation. These structural irregularities create parsing ambiguities that systematically mislead extraction algorithms. This failure pattern reveals a critical vulnerability in current RAG systems. When table structures deviate from standard formatting conventions—through merged cells, unclear boundaries, or non-standard layouts—extraction methods consistently misinterpret cell relationships and conflate distinct data values. This exposes the brittleness of current approaches when faced with real-world document variations that deviate from clean, structured formats.', 'content_type': 'text'}, {'page_no.': 17, 'index_on_page': 5, 'raw_content': 'The case highlights two essential directions for enhancing robustness. RAG systems require layout-aware parsing mechanisms that can recognize and adapt to structural irregularities rather than imposing rigid formatting assumptions. Additionally, integrating visual processing capabilities could significantly improve noise resilience, as visual models can leverage spatial relationships and contextual design cues that are lost in purely structural representations.', 'content_type': 'text'}]

                                            ####  RAW OUTPUT OF minerU  ####

"""
Text/Title:
1- PDF info contains list, which contains the dict of blocks. It could be "dict of discard blocks" or "dict of para blocks".
2- Discarded blocks are those which are breaking the continuity of the meaning for content. 
3- Para blocks are the relevant as either it contains the textual content or non-textual content.
4- For textual content, we are fetching titles (which are basically headings) or text (paragraphs)
5- For para blocks that have the type "title", it contains the headings, and those with type "text" contains paragraphs.
6- Para blocks with type either title or text, contain multiple dicts, each dict contain "lines".
7- Each line contains the list of dict, will have only one span-value. 
8- Span-value contains list, which has dict - which will contain the text of para.
9- In this way, we have the multiple lines dicts, each containing span which further contains text content.


Tables:
1- PDF info contains list, which contains dict of blocks. In case of para blocks, it can have para block type as "table".
2- This type of para_block, contains the multiple dicts, each contains blocks. 
3- Each block contains multiple dicts, each dict contain "lines".
7- Each line-value contains the list of dict, will have only one span-value. 
8- Span-value contains list, which has dict - which will contain the table image path or table caption.
9- In this way, it is possible that one line may contain the span for table caption, and other line may contain the span for image path.

"""




Parsed_minerU_raw = {
    "pdf_info": [
        {
            "para_blocks": [
                {
                    "bbox": [
                        105,
                        79,
                        487,
                        97
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                79,
                                487,
                                97
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        79,
                                        487,
                                        97
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        110,
                        114,
                        387,
                        126
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                110,
                                114,
                                387,
                                126
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        110,
                                        114,
                                        387,
                                        126
                                    ],
                                    "type": "text",
                                    "content": "Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang*"
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        112,
                        126,
                        234,
                        137
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                112,
                                126,
                                234,
                                137
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        112,
                                        126,
                                        234,
                                        137
                                    ],
                                    "type": "text",
                                    "content": "The University of Hong Kong"
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        112,
                        137,
                        432,
                        148
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                112,
                                137,
                                432,
                                148
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        112,
                                        137,
                                        432,
                                        148
                                    ],
                                    "type": "text",
                                    "content": "zrguol01@hku.hk xubinrencs@gmail.com chaohuang75@gmail.com"
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        276,
                        177,
                        335,
                        190
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                276,
                                177,
                                335,
                                190
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        276,
                                        177,
                                        335,
                                        190
                                    ],
                                    "type": "text",
                                    "content": "ABSTRACT"
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        140,
                        201,
                        471,
                        434
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                140,
                                201,
                                471,
                                434
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        140,
                                        201,
                                        471,
                                        434
                                    ],
                                    "type": "text",
                                    "content": "Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding Large Language Models beyond their static training limitations. However, a critical misalignment exists between current RAG capabilities and real-world information environments. Modern knowledge repositories are inherently multimodal, containing rich combinations of textual content, visual elements, structured tables, and mathematical expressions. Yet existing RAG frameworks are limited to textual content, creating fundamental gaps when processing multimodal documents. We present RAG-Anything, a unified framework that enables comprehensive knowledge retrieval across all modalities. Our approach reconceptualizes multimodal content as interconnected knowledge entities rather than isolated data types. The framework introduces dual-graph construction to capture both cross-modal relationships and textual semantics within a unified representation. We develop cross-modal hybrid retrieval that combines structural knowledge navigation with semantic matching. This enables effective reasoning over heterogeneous content where relevant evidence spans multiple modalities. RAG-Anything demonstrates superior performance on challenging multimodal benchmarks, achieving significant improvements over state-of-the-art methods. Performance gains become particularly pronounced on long documents where traditional approaches fail. Our framework establishes a new paradigm for multimodal knowledge access, eliminating the architectural fragmentation that constrains current systems. Our framework is open-sourced at: https://github.com/HKUDS/RAG-Anything."
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        106,
                        453,
                        208,
                        464
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                453,
                                208,
                                464
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        453,
                                        208,
                                        464
                                    ],
                                    "type": "text",
                                    "content": "1 INTRODUCTION"
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        477,
                        506,
                        544
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                477,
                                506,
                                544
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        477,
                                        506,
                                        544
                                    ],
                                    "type": "text",
                                    "content": "Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding the knowledge boundaries of Large Language Models (LLM) beyond their static training limitations Zhang et al. (2025). By enabling dynamic retrieval and incorporation of external knowledge during inference, RAG systems transform static language models into adaptive, knowledge-aware systems. This capability has proven essential for applications requiring up-to-date information, domain-specific knowledge, or factual grounding that extends beyond pre-training corpora."
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        104,
                        548,
                        506,
                        627
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                548,
                                506,
                                627
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        548,
                                        506,
                                        627
                                    ],
                                    "type": "text",
                                    "content": "However, existing RAG frameworks focus exclusively on text-only knowledge while neglecting the rich multimodal information present in real-world documents. This limitation fundamentally misaligns with how information exists in authentic environments. Real-world knowledge repositories are inherently heterogeneous and multimodal Abootorabi et al. (2025). They contain rich combinations of textual content, visual elements, structured tables, and mathematical expressions across diverse document formats. This textual assumption forces existing RAG systems to either discard non-textual information entirely or flatten complex multimodal content into inadequate textual approximations."
                                }
                            ]
                        }
                    ],
                    "index": 9
                },
                {
                    "bbox": [
                        104,
                        632,
                        504,
                        688
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                632,
                                504,
                                688
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        632,
                                        504,
                                        688
                                    ],
                                    "type": "text",
                                    "content": "The consequences of this limitation become particularly severe in document-intensive domains where multimodal content carries essential meaning. Academic research, financial analysis, and technical documentation represent prime examples of knowledge-rich environments. These domains fundamentally depend on visual and structured information. Critical insights are often encoded exclusively in non-textual formats. Such formats resist meaningful conversion to plain text."
                                }
                            ]
                        }
                    ],
                    "index": 10
                },
                {
                    "bbox": [
                        104,
                        692,
                        506,
                        715
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                692,
                                506,
                                715
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        692,
                                        506,
                                        715
                                    ],
                                    "type": "text",
                                    "content": "The consequences of this limitation become particularly severe in knowledge-intensive domains where multimodal content carries essential meaning. Three representative scenarios illustrate the critical"
                                }
                            ]
                        }
                    ],
                    "index": 11
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        340,
                        38
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                340,
                                38
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        340,
                                        38
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        116,
                        720,
                        256,
                        732
                    ],
                    "type": "page_footnote",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                116,
                                720,
                                256,
                                732
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        116,
                                        720,
                                        256,
                                        732
                                    ],
                                    "type": "text",
                                    "content": "*Corresponding Author: Chao Huang"
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        14,
                        223,
                        37,
                        567
                    ],
                    "type": "aside_text",
                    "angle": 270,
                    "lines": [
                        {
                            "bbox": [
                                14,
                                223,
                                37,
                                567
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        14,
                                        223,
                                        37,
                                        567
                                    ],
                                    "type": "text",
                                    "content": "arXiv:2510.12323v1 [cs.AI] 14 Oct 2025"
                                }
                            ]
                        }
                    ],
                    "index": 13
                },
                {
                    "bbox": [
                        302,
                        751,
                        309,
                        761
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                309,
                                761
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        309,
                                        761
                                    ],
                                    "type": "text",
                                    "content": "1"
                                }
                            ]
                        }
                    ],
                    "index": 14
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 0
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        107,
                        82,
                        504,
                        203
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                82,
                                504,
                                203
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        82,
                                        504,
                                        203
                                    ],
                                    "type": "text",
                                    "content": "need for multimodal RAG capabilities. In Scientific Research, experimental results are primarily communicated through plots, diagrams, and statistical visualizations. These contain core discoveries that remain invisible to text-only systems. Financial Analysis relies heavily on market charts, correlation matrices, and performance tables. Investment insights are encoded in visual patterns rather than textual descriptions. Additionally, Medical Literature Analysis depends on radiological images, diagnostic charts, and clinical data tables. These contain life-critical information essential for accurate diagnosis and treatment decisions. Current RAG frameworks systematically exclude these vital knowledge sources across all three scenarios. This creates fundamental gaps that render them inadequate for real-world applications requiring comprehensive information understanding. Therefore, multimodal RAG emerges as a critical advancement. It is necessary to bridge these knowledge gaps and enable truly comprehensive intelligence across all modalities of human knowledge representation."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        107,
                        209,
                        504,
                        285
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                209,
                                504,
                                285
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        209,
                                        504,
                                        285
                                    ],
                                    "type": "text",
                                    "content": "Addressing multimodal RAG presents three fundamental technical challenges that demand principled solutions. This makes it significantly more complex than traditional text-only approaches. The naive solution of converting all multimodal content to textual descriptions introduces severe information loss. Visual elements such as charts, diagrams, and spatial layouts contain semantic richness that cannot be adequately captured through text alone. These inherent limitations necessitate the design of effective technical components. Such components must be specifically designed to handle multimodal complexity and preserve the full spectrum of information contained within diverse content types."
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        107,
                        291,
                        504,
                        434
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                291,
                                504,
                                434
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "text",
                                    "content": "Technical Challenges. "
                                },
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\bullet"
                                },
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "text",
                                    "content": " First, the unified multimodal representation challenge requires seamlessly integrating diverse information types. The system must preserve their unique characteristics and cross-modal relationships. This demands advanced multimodal encoders that can capture both intra-modal and inter-modal dependencies without losing essential visual semantics. "
                                },
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\bullet"
                                },
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "text",
                                    "content": " Second, the structure-aware decomposition challenge demands intelligent parsing of complex layouts. The system must maintain spatial and hierarchical relationships crucial for understanding. This requires specialized layout-aware parsing modules that can interpret document structure and preserve contextual positioning of multimodal elements. "
                                },
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\bullet"
                                },
                                {
                                    "bbox": [
                                        107,
                                        291,
                                        504,
                                        434
                                    ],
                                    "type": "text",
                                    "content": " Third, the cross-modal retrieval challenge necessitates sophisticated mechanisms that can navigate between different modalities. These mechanisms must reason over their interconnections during retrieval. This calls for cross-modal alignment systems capable of understanding semantic correspondences across text, images, and structured data. These challenges are amplified in long-context scenarios. Relevant evidence is dispersed across multiple modalities and sections, requiring coordinated reasoning across heterogeneous information sources."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        107,
                        440,
                        504,
                        539
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                440,
                                504,
                                539
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        440,
                                        504,
                                        539
                                    ],
                                    "type": "text",
                                    "content": "Our Contributions. To address these challenges, we introduce RAG-Anything, a unified framework that fundamentally reimagines multimodal knowledge representation and retrieval. Our approach employs a dual-graph construction strategy that elegantly bridges the gap between cross-modal understanding and fine-grained textual semantics. Rather than forcing diverse modalities into text-centric pipelines, RAG-Anything constructs complementary knowledge graphs that preserve both multimodal contextual relationships and detailed textual knowledge. This design enables seamless integration of visual elements, structured data, and mathematical expressions within a unified retrieval framework. The system maintains semantic integrity across modalities while ensuring efficient cross-modal reasoning capabilities throughout the process."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        107,
                        544,
                        504,
                        654
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                544,
                                504,
                                654
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        544,
                                        504,
                                        654
                                    ],
                                    "type": "text",
                                    "content": "Our cross-modal hybrid retrieval mechanism strategically combines structural knowledge navigation with semantic similarity matching. This architecture addresses the fundamental limitation of existing approaches that rely solely on embedding-based retrieval or keyword matching. RAG-Anything leverages explicit graph relationships to capture multi-hop reasoning patterns. It simultaneously employs dense vector representations to identify semantically relevant content that lacks direct structural connections. The framework introduces modality-aware query processing and cross-modal alignment systems. These enable textual queries to effectively access visual and structured information. This unified approach eliminates the architectural fragmentation that plagues current multimodal RAG systems. It delivers superior performance particularly on long-context documents where relevant evidence spans multiple modalities and document sections."
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        107,
                        659,
                        504,
                        726
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                659,
                                504,
                                726
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        659,
                                        504,
                                        726
                                    ],
                                    "type": "text",
                                    "content": "Experimental Validation. To validate the effectiveness of our proposed approach, we conduct comprehensive experiments on two challenging multimodal benchmarks: DocBench and MMLongBench. Our evaluation demonstrates that RAG-Anything achieves superior performance across diverse domains. The framework represents substantial improvements over state-of-the-art baselines. Notably, our performance gains become increasingly significant as content length increases. We observe particularly pronounced advantages on long-context materials. This validates our core hypothesis"
                                }
                            ]
                        }
                    ],
                    "index": 6
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        308,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                308,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        308,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "2"
                                }
                            ]
                        }
                    ],
                    "index": 7
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 1
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        104,
                        82,
                        504,
                        149
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                82,
                                504,
                                149
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        82,
                                        504,
                                        149
                                    ],
                                    "type": "text",
                                    "content": "that dual-graph construction and cross-modal hybrid retrieval are essential for handling complex multimodal materials. Our ablation studies reveal that graph-based knowledge representation provides the primary performance gains. Traditional chunk-based approaches fail to capture the structural relationships critical for multimodal reasoning. Case studies further demonstrate that our framework excels at precise localization within complex layouts. The system effectively disambiguates similar terminology and navigates multi-panel visualizations through structure-aware retrieval mechanisms."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        105,
                        165,
                        311,
                        178
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                165,
                                311,
                                178
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        165,
                                        311,
                                        178
                                    ],
                                    "type": "text",
                                    "content": "2 THE RAG-ANYTHING FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        105,
                        190,
                        192,
                        201
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                190,
                                192,
                                201
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        190,
                                        192,
                                        201
                                    ],
                                    "type": "text",
                                    "content": "2.1 PRELIMINARY"
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        104,
                        211,
                        506,
                        278
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                211,
                                506,
                                278
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        211,
                                        506,
                                        278
                                    ],
                                    "type": "text",
                                    "content": "Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for dynamically expanding the knowledge boundaries of LLMs. While LLMs demonstrate exceptional reasoning capabilities, their knowledge remains static and bounded by training data cutoffs. This creates an ever-widening gap with the rapidly evolving information landscape. RAG systems address this critical limitation by enabling LLMs to retrieve and incorporate external knowledge sources during inference. This transforms them from static repositories into adaptive, knowledge-aware systems."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        104,
                        282,
                        506,
                        372
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                282,
                                506,
                                372
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        282,
                                        506,
                                        372
                                    ],
                                    "type": "text",
                                    "content": "The Multimodal Reality: Beyond Text-Only RAG. Current RAG systems face a critical limitation that severely restricts their real-world deployment. Existing frameworks operate under the restrictive assumption that knowledge corpus consists exclusively of plain textual documents. This assumption fundamentally misaligns with how information exists in authentic environments. Real-world knowledge repositories are inherently heterogeneous and multimodal, containing rich combinations of textual content, visual elements, structured data, and mathematical expressions. These diverse knowledge sources span multiple document formats and presentation mediums, from research papers and technical slides to web pages and interactive documents."
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        105,
                        383,
                        272,
                        394
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                383,
                                272,
                                394
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        383,
                                        272,
                                        394
                                    ],
                                    "type": "text",
                                    "content": "2.1.1 MOTIVATING RAG-ANYTHING"
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        104,
                        403,
                        504,
                        469
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                403,
                                504,
                                469
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        403,
                                        504,
                                        469
                                    ],
                                    "type": "text",
                                    "content": "This multimodal reality introduces fundamental technical challenges that expose the inadequacy of current text-only RAG approaches. Effective multimodal RAG requires unified indexing strategies that can handle disparate data types, cross-modal retrieval mechanisms that preserve semantic relationships across modalities, and sophisticated synthesis techniques that can coherently integrate diverse information sources. These challenges demand a fundamentally different architectural approach rather than incremental improvements to existing systems."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        474,
                        506,
                        541
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                474,
                                506,
                                541
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        474,
                                        506,
                                        541
                                    ],
                                    "type": "text",
                                    "content": "The RAG-Anything framework introduces a unified approach for retrieving and processing knowledge from heterogeneous multimodal information sources. Our system addresses the fundamental challenge of handling diverse data modalities and document formats within a retrieval pipeline. The framework comprises three core components: universal indexing for multimodal knowledge, cross-modal adaptive retrieval, and knowledge-enhanced response generation. This integrated design enables effective knowledge utilization across modalities while maintaining computational efficiency."
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        105,
                        555,
                        419,
                        566
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                555,
                                419,
                                566
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        555,
                                        419,
                                        566
                                    ],
                                    "type": "text",
                                    "content": "2.2 UNIVERSAL REPRESENTATION FOR HETEROGENEOUS KNOWLEDGE"
                                }
                            ]
                        }
                    ],
                    "index": 9
                },
                {
                    "bbox": [
                        104,
                        575,
                        504,
                        653
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                575,
                                504,
                                653
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        575,
                                        504,
                                        653
                                    ],
                                    "type": "text",
                                    "content": "A key requirement for universal knowledge access is the ability to represent heterogeneous multimodal content in a unified, retrieval-oriented abstraction. Unlike existing pipelines that simply parse documents into text segments, RAG-Anything introduces Multimodal Knowledge Unification. This process decomposes raw inputs into atomic knowledge units while preserving their structural context and semantic alignment. For instance, RAG-Anything ensures that figures remain grounded in their captions, equations remain linked to surrounding definitions, and tables stay connected to explanatory narratives. This transforms heterogeneous files into a coherent substrate for cross-modal retrieval."
                                }
                            ]
                        }
                    ],
                    "index": 10
                },
                {
                    "bbox": [
                        104,
                        658,
                        505,
                        670
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                658,
                                505,
                                670
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        658,
                                        505,
                                        670
                                    ],
                                    "type": "text",
                                    "content": "Formally, each knowledge source "
                                },
                                {
                                    "bbox": [
                                        104,
                                        658,
                                        505,
                                        670
                                    ],
                                    "type": "inline_equation",
                                    "content": "k_{i}\\in \\mathcal{K}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        658,
                                        505,
                                        670
                                    ],
                                    "type": "text",
                                    "content": " (e.g., a web page) is decomposed into atomic content units:"
                                }
                            ]
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        231,
                        676,
                        505,
                        694
                    ],
                    "type": "interline_equation",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                231,
                                676,
                                505,
                                694
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        231,
                                        676,
                                        505,
                                        694
                                    ],
                                    "type": "interline_equation",
                                    "content": "k _ {i} \\xrightarrow {\\text {D e c o m p o s e}} \\left\\{c _ {j} = \\left(t _ {j}, x _ {j}\\right) \\right\\} _ {j = 1} ^ {n _ {i}}, \\tag {1}",
                                    "image_path": "63910f60e9e4925ee507262c1c9525b88647705a8ab6a632d1eab702bb6220c0.jpg"
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        104,
                        699,
                        506,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                699,
                                506,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "where each unit "
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "inline_equation",
                                    "content": "c_{j}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": " consists of a modality type "
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "inline_equation",
                                    "content": "t_{j} \\in \\text{text}, \\text{image}, \\text{table}, \\text{equation}, \\ldots"
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": " and its corresponding raw content "
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "inline_equation",
                                    "content": "x_{j}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": ". The content "
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "inline_equation",
                                    "content": "x_{j}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": " represents the extracted information from the original knowledge source, processed in a modality-aware manner to preserve semantic integrity."
                                }
                            ]
                        }
                    ],
                    "index": 13
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        337,
                        36
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                337,
                                36
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        337,
                                        36
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        308,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                308,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        308,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "3"
                                }
                            ]
                        }
                    ],
                    "index": 14
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 2
        },
        {
            "para_blocks": [
                {
                    "type": "image",
                    "bbox": [
                        108,
                        82,
                        504,
                        253
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                108,
                                82,
                                504,
                                253
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        108,
                                        82,
                                        504,
                                        253
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                108,
                                                82,
                                                504,
                                                253
                                            ],
                                            "type": "image",
                                            "image_path": "79d305ec95ae13155ae24774f90bcf3e48505e8195dad639318bda6230d9815a.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                143,
                                258,
                                465,
                                270
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        143,
                                        258,
                                        465,
                                        270
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                143,
                                                258,
                                                465,
                                                270
                                            ],
                                            "type": "text",
                                            "content": "Figure 1: Overview of our proposed universal RAG framework RAG-Anything."
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        104,
                        289,
                        506,
                        378
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                289,
                                506,
                                378
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        289,
                                        506,
                                        378
                                    ],
                                    "type": "text",
                                    "content": "To ensure high-fidelity extraction, RAG-Anything leverages specialized parsers for different content types. Text is segmented into coherent paragraphs or list items. Figures are extracted with associated metadata such as captions and cross-references. Tables are parsed into structured cells with headers and values. Mathematical expressions are converted into symbolic representations. The resulting "
                                },
                                {
                                    "bbox": [
                                        104,
                                        289,
                                        506,
                                        378
                                    ],
                                    "type": "inline_equation",
                                    "content": "x_{j}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        289,
                                        506,
                                        378
                                    ],
                                    "type": "text",
                                    "content": " preserves both content and structural context within the source. This provides a faithful, modality-consistent representation. The decomposition abstracts diverse file formats into atomic units while maintaining their hierarchical order and contextual relationships. This canonicalization enables uniform processing, indexing, and retrieval of multimodal content within our framework."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        105,
                        395,
                        413,
                        406
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                395,
                                413,
                                406
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        395,
                                        413,
                                        406
                                    ],
                                    "type": "text",
                                    "content": "2.2.1 DUAL-GRAPH CONSTRUCTION FOR MULTIMODAL KNOWLEDGE"
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        104,
                        416,
                        506,
                        506
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                416,
                                506,
                                506
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        416,
                                        506,
                                        506
                                    ],
                                    "type": "text",
                                    "content": "While multimodal knowledge unification provides a uniform abstraction across modalities, directly constructing a single unified graph often risks overlooking modality-specific structural signals. The proposed RAG-Anything addresses this challenge through a dual-graph construction strategy. The system first builds a cross-modal knowledge graph that faithfully grounds non-textual modalities within their contextual environment. It then constructs a text-based knowledge graph using established text-centric extraction pipelines. These complementary graphs are merged through entity alignment. This design ensures accurate cross-modal grounding and comprehensive coverage of textual semantics, enabling richer knowledge representation and robust retrieval."
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        105,
                        519,
                        506,
                        633
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                519,
                                506,
                                633
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "text",
                                    "content": "- Cross-Modal Knowledge Graph: Non-textual content like images, tables, and equations contains rich semantic information that traditional text-only approaches often overlook. To preserve this knowledge, RAG-Anything constructs a multimodal knowledge graph where non-text atomic units are transformed into structured graph entities. RAG-Anything leverages multimodal large language models to derive two complementary textual representations from each atomic content unit. The first is a detailed description "
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "inline_equation",
                                    "content": "d_j^{\\mathrm{chunk}}"
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "text",
                                    "content": " optimized for cross-modal retrieval. The second is an entity summary "
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "inline_equation",
                                    "content": "e_j^{\\mathrm{entity}}"
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "text",
                                    "content": " containing key attributes such as entity name, type, and description for graph construction. The generation process is context-aware, processing each unit with its local neighborhood "
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "inline_equation",
                                    "content": "C_j = \\{c_k\\mid |k - j|\\leq \\delta \\}"
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "text",
                                    "content": ", where "
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\delta"
                                },
                                {
                                    "bbox": [
                                        105,
                                        519,
                                        506,
                                        633
                                    ],
                                    "type": "text",
                                    "content": " controls the contextual window size. This ensures representations accurately reflect each unit's role within the broader document structure."
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        112,
                        636,
                        506,
                        673
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                112,
                                636,
                                506,
                                673
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "text",
                                    "content": "Building on these textual representations, RAG-Anything constructs the graph structure using nontext units as anchor points. For each non-text unit "
                                },
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "inline_equation",
                                    "content": "c_{j}"
                                },
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "text",
                                    "content": ", the graph extraction routine "
                                },
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "inline_equation",
                                    "content": "R(\\cdot)"
                                },
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "text",
                                    "content": " processes its description "
                                },
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "inline_equation",
                                    "content": "d_{j}^{\\mathrm{chunk}}"
                                },
                                {
                                    "bbox": [
                                        112,
                                        636,
                                        506,
                                        673
                                    ],
                                    "type": "text",
                                    "content": " to identify fine-grained entities and relations:"
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        263,
                        684,
                        505,
                        700
                    ],
                    "type": "interline_equation",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                263,
                                684,
                                505,
                                700
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        263,
                                        684,
                                        505,
                                        700
                                    ],
                                    "type": "interline_equation",
                                    "content": "\\left(\\mathcal {V} _ {j}, \\mathcal {E} _ {j}\\right) = R \\left(d _ {j} ^ {\\text {c h u n k}}\\right), \\tag {2}",
                                    "image_path": "b9851632beb8c29bf6cb2626978536655a44d57e4428e974ebb6c457e8800a8c.jpg"
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        113,
                        709,
                        505,
                        735
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                113,
                                709,
                                505,
                                735
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "text",
                                    "content": "where "
                                },
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{V}_j"
                                },
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "text",
                                    "content": " and "
                                },
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{E}_j"
                                },
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "text",
                                    "content": " denote the sets of intra-chunk entities and their relations, respectively. Each atomic non-text unit is associated with a multimodal entity node "
                                },
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "inline_equation",
                                    "content": "v_j^{\\mathrm{mm}}"
                                },
                                {
                                    "bbox": [
                                        113,
                                        709,
                                        505,
                                        735
                                    ],
                                    "type": "text",
                                    "content": " that serves as an anchor for"
                                }
                            ]
                        }
                    ],
                    "index": 9
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        338,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                338,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        338,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        301,
                        751,
                        309,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                301,
                                751,
                                309,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        301,
                                        751,
                                        309,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "4"
                                }
                            ]
                        }
                    ],
                    "index": 10
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 3
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        113,
                        82,
                        362,
                        95
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                113,
                                82,
                                362,
                                95
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        113,
                                        82,
                                        362,
                                        95
                                    ],
                                    "type": "text",
                                    "content": "its intra-chunk entities through explicit belongs_to edges:"
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        203,
                        100,
                        505,
                        126
                    ],
                    "type": "interline_equation",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                203,
                                100,
                                505,
                                126
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        203,
                                        100,
                                        505,
                                        126
                                    ],
                                    "type": "interline_equation",
                                    "content": "\\tilde {V} = \\left\\{v _ {j} ^ {\\mathrm {m m}} \\right\\} _ {j} \\cup \\bigcup_ {j} \\mathcal {V} _ {j}, \\tag {3}",
                                    "image_path": "0951e5c70e49c0de2fe6768117a0450374dc7761229edd67ce2149547b8f674b.jpg"
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        205,
                        129,
                        505,
                        156
                    ],
                    "type": "interline_equation",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                205,
                                129,
                                505,
                                156
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        205,
                                        129,
                                        505,
                                        156
                                    ],
                                    "type": "interline_equation",
                                    "content": "\\tilde {E} = \\bigcup_ {j} \\mathcal {E} _ {j} \\cup \\bigcup_ {j} \\left\\{\\left(u \\xrightarrow {\\text {b e l o n g s} _ {\\text {t o}}} v _ {j} ^ {\\mathrm {m m}}\\right): u \\in \\mathcal {V} _ {j} \\right\\}. \\tag {4}",
                                    "image_path": "cf6b5d5bc4c73dd08ec85b8f651e3110fe4a145d16937504650e264ea00e93b8.jpg"
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        112,
                        162,
                        506,
                        185
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                112,
                                162,
                                506,
                                185
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        112,
                                        162,
                                        506,
                                        185
                                    ],
                                    "type": "text",
                                    "content": "This construction preserves modality-specific grounding while ensuring non-textual content is contextualized by its textual neighborhood. This enables reliable cross-modal retrieval and reasoning."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        105,
                        188,
                        506,
                        277
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                188,
                                506,
                                277
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        188,
                                        506,
                                        277
                                    ],
                                    "type": "text",
                                    "content": "- Text-based Knowledge Graph: For text modality chunks, we construct a traditional text-based knowledge graph following established methodologies similar to LightRAG (Guo et al., 2024) and GraphRAG (Edge et al., 2024). The extraction process operates directly on textual content "
                                },
                                {
                                    "bbox": [
                                        105,
                                        188,
                                        506,
                                        277
                                    ],
                                    "type": "inline_equation",
                                    "content": "x_{j}"
                                },
                                {
                                    "bbox": [
                                        105,
                                        188,
                                        506,
                                        277
                                    ],
                                    "type": "text",
                                    "content": " where "
                                },
                                {
                                    "bbox": [
                                        105,
                                        188,
                                        506,
                                        277
                                    ],
                                    "type": "inline_equation",
                                    "content": "t_{j} ="
                                },
                                {
                                    "bbox": [
                                        105,
                                        188,
                                        506,
                                        277
                                    ],
                                    "type": "text",
                                    "content": " text, leveraging named entity recognition and relation extraction techniques to identify entities and their semantic relationships. Given the rich semantic information inherent in textual content, multimodal context integration is not required for this component. The resulting text-based knowledge graph captures explicit knowledge and semantic connections present in textual portions of documents, complementing the multimodal graph's cross-modal grounding capabilities."
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        105,
                        289,
                        306,
                        300
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                289,
                                306,
                                300
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        289,
                                        306,
                                        300
                                    ],
                                    "type": "text",
                                    "content": "2.2.2 GRAPH FUSION AND INDEX CREATION"
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        104,
                        308,
                        506,
                        343
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                308,
                                506,
                                343
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        308,
                                        506,
                                        343
                                    ],
                                    "type": "text",
                                    "content": "The separate cross-modal and text-based knowledge graphs capture complementary aspects of document semantics. Integrating them creates a unified representation leveraging visual-textual associations and fine-grained textual relationships for enhanced retrieval."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        347,
                        505,
                        498
                    ],
                    "type": "list",
                    "angle": 0,
                    "index": 10,
                    "blocks": [
                        {
                            "bbox": [
                                104,
                                347,
                                505,
                                438
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        347,
                                        505,
                                        438
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                347,
                                                505,
                                                438
                                            ],
                                            "type": "text",
                                            "content": "- (i) Entity Alignment and Graph Fusion. To create a unified knowledge representation, we merge the multimodal knowledge graph "
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                347,
                                                505,
                                                438
                                            ],
                                            "type": "inline_equation",
                                            "content": "(\\tilde{V}, \\tilde{E})"
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                347,
                                                505,
                                                438
                                            ],
                                            "type": "text",
                                            "content": " and text-based knowledge graph through entity alignment. This process uses entity names as primary matching keys to identify semantically equivalent entities across both graph structures. The integration consolidates their representations, creating a comprehensive knowledge graph "
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                347,
                                                505,
                                                438
                                            ],
                                            "type": "inline_equation",
                                            "content": "\\mathcal{G} = (\\mathcal{V}, \\mathcal{E})"
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                347,
                                                505,
                                                438
                                            ],
                                            "type": "text",
                                            "content": ". This graph captures both multimodal contextual relationships and text-based semantic connections. The merged graph provides a holistic view of the document collection. This enables effective retrieval by leveraging visual-textual associations from the multimodal graph and fine-grained textual knowledge relationships from the text-based graph."
                                        }
                                    ]
                                }
                            ],
                            "index": 8
                        },
                        {
                            "bbox": [
                                104,
                                443,
                                504,
                                498
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        443,
                                        504,
                                        498
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                443,
                                                504,
                                                498
                                            ],
                                            "type": "text",
                                            "content": "- (ii) Dense Representation Generation. To enable efficient similarity-based retrieval, we construct a comprehensive embedding table "
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                443,
                                                504,
                                                498
                                            ],
                                            "type": "inline_equation",
                                            "content": "\\mathcal{T}"
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                443,
                                                504,
                                                498
                                            ],
                                            "type": "text",
                                            "content": " that encompasses all components generated during the indexing process. We encode dense representations for all graph entities, relationships, and atomic content chunks across modalities using an appropriate encoder. This creates a unified embedding space where each component "
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                443,
                                                504,
                                                498
                                            ],
                                            "type": "inline_equation",
                                            "content": "s \\in"
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                443,
                                                504,
                                                498
                                            ],
                                            "type": "text",
                                            "content": " entities, relations, chunks is mapped to its corresponding dense representation:"
                                        }
                                    ]
                                }
                            ],
                            "index": 9
                        }
                    ],
                    "sub_type": "text"
                },
                {
                    "bbox": [
                        239,
                        504,
                        504,
                        519
                    ],
                    "type": "interline_equation",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                239,
                                504,
                                504,
                                519
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        239,
                                        504,
                                        504,
                                        519
                                    ],
                                    "type": "interline_equation",
                                    "content": "\\mathcal {T} = \\operatorname {e m b} (s): s \\in \\mathcal {V} \\cup \\mathcal {E} \\cup c _ {j j}, \\tag {5}",
                                    "image_path": "6af18adbe14bb75f8ff68ed4ab3a43a854fd02292b2e272c621eeed1fd760195.jpg"
                                }
                            ]
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        104,
                        525,
                        504,
                        571
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                525,
                                504,
                                571
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "text",
                                    "content": "where "
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathrm{emb}(\\cdot)"
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "text",
                                    "content": " denotes the embedding function tailored for each component type. Together, the unified knowledge graph "
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{G}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "text",
                                    "content": " and the embedding table "
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{T}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "text",
                                    "content": " constitute the complete retrieval index "
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{I} = (\\mathcal{G},\\mathcal{T})"
                                },
                                {
                                    "bbox": [
                                        104,
                                        525,
                                        504,
                                        571
                                    ],
                                    "type": "text",
                                    "content": ". This provides both structural knowledge representation and dense vector space for efficient cross-modal similarity search during the subsequent retrieval stage."
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        105,
                        584,
                        286,
                        594
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                584,
                                286,
                                594
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        584,
                                        286,
                                        594
                                    ],
                                    "type": "text",
                                    "content": "2.3 CROSS-MODAL HYBRID RETRIEVAL"
                                }
                            ]
                        }
                    ],
                    "index": 13
                },
                {
                    "bbox": [
                        104,
                        604,
                        504,
                        672
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                604,
                                504,
                                672
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        604,
                                        504,
                                        672
                                    ],
                                    "type": "text",
                                    "content": "The retrieval stage operates on the index "
                                },
                                {
                                    "bbox": [
                                        104,
                                        604,
                                        504,
                                        672
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{I} = (\\mathcal{G},\\mathcal{T})"
                                },
                                {
                                    "bbox": [
                                        104,
                                        604,
                                        504,
                                        672
                                    ],
                                    "type": "text",
                                    "content": " to identify relevant knowledge components for a given user query. Traditional RAG methods face significant limitations when dealing with multimodal documents. They typically rely on semantic similarity within single modalities and fail to capture the rich interconnections between visual, mathematical, tabular, and textual elements. To address these challenges, our framework introduces a cross-modal hybrid retrieval mechanism. This mechanism leverages structural knowledge and semantic representations across heterogeneous modalities."
                                }
                            ]
                        }
                    ],
                    "index": 14
                },
                {
                    "bbox": [
                        104,
                        677,
                        506,
                        734
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                677,
                                506,
                                734
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        506,
                                        734
                                    ],
                                    "type": "text",
                                    "content": "Modality-Aware Query Encoding. Given a user query "
                                },
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        506,
                                        734
                                    ],
                                    "type": "inline_equation",
                                    "content": "q"
                                },
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        506,
                                        734
                                    ],
                                    "type": "text",
                                    "content": ", we first perform modality-aware query analysis to extract lexical cues and potential modality preferences embedded within the query. For instance, queries containing terms such as \"figure,\" \"chart,\" \"table,\" or \"equation\" provide explicit signals about the expected modality of relevant information. We then compute a unified text embedding "
                                },
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        506,
                                        734
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathbf{e}_q"
                                },
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        506,
                                        734
                                    ],
                                    "type": "text",
                                    "content": " using the same encoder employed during indexing, ensuring consistency between"
                                }
                            ]
                        }
                    ],
                    "index": 15
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        308,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                308,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        308,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "5"
                                }
                            ]
                        }
                    ],
                    "index": 16
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 4
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        104,
                        82,
                        504,
                        117
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                82,
                                504,
                                117
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        82,
                                        504,
                                        117
                                    ],
                                    "type": "text",
                                    "content": "query and knowledge representations. This embedding-based approach enables cross-modal retrieval capabilities where textual queries can effectively access multimodal content through their shared representations, maintaining retrieval consistency while preserving cross-modal accessibility."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        104,
                        121,
                        504,
                        156
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                121,
                                504,
                                156
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        121,
                                        504,
                                        156
                                    ],
                                    "type": "text",
                                    "content": "Hybrid Knowledge Retrieval Architecture. Recognizing that knowledge relevance manifests through both explicit structural connections and implicit semantic relationships, we design a hybrid retrieval architecture that strategically combines two complementary mechanisms."
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        104,
                        159,
                        504,
                        227
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                159,
                                504,
                                227
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        159,
                                        504,
                                        227
                                    ],
                                    "type": "text",
                                    "content": "- (i) Structural Knowledge Navigation. This mechanism addresses the challenge of capturing explicit relationships and multi-hop reasoning patterns. Traditional keyword-based retrieval often fails to identify knowledge connected through intermediate entities or cross-modal relationships. To overcome this limitation, we exploit the structural properties encoded within our unified knowledge graph G. We employ keyword matching and entity recognition to locate relevant graph components. The retrieval process begins with exact entity matching against query terms."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        104,
                        232,
                        505,
                        299
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                232,
                                505,
                                299
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        232,
                                        505,
                                        299
                                    ],
                                    "type": "text",
                                    "content": "We then perform strategic neighborhood expansion to include related entities and relationships within a specified hop distance. This structural approach proves particularly effective at uncovering high-level semantic connections and entity-relation patterns that span multiple modalities. It capitalizes on the rich cross-modal linkages established in our multimodal knowledge graph. The structural navigation yields candidate set "
                                },
                                {
                                    "bbox": [
                                        104,
                                        232,
                                        505,
                                        299
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{C}_{\\mathrm{stru}}(q)"
                                },
                                {
                                    "bbox": [
                                        104,
                                        232,
                                        505,
                                        299
                                    ],
                                    "type": "text",
                                    "content": " containing relevant entities, relationships, and their associated content chunks that provide comprehensive contextual information."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        104,
                        303,
                        504,
                        360
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                303,
                                504,
                                360
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        303,
                                        504,
                                        360
                                    ],
                                    "type": "text",
                                    "content": "- (ii) Semantic Similarity Matching. This mechanism addresses the challenge of identifying semantically relevant knowledge that lacks explicit structural connections. While structural navigation excels at following explicit relationships, it may miss relevant content that is semantically related but not directly connected in the graph topology. To bridge this gap, we conduct dense vector similarity search between the query embedding "
                                },
                                {
                                    "bbox": [
                                        104,
                                        303,
                                        504,
                                        360
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathbf{e}_q"
                                },
                                {
                                    "bbox": [
                                        104,
                                        303,
                                        504,
                                        360
                                    ],
                                    "type": "text",
                                    "content": " and all components stored in embedding table "
                                },
                                {
                                    "bbox": [
                                        104,
                                        303,
                                        504,
                                        360
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{T}"
                                },
                                {
                                    "bbox": [
                                        104,
                                        303,
                                        504,
                                        360
                                    ],
                                    "type": "text",
                                    "content": "."
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        104,
                        364,
                        505,
                        441
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                364,
                                505,
                                441
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        364,
                                        505,
                                        441
                                    ],
                                    "type": "text",
                                    "content": "This approach encompasses atomic content chunks across all modalities, graph entities, and relationship representations, enabling fine-grained semantic matching that can surface relevant knowledge even when traditional lexical or structural signals are absent. The learned embedding space captures nuanced semantic relationships and contextual similarities that complement the explicit structural signals from the navigation mechanism. This retrieval pathway returns the top-k most semantically similar chunks "
                                },
                                {
                                    "bbox": [
                                        104,
                                        364,
                                        505,
                                        441
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{C}_{\\mathrm{seman}}(q)"
                                },
                                {
                                    "bbox": [
                                        104,
                                        364,
                                        505,
                                        441
                                    ],
                                    "type": "text",
                                    "content": " ranked by cosine similarity scores, ensuring comprehensive coverage of both structurally and semantically relevant knowledge."
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        104,
                        447,
                        504,
                        502
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                447,
                                504,
                                502
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        447,
                                        504,
                                        502
                                    ],
                                    "type": "text",
                                    "content": "Candidate Pool Unification. Both retrieval pathways may return overlapping candidates with differing relevance signals. This necessitates a principled approach to unify and rank results. Retrieval candidates from both pathways are unified into a comprehensive candidate pool: "
                                },
                                {
                                    "bbox": [
                                        104,
                                        447,
                                        504,
                                        502
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{C}(q) = \\mathcal{C}_{\\mathrm{stru}}(q)\\cup"
                                },
                                {
                                    "bbox": [
                                        104,
                                        447,
                                        504,
                                        502
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{C}_{\\mathrm{seman}}(q)"
                                },
                                {
                                    "bbox": [
                                        104,
                                        447,
                                        504,
                                        502
                                    ],
                                    "type": "text",
                                    "content": ". Simply merging candidates would ignore distinct evidence each pathway provides. It would fail to account for redundancy between retrieved content."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        508,
                        504,
                        613
                    ],
                    "type": "list",
                    "angle": 0,
                    "index": 10,
                    "blocks": [
                        {
                            "bbox": [
                                104,
                                508,
                                504,
                                575
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        508,
                                        504,
                                        575
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                508,
                                                504,
                                                575
                                            ],
                                            "type": "text",
                                            "content": "- (i) Multi-Signal Fusion Scoring. To address these challenges, we apply a sophisticated fusion scoring mechanism integrating multiple complementary relevance signals. These include structural importance derived from graph topology, semantic similarity scores from embedding space, and query-inferred modality preferences obtained through lexical analysis. This multi-faceted scoring approach ensures that final ranked candidates "
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                508,
                                                504,
                                                575
                                            ],
                                            "type": "inline_equation",
                                            "content": "\\mathcal{C}^{\\star}(q)"
                                        },
                                        {
                                            "bbox": [
                                                104,
                                                508,
                                                504,
                                                575
                                            ],
                                            "type": "text",
                                            "content": " effectively balance structural knowledge relationships with semantic relevance while appropriately weighting different modalities based on query characteristics."
                                        }
                                    ]
                                }
                            ],
                            "index": 8
                        },
                        {
                            "bbox": [
                                104,
                                579,
                                504,
                                613
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        579,
                                        504,
                                        613
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                579,
                                                504,
                                                613
                                            ],
                                            "type": "text",
                                            "content": "- (ii) Hybrid Retrieval Integration. The resulting hybrid retrieval mechanism enables our framework to leverage the complementary strengths of both knowledge graphs and dense representations. This provides comprehensive coverage of relevant multimodal knowledge for response generation."
                                        }
                                    ]
                                }
                            ],
                            "index": 9
                        }
                    ],
                    "sub_type": "text"
                },
                {
                    "bbox": [
                        105,
                        628,
                        273,
                        639
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                628,
                                273,
                                639
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        628,
                                        273,
                                        639
                                    ],
                                    "type": "text",
                                    "content": "2.4 FROM RETRIEVAL TO SYNTHESIS"
                                }
                            ]
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        104,
                        648,
                        504,
                        704
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                648,
                                504,
                                704
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        648,
                                        504,
                                        704
                                    ],
                                    "type": "text",
                                    "content": "Effective multimodal question answering requires preserving rich visual semantics while maintaining coherent grounding across heterogeneous knowledge sources. Simple text-only approaches lose crucial visual information, while naive multimodal methods struggle with coherent cross-modal integration. Our synthesis stage addresses these challenges by systematically combining retrieved multimodal knowledge into comprehensive, evidence-grounded responses."
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        104,
                        709,
                        505,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                709,
                                505,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        709,
                                        505,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "- (i) Building Textual Context. Given the top-ranked retrieval candidates "
                                },
                                {
                                    "bbox": [
                                        104,
                                        709,
                                        505,
                                        733
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{C}^{\\star}(q)"
                                },
                                {
                                    "bbox": [
                                        104,
                                        709,
                                        505,
                                        733
                                    ],
                                    "type": "text",
                                    "content": ", we construct a structured textual context. We concatenate textual representations of all retrieved components, includ"
                                }
                            ]
                        }
                    ],
                    "index": 13
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        309,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                309,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        309,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "6"
                                }
                            ]
                        }
                    ],
                    "index": 14
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 5
        },
        {
            "para_blocks": [
                {
                    "type": "table",
                    "bbox": [
                        106,
                        91,
                        504,
                        137
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                91,
                                504,
                                137
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        91,
                                        504,
                                        137
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                91,
                                                504,
                                                137
                                            ],
                                            "type": "table",
                                            "html": "<table><tr><td>Dataset</td><td># Documents</td><td># Avg. Pages</td><td># Avg. Tokens</td><td># Doc Types</td><td># Questions</td></tr><tr><td>DocBench</td><td>229</td><td>66</td><td>46377</td><td>5</td><td>1102</td></tr><tr><td>MMLongBench</td><td>135</td><td>47.5</td><td>21214</td><td>7</td><td>1082</td></tr></table>",
                                            "image_path": "14861407bef7f68b31affea722351d2101dcf3994e2fdd92f50524cbefb3411b.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "table_body"
                        },
                        {
                            "bbox": [
                                214,
                                80,
                                394,
                                91
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        214,
                                        80,
                                        394,
                                        91
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                214,
                                                80,
                                                394,
                                                91
                                            ],
                                            "type": "text",
                                            "content": "Table 1: Statistics of Experimental Datasets."
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "table_caption"
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        104,
                        148,
                        504,
                        183
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                148,
                                504,
                                183
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        148,
                                        504,
                                        183
                                    ],
                                    "type": "text",
                                    "content": "ing entity summaries, relationship descriptions, and chunk contents. The concatenation incorporates appropriate delimiters to indicate modality types and hierarchical origins. This approach ensures the language model can effectively parse and reason over heterogeneous knowledge components."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        104,
                        187,
                        506,
                        232
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                187,
                                506,
                                232
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        187,
                                        506,
                                        232
                                    ],
                                    "type": "text",
                                    "content": "- (ii) Recovering Visual Content. For multimodal chunks corresponding to visual artifacts, we perform dereferencing to recover original visual content, creating "
                                },
                                {
                                    "bbox": [
                                        104,
                                        187,
                                        506,
                                        232
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\mathcal{V}^{\\star}(q)"
                                },
                                {
                                    "bbox": [
                                        104,
                                        187,
                                        506,
                                        232
                                    ],
                                    "type": "text",
                                    "content": ". This design maintains consistency with our unified embedding strategy. Textual proxies enable efficient retrieval while authentic visual content provides rich semantics necessary for sophisticated reasoning during synthesis."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        104,
                        237,
                        504,
                        260
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                237,
                                504,
                                260
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        237,
                                        504,
                                        260
                                    ],
                                    "type": "text",
                                    "content": "The synthesis process jointly conditions on both the assembled comprehensive textual context and dereferenced visual artifacts using a vision-language model:"
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        231,
                        266,
                        504,
                        279
                    ],
                    "type": "interline_equation",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                231,
                                266,
                                504,
                                279
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        231,
                                        266,
                                        504,
                                        279
                                    ],
                                    "type": "interline_equation",
                                    "content": "\\operatorname {R e s p o n s e} = \\operatorname {V L M} (q, \\mathcal {P} (q), \\mathcal {V} ^ {\\star} (q)), \\tag {6}",
                                    "image_path": "eef8a4b4967a5a765dd610dfafd2ec168406b75d42c7ae466f41f0a37c5be0ae.jpg"
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        104,
                        285,
                        504,
                        319
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                285,
                                504,
                                319
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        285,
                                        504,
                                        319
                                    ],
                                    "type": "text",
                                    "content": "where the VLM integrates information from query, textual context, and visual content. This unified conditioning enables sophisticated visual interpretation while maintaining grounding in retrieved evidence. The resulting responses are both visually informed and factually grounded."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        105,
                        335,
                        194,
                        346
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                335,
                                194,
                                346
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        335,
                                        194,
                                        346
                                    ],
                                    "type": "text",
                                    "content": "3 EVALUATION"
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        105,
                        360,
                        246,
                        371
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                360,
                                246,
                                371
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        360,
                                        246,
                                        371
                                    ],
                                    "type": "text",
                                    "content": "3.1 EXPERIMENTAL SETTINGS"
                                }
                            ]
                        }
                    ],
                    "index": 9
                },
                {
                    "bbox": [
                        104,
                        380,
                        506,
                        449
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                380,
                                506,
                                449
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        380,
                                        506,
                                        449
                                    ],
                                    "type": "text",
                                    "content": "Evaluation Datasets. We conduct comprehensive evaluations on two challenging multimodal Document Question Answering (DQA) benchmarks that reflect real-world complexity and diversity. DocBench (Zou et al., 2024) provides a rigorous testbed with 229 multimodal documents spanning five critical domains: Academia, Finance, Government, Laws, and News. The dataset includes 1,102 expert-crafted question-answer pairs. These documents are notably extensive, averaging 66 pages and approximately 46,377 tokens, which presents substantial challenges for long-context understanding."
                                }
                            ]
                        }
                    ],
                    "index": 10
                },
                {
                    "bbox": [
                        104,
                        453,
                        506,
                        520
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                453,
                                506,
                                520
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        453,
                                        506,
                                        520
                                    ],
                                    "type": "text",
                                    "content": "MMLongBench (Ma et al., 2024) complements this evaluation by focusing specifically on long-context multimodal document comprehension. It features 135 documents across 7 diverse document types with 1,082 expert-annotated questions. Together, these benchmarks provide comprehensive coverage of the multimodal document understanding challenges that RAG-Anything aims to address. They ensure our evaluation captures both breadth across domains and depth in document complexity. Detailed dataset statistics and characteristics are provided in Appendix A.1."
                                }
                            ]
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        104,
                        524,
                        502,
                        536
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                524,
                                502,
                                536
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        524,
                                        502,
                                        536
                                    ],
                                    "type": "text",
                                    "content": "Baselines. We compare RAG-Anything against the following methods for performance evaluation:"
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        104,
                        546,
                        504,
                        656
                    ],
                    "type": "list",
                    "angle": 0,
                    "index": 16,
                    "blocks": [
                        {
                            "bbox": [
                                104,
                                546,
                                504,
                                582
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        546,
                                        504,
                                        582
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                546,
                                                504,
                                                582
                                            ],
                                            "type": "text",
                                            "content": "- GPT-4o-mini: A powerful multimodal language model with native text and image understanding capabilities. Its 128K token context window enables direct processing of entire documents. We evaluate this model as a strong baseline for long-context multimodal understanding."
                                        }
                                    ]
                                }
                            ],
                            "index": 13
                        },
                        {
                            "bbox": [
                                104,
                                584,
                                504,
                                618
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        584,
                                        504,
                                        618
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                584,
                                                504,
                                                618
                                            ],
                                            "type": "text",
                                            "content": "- LightRAG (Guo et al., 2024): A graph-enhanced RAG system that integrates structured knowledge representation with dual-level retrieval mechanisms. It captures both fine-grained entity-relation information and broader semantic context, improving retrieval precision and response coherence."
                                        }
                                    ]
                                }
                            ],
                            "index": 14
                        },
                        {
                            "bbox": [
                                104,
                                621,
                                504,
                                656
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        621,
                                        504,
                                        656
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                621,
                                                504,
                                                656
                                            ],
                                            "type": "text",
                                            "content": "- MMGraphRAG (Wan & Yu, 2025): A multimodal retrieval framework that constructs unified knowledge graphs spanning textual and visual content. This method employs spectral clustering for multimodal entity analysis and retrieves context along reasoning paths to guide generation."
                                        }
                                    ]
                                }
                            ],
                            "index": 15
                        }
                    ],
                    "sub_type": "text"
                },
                {
                    "bbox": [
                        104,
                        665,
                        506,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                665,
                                506,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        665,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "Experimental Settings. In our experiments, we implement all baselines using GPT-4o-mini as the backbone LLM. Documents are parsed using MinerU (Wang et al., 2024) to extract text, images, tables, and equations for downstream RAG processing. For the retrieval pipeline, we employ the text-embedding-3-large model with 3072-dimensional embeddings. We use the bge-reranker-v2-m3 model for reranking. For graph-based RAG methods, we enforce a combined entity-and-relation token limit of 20,000 tokens and a chunk token limit of 12,000 tokens."
                                }
                            ]
                        }
                    ],
                    "index": 17
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        308,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                308,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        308,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "7"
                                }
                            ]
                        }
                    ],
                    "index": 18
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 6
        },
        {
            "para_blocks": [
                {
                    "type": "table",
                    "bbox": [
                        108,
                        130,
                        504,
                        220
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                108,
                                130,
                                504,
                                220
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        108,
                                        130,
                                        504,
                                        220
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                108,
                                                130,
                                                504,
                                                220
                                            ],
                                            "type": "table",
                                            "html": "<table><tr><td rowspan=\"2\">Method</td><td colspan=\"5\">Domains</td><td colspan=\"3\">Types</td><td rowspan=\"2\">Overall</td></tr><tr><td>Aca.</td><td>Fin.</td><td>Gov.</td><td>Law.</td><td>News</td><td>Txt.</td><td>Mm.</td><td>Una.</td></tr><tr><td>GPT-4o-mini</td><td>40.3</td><td>46.9</td><td>60.3</td><td>59.2</td><td>61.0</td><td>61.0</td><td>43.8</td><td>49.6</td><td>51.2</td></tr><tr><td>LightRAG</td><td>53.8</td><td>56.2</td><td>59.5</td><td>61.8</td><td>65.7</td><td>85.0</td><td>59.7</td><td>46.8</td><td>58.4</td></tr><tr><td>MMGraphRAG</td><td>64.3</td><td>52.8</td><td>64.9</td><td>40.0</td><td>61.5</td><td>67.6</td><td>66.0</td><td>60.5</td><td>61.0</td></tr><tr><td>RAGAnything</td><td>61.4</td><td>67.0</td><td>61.5</td><td>60.2</td><td>66.3</td><td>85.0</td><td>76.3</td><td>46.0</td><td>63.4</td></tr></table>",
                                            "image_path": "f137776e8af0d7ce8bd6e318be2769a719ac6f4b1fb06f3adb4a40b32fbf9acc.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "table_body"
                        },
                        {
                            "bbox": [
                                104,
                                79,
                                506,
                                126
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        79,
                                        506,
                                        126
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                79,
                                                506,
                                                126
                                            ],
                                            "type": "text",
                                            "content": "Table 2: Accuracy (%) on DocBench Dataset. Performance results with best scores highlighted in dark blue and second-best in light blue. Domain categories include Academia (Aca.), Finance (Fin.), Government (Gov.), Legal Documents (Law), and News Articles (News). Document types are categorized as Text-only (Txt.), Multimodal (Mm.), and Unanswerable queries (Una.)."
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "table_caption"
                        }
                    ],
                    "index": 2
                },
                {
                    "type": "table",
                    "bbox": [
                        108,
                        281,
                        504,
                        373
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                108,
                                281,
                                504,
                                373
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        108,
                                        281,
                                        504,
                                        373
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                108,
                                                281,
                                                504,
                                                373
                                            ],
                                            "type": "table",
                                            "html": "<table><tr><td rowspan=\"2\">Method</td><td colspan=\"7\">Domains</td><td rowspan=\"2\">Overall</td></tr><tr><td>Res.</td><td>Tut.</td><td>Acad.</td><td>Guid.</td><td>Broch.</td><td>Admin.</td><td>Fin.</td></tr><tr><td>GPT-4o-mini</td><td>35.5</td><td>44.0</td><td>24.6</td><td>33.1</td><td>29.5</td><td>46.8</td><td>31.1</td><td>33.5</td></tr><tr><td>LightRAG</td><td>40.8</td><td>34.1</td><td>36.2</td><td>39.4</td><td>41.0</td><td>44.4</td><td>38.3</td><td>38.9</td></tr><tr><td>MMGraphRAG</td><td>40.8</td><td>36.5</td><td>35.7</td><td>35.8</td><td>28.2</td><td>46.9</td><td>38.5</td><td>37.7</td></tr><tr><td>RAGAnything</td><td>46.6</td><td>43.5</td><td>38.7</td><td>43.9</td><td>34.0</td><td>45.7</td><td>43.6</td><td>42.8</td></tr></table>",
                                            "image_path": "57705f0111dadaea4f4a06fd5b20d4c30089c4619ce0e95c4a9ba9af82a04625.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "table_body"
                        },
                        {
                            "bbox": [
                                104,
                                232,
                                507,
                                277
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        232,
                                        507,
                                        277
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                232,
                                                507,
                                                277
                                            ],
                                            "type": "text",
                                            "content": "Table 3: Accuracy (%) on MMLongBench across different domains and overall performance. Best results are highlighted in dark blue and second-best in light blue.. Domain categories include Research Reports/Introductions (Res.), Tutorials/Workshops (Tut.), Academic Papers (Acad.), Guidebooks (Guid.), Brochures (Broch.), Administration/Industry Files (Admin.), and Financial Reports (Fin.)."
                                        }
                                    ]
                                }
                            ],
                            "index": 3,
                            "angle": 0,
                            "type": "table_caption"
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        104,
                        396,
                        506,
                        431
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                396,
                                506,
                                431
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        396,
                                        506,
                                        431
                                    ],
                                    "type": "text",
                                    "content": "Outputs are constrained to a one-sentence format. For the baseline GPT-4o-mini in our QA scenario, documents are concatenated into image form with a maximum of 50 pages per document, rendered at 144 dpi. Finally, all query results are evaluated for accuracy by GPT-4o-mini."
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        105,
                        445,
                        259,
                        456
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                445,
                                259,
                                456
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        445,
                                        259,
                                        456
                                    ],
                                    "type": "text",
                                    "content": "3.2 PERFORMANCE COMPARISON"
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        104,
                        467,
                        507,
                        545
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                467,
                                507,
                                545
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        467,
                                        507,
                                        545
                                    ],
                                    "type": "text",
                                    "content": "Superior Performance and Cross-Domain Generalization. RAG-Anything demonstrates superior overall performance over baselines through its unified multimodal framework. Unlike LightRAG, which is restricted to text-only content processing, RAG-Anything treats text, images, tables, and equations as first-class entities. MMGraphRAG only adds basic image processing while treating tables and equations as plain text, missing crucial structural information. RAG-Anything introduces a comprehensive dual-graph construction strategy that preserves structural relationships across all modalities. This unified approach enables superior performance across both evaluation benchmarks."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        550,
                        506,
                        672
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                550,
                                506,
                                672
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        550,
                                        506,
                                        672
                                    ],
                                    "type": "text",
                                    "content": "Enhanced Long-Context Performance. RAG-Anything demonstrates superior performance on long-context documents. The framework excels where relevant evidence is dispersed across multiple modalities and sections. It achieves the best results in information-dense domains such as Research Reports and Financial Reports on MMLongBench. These improvements stem from the structured context injection mechanism. This mechanism integrates dual-graph construction for cross-page entity alignment. It combines semantic retrieval with structural navigation. The framework also employs modality-aware processing for efficient context window utilization. Unlike baselines that cannot uniformly process diverse modalities, RAG-Anything effectively captures scattered multimodal evidence. Its cross-modal hybrid retrieval architecture combines structural knowledge navigation with semantic similarity matching. This enables the framework to leverage both explicit relationships and implicit semantic connections across modalities."
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        104,
                        677,
                        507,
                        734
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                677,
                                507,
                                734
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        507,
                                        734
                                    ],
                                    "type": "text",
                                    "content": "To systematically evaluate model performance across varying document lengths, we conducted comprehensive experiments on both datasets. As illustrated in Figure 2, RAG-Anything and MMGraphRAG exhibit comparable performance on shorter documents. However, RAG-Anything's advantages become increasingly pronounced as document length grows. On DocBench, the performance gap expands dramatically to over 13 points for documents exceeding 100 pages (68.2% vs."
                                }
                            ]
                        }
                    ],
                    "index": 9
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        340,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                340,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        340,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        309,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                309,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        309,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "8"
                                }
                            ]
                        }
                    ],
                    "index": 10
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 7
        },
        {
            "para_blocks": [
                {
                    "type": "image",
                    "bbox": [
                        106,
                        79,
                        200,
                        175
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                79,
                                200,
                                175
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        79,
                                        200,
                                        175
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                79,
                                                200,
                                                175
                                            ],
                                            "type": "image",
                                            "image_path": "ee5785bf216b1f1ff615384c72547c777bf46804bbf3fe634e3965a1465bc444.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                160,
                                186,
                                448,
                                198
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        160,
                                        186,
                                        448,
                                        198
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                160,
                                                186,
                                                448,
                                                198
                                            ],
                                            "type": "text",
                                            "content": "Figure 2: Performance evaluation across documents of varying lengths."
                                        }
                                    ]
                                }
                            ],
                            "index": 5,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 1
                },
                {
                    "type": "image",
                    "bbox": [
                        205,
                        79,
                        298,
                        175
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                205,
                                79,
                                298,
                                175
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        205,
                                        79,
                                        298,
                                        175
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                205,
                                                79,
                                                298,
                                                175
                                            ],
                                            "type": "image",
                                            "image_path": "0ff7dfded638c32c52f0c49484c85a6a53ac5bf3523d2e039357c7c3273a021a.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 2
                },
                {
                    "type": "image",
                    "bbox": [
                        304,
                        79,
                        402,
                        175
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                304,
                                79,
                                402,
                                175
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        304,
                                        79,
                                        402,
                                        175
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                304,
                                                79,
                                                402,
                                                175
                                            ],
                                            "type": "image",
                                            "image_path": "8b03d3d346839fe45a7bbe8bf5c15bee1bfd8c4582617bfcd4d5bd9e11de5d2b.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 3,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 3
                },
                {
                    "type": "image",
                    "bbox": [
                        403,
                        80,
                        504,
                        175
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                403,
                                80,
                                504,
                                175
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        403,
                                        80,
                                        504,
                                        175
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                403,
                                                80,
                                                504,
                                                175
                                            ],
                                            "type": "image",
                                            "image_path": "197235d0fe54a2c177dedbccad166db0ae7d067b61ae4ba586672bd0bc4c7ceb.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 4
                },
                {
                    "type": "table",
                    "bbox": [
                        108,
                        242,
                        503,
                        321
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                108,
                                242,
                                503,
                                321
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        108,
                                        242,
                                        503,
                                        321
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                108,
                                                242,
                                                503,
                                                321
                                            ],
                                            "type": "table",
                                            "html": "<table><tr><td rowspan=\"2\">Method</td><td colspan=\"5\">Domains</td><td colspan=\"3\">Types</td><td rowspan=\"2\">Overall</td></tr><tr><td>Aca.</td><td>Fin.</td><td>Gov.</td><td>Law.</td><td>News</td><td>Txt.</td><td>Mm.</td><td>Una.</td></tr><tr><td>Chunk-only</td><td>55.8</td><td>61.5</td><td>60.1</td><td>60.7</td><td>64.0</td><td>81.6</td><td>66.2</td><td>43.5</td><td>60.0</td></tr><tr><td>w/o Reranker</td><td>60.9</td><td>63.5</td><td>58.8</td><td>60.2</td><td>68.6</td><td>81.7</td><td>74.7</td><td>45.4</td><td>62.4</td></tr><tr><td>RAGAnything</td><td>61.4</td><td>67.0</td><td>61.5</td><td>60.2</td><td>66.3</td><td>85.0</td><td>76.3</td><td>46.0</td><td>63.4</td></tr></table>",
                                            "image_path": "12f755ba148b16e967d1e4628d11a09039e7a1dfd51629985b340dbb3b7bc223.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 7,
                            "angle": 0,
                            "type": "table_body"
                        },
                        {
                            "bbox": [
                                104,
                                202,
                                504,
                                237
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        202,
                                        504,
                                        237
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                202,
                                                504,
                                                237
                                            ],
                                            "type": "text",
                                            "content": "Table 4: Ablation study results on DocBench. The \"Chunk-only\" variant bypasses dual-graph construction and relies solely on traditional chunk-based retrieval, while \"w/o Reranker\" eliminates cross-modal reranking but preserves the core graph-based architecture."
                                        }
                                    ]
                                }
                            ],
                            "index": 6,
                            "angle": 0,
                            "type": "table_caption"
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        341,
                        504,
                        397
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                341,
                                504,
                                397
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        341,
                                        504,
                                        397
                                    ],
                                    "type": "text",
                                    "content": "54.6% for 101-200 pages; 68.8% vs. 55.0% for 200+ pages). On MMLongBench, RAG-Anything demonstrates consistent improvements across all length categories, achieving accuracy gains of 3.4 points for 11-50 pages, 9.3 points for 51-100 pages, and 7.9 points for 101-200 pages. These findings confirm that our dual-graph construction and cross-modal hybrid retrieval mechanism is particularly effective for long-document reasoning tasks."
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        105,
                        411,
                        375,
                        422
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                411,
                                375,
                                422
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        411,
                                        375,
                                        422
                                    ],
                                    "type": "text",
                                    "content": "3.3 ARCHITECTURAL VALIDATION WITH ABLATION STUDIES"
                                }
                            ]
                        }
                    ],
                    "index": 9
                },
                {
                    "bbox": [
                        104,
                        431,
                        504,
                        498
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                431,
                                504,
                                498
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        431,
                                        504,
                                        498
                                    ],
                                    "type": "text",
                                    "content": "To isolate and quantify the contributions of key architectural components in RAG-Anything, we conducted systematic ablation studies examining two critical design choices. Given that our approach fundamentally differs from existing methods through dual-graph construction and hybrid retrieval, we specifically evaluated: i) Chunk-only, which bypasses graph construction entirely and relies solely on traditional chunk-based retrieval, and ii) w/o Reranker, which eliminates the cross-modal reranking component while preserving the core graph-based architecture."
                                }
                            ]
                        }
                    ],
                    "index": 10
                },
                {
                    "bbox": [
                        104,
                        503,
                        504,
                        581
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                503,
                                504,
                                581
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "text",
                                    "content": "As demonstrated in Table 4, the results validate our architectural design through striking performance variations. "
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\bullet"
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "text",
                                    "content": " Graph Construction is Essential. The chunk-only variant achieves merely "
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "inline_equation",
                                    "content": "60.0\\%"
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "text",
                                    "content": " accuracy with substantial cross-domain drops. This demonstrates that traditional chunking fails to capture structural and cross-modal relationships essential for multimodal documents. "
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\bullet"
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "text",
                                    "content": " Reranking Provides Marginal Gains. Removing the reranker yields only a modest decline to "
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "inline_equation",
                                    "content": "62.4\\%"
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "text",
                                    "content": ", while the full model achieves "
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "inline_equation",
                                    "content": "63.4\\%"
                                },
                                {
                                    "bbox": [
                                        104,
                                        503,
                                        504,
                                        581
                                    ],
                                    "type": "text",
                                    "content": " accuracy. This indicates that cross-modal reranking provides valuable refinement, but primary gains stem from our graph-based retrieval and cross-modal integration."
                                }
                            ]
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        105,
                        595,
                        195,
                        605
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                595,
                                195,
                                605
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        595,
                                        195,
                                        605
                                    ],
                                    "type": "text",
                                    "content": "3.4 CASE STUDIES"
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        104,
                        616,
                        504,
                        693
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                616,
                                504,
                                693
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "text",
                                    "content": "Multimodal documents contain rich structural information within each modality. Understanding these intra-modal structures is crucial for accurate reasoning. We analyze two representative cases from DocBench to demonstrate how RAG-Anything leverages these structures. These cases highlight a key limitation of existing methods. Baselines either rely on superficial textual cues or flatten complex visual elements into plain text. In contrast, RAG-Anything builds modality-aware graphs that preserve essential relationships (e.g., table header "
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\leftrightarrow"
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "text",
                                    "content": " cell "
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\leftrightarrow"
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "text",
                                    "content": " unit edges; panel "
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\leftrightarrow"
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "text",
                                    "content": " caption "
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "inline_equation",
                                    "content": "\\leftrightarrow"
                                },
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        504,
                                        693
                                    ],
                                    "type": "text",
                                    "content": " axis edges). This enables precise reasoning over complex document layouts."
                                }
                            ]
                        }
                    ],
                    "index": 13
                },
                {
                    "bbox": [
                        104,
                        699,
                        504,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                699,
                                504,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        699,
                                        504,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "- Case 1: Multi-panel Figure Interpretation. This case examines a common scenario in academic literature. Researchers often need to compare results across different experimental conditions. These results are typically presented in multi-panel visualizations. Figure 3 shows a challenging t-SNE"
                                }
                            ]
                        }
                    ],
                    "index": 14
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        338,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                338,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        338,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        302,
                        751,
                        308,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                302,
                                751,
                                308,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        302,
                                        751,
                                        308,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "9"
                                }
                            ]
                        }
                    ],
                    "index": 15
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 8
        },
        {
            "para_blocks": [
                {
                    "type": "image",
                    "bbox": [
                        106,
                        79,
                        258,
                        163
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                79,
                                258,
                                163
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        79,
                                        258,
                                        163
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                79,
                                                258,
                                                163
                                            ],
                                            "type": "image",
                                            "image_path": "5abdb40d6c5176c729250cae9c21a6ca90e330a217e51b6f7238aeca93f2de5d.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                104,
                                168,
                                504,
                                192
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        168,
                                        504,
                                        192
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                168,
                                                504,
                                                192
                                            ],
                                            "type": "text",
                                            "content": "Figure 3: Multi-panel figure interpretation case. The query requires identifying cluster separation patterns from the style-space panel, while avoiding confusion from the adjacent content-space panel."
                                        }
                                    ]
                                }
                            ],
                            "index": 6,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 1
                },
                {
                    "type": "image",
                    "bbox": [
                        263,
                        86,
                        361,
                        118
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                263,
                                86,
                                361,
                                118
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        263,
                                        86,
                                        361,
                                        118
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                263,
                                                86,
                                                361,
                                                118
                                            ],
                                            "type": "image",
                                            "image_path": "5bb382d86ae1c16bea75346062629dab147569e9baf0d5b6f3c8dd01e1f738e8.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 2
                },
                {
                    "type": "image",
                    "bbox": [
                        264,
                        121,
                        376,
                        158
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                264,
                                121,
                                376,
                                158
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        264,
                                        121,
                                        376,
                                        158
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                264,
                                                121,
                                                376,
                                                158
                                            ],
                                            "type": "image",
                                            "image_path": "fdd25aeb3faaa5a667553e3621c1404b3d0be7811b24638b1382584091ad29d6.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 3,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 3
                },
                {
                    "type": "image",
                    "bbox": [
                        379,
                        86,
                        499,
                        123
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                379,
                                86,
                                499,
                                123
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        379,
                                        86,
                                        499,
                                        123
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                379,
                                                86,
                                                499,
                                                123
                                            ],
                                            "type": "image",
                                            "image_path": "bcb4ce14f67622e6f93a0b52cb318b9b8d7819fa1765123d207596c85a9f081b.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 4
                },
                {
                    "type": "image",
                    "bbox": [
                        379,
                        127,
                        500,
                        159
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                379,
                                127,
                                500,
                                159
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        379,
                                        127,
                                        500,
                                        159
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                379,
                                                127,
                                                500,
                                                159
                                            ],
                                            "type": "image",
                                            "image_path": "339f8a81f5056da5479ef86f30bc4548b82b0cecc47a1310462d0a0d8a9b326c.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 5,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        104,
                        204,
                        506,
                        271
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                204,
                                506,
                                271
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        204,
                                        506,
                                        271
                                    ],
                                    "type": "text",
                                    "content": "visualization with multiple subpanels. The query requires distinguishing between two related but distinct panels. RAG-Anything constructs a visual layout graph where panels, axis titles, legends, and captions become nodes. Key edges encode semantic relationships. Panels contain specific plots. Captions provide contextual information. Subfigures relate hierarchically. This structure guides the retriever to focus on the style-space panel for comparing cluster separation patterns. The system avoids confusion from the adjacent content space panel. This panel shows less clear distinctions."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "type": "image",
                    "bbox": [
                        106,
                        280,
                        293,
                        354
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                280,
                                293,
                                354
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        280,
                                        293,
                                        354
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                280,
                                                293,
                                                354
                                            ],
                                            "type": "image",
                                            "image_path": "4efcc7b231e88868329019ba04609322a60aceebe5435cba29b3c24476259386.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 8,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                104,
                                361,
                                504,
                                384
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        361,
                                        504,
                                        384
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                361,
                                                504,
                                                384
                                            ],
                                            "type": "text",
                                            "content": "Figure 4: Financial table navigation case. The query involves locating the specific intersection of \"Wages and salaries\" row and \"2020\" column amid similar terminological entries."
                                        }
                                    ]
                                }
                            ],
                            "index": 13,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 8
                },
                {
                    "type": "image",
                    "bbox": [
                        296,
                        284,
                        372,
                        316
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                296,
                                284,
                                372,
                                316
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        296,
                                        284,
                                        372,
                                        316
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                296,
                                                284,
                                                372,
                                                316
                                            ],
                                            "type": "image",
                                            "image_path": "017b51dac7be18ff6fcddc3b5072542f031363f5716c4ad77e74f2dc9e4ef607.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 9,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 9
                },
                {
                    "type": "image",
                    "bbox": [
                        378,
                        284,
                        481,
                        307
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                378,
                                284,
                                481,
                                307
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        378,
                                        284,
                                        481,
                                        307
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                378,
                                                284,
                                                481,
                                                307
                                            ],
                                            "type": "image",
                                            "image_path": "01e1478504685d8dbcae275c2716f848420e9e291c6986d589fc4678edccbcca.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 10,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 10
                },
                {
                    "type": "image",
                    "bbox": [
                        298,
                        319,
                        371,
                        349
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                298,
                                319,
                                371,
                                349
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        298,
                                        319,
                                        371,
                                        349
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                298,
                                                319,
                                                371,
                                                349
                                            ],
                                            "type": "image",
                                            "image_path": "958f204e6bd4fafd22ffad6ad5f9cc081781762ddad334596b3d71d84ec2151b.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 11,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 11
                },
                {
                    "type": "image",
                    "bbox": [
                        377,
                        319,
                        500,
                        344
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                377,
                                319,
                                500,
                                344
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        377,
                                        319,
                                        500,
                                        344
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                377,
                                                319,
                                                500,
                                                344
                                            ],
                                            "type": "image",
                                            "image_path": "8529cfbb4c818cc4327c6dd2454e3d8084aaae1736b1272bb54238fae2205cca.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 12,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        104,
                        387,
                        504,
                        433
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                387,
                                504,
                                433
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        387,
                                        504,
                                        433
                                    ],
                                    "type": "text",
                                    "content": "- Case 2: Financial Table Navigation. This case addresses a common challenge in financial document analysis. Analysts must extract specific metrics from tables with similar terminology and multiple time periods. Figure 4 shows this scenario. The query involves resolving ambiguous financial terms and selecting the correct column for a specified year."
                                }
                            ]
                        }
                    ],
                    "index": 14
                },
                {
                    "bbox": [
                        104,
                        437,
                        506,
                        527
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                437,
                                506,
                                527
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        437,
                                        506,
                                        527
                                    ],
                                    "type": "text",
                                    "content": "RAG-Anything transforms the financial report table into a structured graph. Each row header, column header (year), data cell, and unit becomes a node. The edges capture key relationships: row-of, column-of, header-applies-to, and unit-of. This structure enables precise navigation. The retriever focuses on the row \"Wages and salaries\" and the column for \"2020\". It directs attention to the target cell (26,778 million). The system successfully disambiguates nearby entries like \"Share-based payments.\" Competing methods treat tables as linear text. They often confuse numerical spans and years. This leads to significantly inaccurate answers. RAG-Anything explicitly models relationships within the table. It achieves precise selection and numeric grounding. This ensures accurate responses."
                                }
                            ]
                        }
                    ],
                    "index": 15
                },
                {
                    "bbox": [
                        104,
                        531,
                        504,
                        588
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                531,
                                504,
                                588
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        531,
                                        504,
                                        588
                                    ],
                                    "type": "text",
                                    "content": "- Key Insights. Both cases demonstrate how RAG-Anything's structure-aware design delivers targeted advantages. Our approach transforms documents into explicit graph representations. These graphs capture intra-modal relationships that traditional methods miss. In figures, connections between panels, captions, and axes enable panel-level comparisons. This goes beyond keyword matching. In tables, row-column-unit graphs ensure accurate identification through modeling."
                                }
                            ]
                        }
                    ],
                    "index": 16
                },
                {
                    "bbox": [
                        104,
                        592,
                        506,
                        670
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                592,
                                506,
                                670
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        592,
                                        506,
                                        670
                                    ],
                                    "type": "text",
                                    "content": "This structure-aware retrieval design reduces confusion from repeated terminology and complex layouts. Traditional RAG systems struggle with these scenarios due to lack of structural understanding. Even MMGraphRAG fails here because it only considers image modality entities. It ignores other modality entities like table cells, row headers, and column headers. RAG-Anything's comprehensive graph representation captures all modality-specific entities and their relationships. This enables precise, modality-specific grounding that leads to consistent improvements in document Q&A tasks requiring fine-grained localization. Additional cases are available in Appendix A.2."
                                }
                            ]
                        }
                    ],
                    "index": 17
                },
                {
                    "bbox": [
                        105,
                        685,
                        212,
                        696
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                685,
                                212,
                                696
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        685,
                                        212,
                                        696
                                    ],
                                    "type": "text",
                                    "content": "4 RELATED WORK"
                                }
                            ]
                        }
                    ],
                    "index": 18
                },
                {
                    "bbox": [
                        104,
                        709,
                        506,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                709,
                                506,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        709,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "- Graph-Enhanced Retrieval-Augmented Generation. Large language models struggle with long-context inputs and multi-hop queries, failing to precisely locate dispersed evidence (Zhang et al.,"
                                }
                            ]
                        }
                    ],
                    "index": 19
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        338,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                338,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        338,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        299,
                        750,
                        311,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                299,
                                750,
                                311,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        299,
                                        750,
                                        311,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "10"
                                }
                            ]
                        }
                    ],
                    "index": 20
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 9
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        104,
                        82,
                        504,
                        105
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                82,
                                504,
                                105
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        82,
                                        504,
                                        105
                                    ],
                                    "type": "text",
                                    "content": "2025). Graph structures address this limitation by introducing explicit relational modeling, improving both retrieval efficiency and reasoning accuracy (Bei et al., 2025)."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        104,
                        110,
                        506,
                        210
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                110,
                                506,
                                210
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        110,
                                        506,
                                        210
                                    ],
                                    "type": "text",
                                    "content": "Since GraphRAG (Edge et al., 2024), research has evolved along two complementary directions. First, graph construction approaches optimize structures for retrieval efficiency, ranging from LightRAG's (Guo et al., 2024) sparsified indices to neural models like GNN-RAG (Mavromatis & Karypis, 2024) and memory-augmented variants like HippoRAG (Jimenez Gutierrez et al., 2024). Second, knowledge aggregation approaches integrate information for multi-level reasoning through hierarchical methods like RAPTOR (Sarthi et al., 2024) and ArchRAG (Wang et al., 2025). Despite these advances, existing systems remain text-centric with homogeneous inputs. This limits their applicability to multimodal documents and constrains robust reasoning over heterogeneous content. RAG-Anything addresses this gap by extending GraphRAG to all modalities."
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        104,
                        215,
                        506,
                        326
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                215,
                                506,
                                326
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        215,
                                        506,
                                        326
                                    ],
                                    "type": "text",
                                    "content": "- Multimodal Retrieval-Augmented Generation. Multimodal RAG represents a natural evolution from text-based RAG systems, addressing the need to integrate external knowledge from diverse data modalities for comprehensive response generation (Abootorabi et al., 2025). However, current approaches are fundamentally constrained by their reliance on modality-specific architectures. Existing methods demonstrate these constraints across domains: VideoRAG (Ren et al., 2025) employs dual-channel architectures for video understanding while MM-VID (Lin et al., 2023) converts videos to text, losing visual information; VisRAG (Yu et al., 2025) preserves document layouts as images but misses granular relationships; MMGraphRAG (Wan & Yu, 2025) links scene graphs with textual representations but suffers from structural blindness—treating tables and formulas as plain text without proper entity extraction, losing structural information for reasoning."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        104,
                        331,
                        504,
                        418
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                331,
                                504,
                                418
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        331,
                                        504,
                                        418
                                    ],
                                    "type": "text",
                                    "content": "The fundamental problem underlying these limitations is architectural fragmentation. Current systems require specialized processing pipelines for each modality. This creates poor generalizability as new modalities demand custom architectures and fusion mechanisms. Such fragmentation introduces cross-modal alignment difficulties, modality biases, and information bottlenecks. These issues systematically compromise system performance and scalability. RAG-Anything addresses this fragmentation through a unified graph-based framework. Our approach processes all modalities with consistent structured modeling. This eliminates architectural constraints while preserving multimodal information integrity. The result is seamless cross-modal reasoning across heterogeneous content."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        105,
                        434,
                        196,
                        447
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                434,
                                196,
                                447
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        434,
                                        196,
                                        447
                                    ],
                                    "type": "text",
                                    "content": "5 CONCLUSION"
                                }
                            ]
                        }
                    ],
                    "index": 5
                },
                {
                    "bbox": [
                        104,
                        460,
                        506,
                        570
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                460,
                                506,
                                570
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        460,
                                        506,
                                        570
                                    ],
                                    "type": "text",
                                    "content": "RAG-Anything introduces a paradigm shift in multimodal retrieval through its unified graph-based framework. Our core technical innovation is the dual-graph construction strategy that seamlessly integrates cross-modal and text-based knowledge graphs. Rather than forcing diverse modalities into text-centric pipelines that lose critical structural information, our approach fundamentally reconceptualizes multimodal content as interconnected knowledge entities with rich semantic relationships. The hybrid retrieval mechanism strategically combines structural navigation with semantic matching, enabling precise reasoning over complex document layouts. Comprehensive evaluation demonstrates superior performance on long-context documents, particularly those exceeding 100 pages where traditional methods fail. This work establishes a new foundation for multimodal RAG systems that can handle the heterogeneous nature of diverse information landscapes."
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        104,
                        575,
                        506,
                        654
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                575,
                                506,
                                654
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        575,
                                        506,
                                        654
                                    ],
                                    "type": "text",
                                    "content": "Our analysis in Appendix A.5 reveals critical challenges facing current multimodal RAG systems. Two fundamental issues emerge through systematic failure case examination. First, systems exhibit text-centric retrieval bias, preferentially accessing textual sources even when queries explicitly require visual information. Second, rigid spatial processing patterns fail to adapt to non-standard document layouts. These limitations manifest in cross-modal misalignment scenarios and structurally ambiguous tables. The findings highlight the need for adaptive spatial reasoning and layout-aware parsing mechanisms to handle real-world multimodal document complexity."
                                }
                            ]
                        }
                    ],
                    "index": 7
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        337,
                        36
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                337,
                                36
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        337,
                                        36
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        299,
                        750,
                        310,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                299,
                                750,
                                310,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        299,
                                        750,
                                        310,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "11"
                                }
                            ]
                        }
                    ],
                    "index": 8
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 10
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        106,
                        81,
                        176,
                        94
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                81,
                                176,
                                94
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        81,
                                        176,
                                        94
                                    ],
                                    "type": "text",
                                    "content": "REFERENCES"
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        105,
                        99,
                        507,
                        715
                    ],
                    "type": "list",
                    "angle": 0,
                    "index": 18,
                    "blocks": [
                        {
                            "bbox": [
                                105,
                                99,
                                507,
                                146
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        99,
                                        507,
                                        146
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                99,
                                                507,
                                                146
                                            ],
                                            "type": "text",
                                            "content": "Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. arXiv preprint arXiv:2502.08826, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 2
                        },
                        {
                            "bbox": [
                                105,
                                150,
                                507,
                                186
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        150,
                                        507,
                                        186
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                150,
                                                507,
                                                186
                                            ],
                                            "type": "text",
                                            "content": "Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu, Shirui Pan, Yizhou Yu, et al. Graphs meet ai agents: Taxonomy, progress, and future opportunities. arXiv preprint arXiv:2506.18019, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 3
                        },
                        {
                            "bbox": [
                                105,
                                191,
                                506,
                                228
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        191,
                                        506,
                                        228
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                191,
                                                506,
                                                228
                                            ],
                                            "type": "text",
                                            "content": "Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 4
                        },
                        {
                            "bbox": [
                                105,
                                232,
                                506,
                                257
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        232,
                                        506,
                                        257
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                232,
                                                506,
                                                257
                                            ],
                                            "type": "text",
                                            "content": "Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 5
                        },
                        {
                            "bbox": [
                                105,
                                261,
                                507,
                                297
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        261,
                                        507,
                                        297
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                261,
                                                507,
                                                297
                                            ],
                                            "type": "text",
                                            "content": "Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. NeurIPS, 37:59532-59569, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 6
                        },
                        {
                            "bbox": [
                                105,
                                302,
                                506,
                                339
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        302,
                                        506,
                                        339
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                302,
                                                506,
                                                339
                                            ],
                                            "type": "text",
                                            "content": "Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, Ce Liu, and Lijuan Wang. Mm-vid: Advancing video understanding with gpt-4v(ison). arXiv preprint arXiv:2310.19773, 2023."
                                        }
                                    ]
                                }
                            ],
                            "index": 7
                        },
                        {
                            "bbox": [
                                105,
                                343,
                                507,
                                380
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        343,
                                        507,
                                        380
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                343,
                                                507,
                                                380
                                            ],
                                            "type": "text",
                                            "content": "Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems, 37:95963-96010, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 8
                        },
                        {
                            "bbox": [
                                105,
                                384,
                                506,
                                410
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        384,
                                        506,
                                        410
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                384,
                                                506,
                                                410
                                            ],
                                            "type": "text",
                                            "content": "Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language model reasoning. arXiv preprint arXiv:2405.20139, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 9
                        },
                        {
                            "bbox": [
                                105,
                                414,
                                507,
                                449
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        414,
                                        507,
                                        449
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                414,
                                                507,
                                                449
                                            ],
                                            "type": "text",
                                            "content": "Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-augmented generation with extreme long-context videos. arXiv preprint arXiv:2502.01549, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 10
                        },
                        {
                            "bbox": [
                                105,
                                455,
                                507,
                                491
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        455,
                                        507,
                                        491
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                455,
                                                507,
                                                491
                                            ],
                                            "type": "text",
                                            "content": "Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In *The Twelfth International Conference on Learning Representations*, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 11
                        },
                        {
                            "bbox": [
                                105,
                                495,
                                506,
                                521
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        495,
                                        506,
                                        521
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                495,
                                                506,
                                                521
                                            ],
                                            "type": "text",
                                            "content": "Xueyao Wan and Hang Yu. Mmgraphrag: Bridging vision and language with interpretable multimodal knowledge graphs. arXiv preprint arXiv:2507.20804, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 12
                        },
                        {
                            "bbox": [
                                105,
                                526,
                                507,
                                562
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        526,
                                        507,
                                        562
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                526,
                                                507,
                                                562
                                            ],
                                            "type": "text",
                                            "content": "Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 13
                        },
                        {
                            "bbox": [
                                105,
                                567,
                                507,
                                592
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        567,
                                        507,
                                        592
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                567,
                                                507,
                                                592
                                            ],
                                            "type": "text",
                                            "content": "Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. Archrag: Attributed community-based hierarchical retrieval-augmented generation. arXiv preprint arXiv:2502.09891, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 14
                        },
                        {
                            "bbox": [
                                105,
                                597,
                                507,
                                633
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        597,
                                        507,
                                        633
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                597,
                                                507,
                                                633
                                            ],
                                            "type": "text",
                                            "content": "Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 15
                        },
                        {
                            "bbox": [
                                105,
                                637,
                                507,
                                673
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        637,
                                        507,
                                        673
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                637,
                                                507,
                                                673
                                            ],
                                            "type": "text",
                                            "content": "Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Yi Chang, and Xiao Huang. A survey of graph retrieval-augmented generation for customized large language models. arXiv preprint arXiv:2501.13958, 2025."
                                        }
                                    ]
                                }
                            ],
                            "index": 16
                        },
                        {
                            "bbox": [
                                105,
                                678,
                                507,
                                715
                            ],
                            "type": "ref_text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        105,
                                        678,
                                        507,
                                        715
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                105,
                                                678,
                                                507,
                                                715
                                            ],
                                            "type": "text",
                                            "content": "Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. Docbench: A benchmark for evaluating lvm-based document reading systems. arXiv preprint arXiv:2407.10701, 2024."
                                        }
                                    ]
                                }
                            ],
                            "index": 17
                        }
                    ],
                    "sub_type": "ref_text"
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        340,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                340,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        340,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        300,
                        750,
                        311,
                        761
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                300,
                                750,
                                311,
                                761
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        300,
                                        750,
                                        311,
                                        761
                                    ],
                                    "type": "text",
                                    "content": "12"
                                }
                            ]
                        }
                    ],
                    "index": 19
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 11
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        105,
                        81,
                        185,
                        94
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                81,
                                185,
                                94
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        81,
                                        185,
                                        94
                                    ],
                                    "type": "text",
                                    "content": "A APPENDIX"
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        104,
                        105,
                        507,
                        196
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                105,
                                507,
                                196
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        105,
                                        507,
                                        196
                                    ],
                                    "type": "text",
                                    "content": "This appendix provides comprehensive supporting materials for our experimental evaluation and implementation details. Section A.1 presents detailed dataset statistics for the DocBench and MMLongBench multi-modal benchmarks, including document type distributions and complexity metrics. Section A.2 showcases additional case studies that demonstrate RAG-Anything's structure-aware capabilities across diverse multimodal content understanding tasks. Section A.3 documents the complete set of multimodal analysis prompts for vision, table, and equation processing that enable context-aware interpretation. Section A.4 provides the standardized accuracy evaluation prompt used for consistent response assessment across all experimental conditions."
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        105,
                        206,
                        328,
                        217
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                206,
                                328,
                                217
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        206,
                                        328,
                                        217
                                    ],
                                    "type": "text",
                                    "content": "A.1 DATASET CHARACTERISTICS AND STATISTICS"
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "type": "table",
                    "bbox": [
                        194,
                        248,
                        416,
                        306
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                194,
                                248,
                                416,
                                306
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        194,
                                        248,
                                        416,
                                        306
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                194,
                                                248,
                                                416,
                                                306
                                            ],
                                            "type": "table",
                                            "html": "<table><tr><td>Type</td><td>Acad.</td><td>Fin.</td><td>Gov.</td><td>Law.</td><td>News</td></tr><tr><td># Docs</td><td>49</td><td>40</td><td>44</td><td>46</td><td>50</td></tr><tr><td># Questions</td><td>303</td><td>288</td><td>148</td><td>191</td><td>172</td></tr><tr><td>Avg. Pages</td><td>11</td><td>192</td><td>69</td><td>58</td><td>1</td></tr></table>",
                                            "image_path": "33981691987ca5749049465db763f9c0e43875229ba6d96c9745466747165c90.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 5,
                            "angle": 0,
                            "type": "table_body"
                        },
                        {
                            "bbox": [
                                141,
                                228,
                                468,
                                240
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        141,
                                        228,
                                        468,
                                        240
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                141,
                                                228,
                                                468,
                                                240
                                            ],
                                            "type": "text",
                                            "content": "Table 5: Document type distribution and statistics for the DocBench benchmark."
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "table_caption"
                        }
                    ],
                    "index": 5
                },
                {
                    "type": "table",
                    "bbox": [
                        156,
                        338,
                        455,
                        396
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                156,
                                338,
                                455,
                                396
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        156,
                                        338,
                                        455,
                                        396
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                156,
                                                338,
                                                455,
                                                396
                                            ],
                                            "type": "table",
                                            "html": "<table><tr><td>Type</td><td>Res.</td><td>Tut.</td><td>Acad.</td><td>Guid.</td><td>Broch.</td><td>Admin.</td><td>Fin.</td></tr><tr><td># Docs</td><td>34</td><td>17</td><td>26</td><td>22</td><td>15</td><td>10</td><td>11</td></tr><tr><td># Questions</td><td>292</td><td>138</td><td>199</td><td>155</td><td>100</td><td>81</td><td>117</td></tr><tr><td>Avg. Pages</td><td>39</td><td>58</td><td>35</td><td>78</td><td>30</td><td>17</td><td>87</td></tr></table>",
                                            "image_path": "968b2751f1362050c4c4adf30dfc6ad94c9d6eab5836a3535a145576ab107877.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 7,
                            "angle": 0,
                            "type": "table_body"
                        },
                        {
                            "bbox": [
                                131,
                                318,
                                477,
                                331
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        131,
                                        318,
                                        477,
                                        331
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                131,
                                                318,
                                                477,
                                                331
                                            ],
                                            "type": "text",
                                            "content": "Table 6: Document type distribution and statistics for the MMLongBench benchmark."
                                        }
                                    ]
                                }
                            ],
                            "index": 6,
                            "angle": 0,
                            "type": "table_caption"
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        407,
                        506,
                        497
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                407,
                                506,
                                497
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        407,
                                        506,
                                        497
                                    ],
                                    "type": "text",
                                    "content": "Tables 5 and 6 present the distribution of document types across the DocBench and MMLong-Bench benchmarks. • DocBench encompasses medium- to long-length documents spanning various domains, including legal, governmental, and financial files. Financial reports represent the most extensive category, averaging 192 pages per document, while the News category consists of concise single-page newspapers. • MMLongBench demonstrates a broader spectrum of document types and lengths. Research reports, tutorials, and academic papers maintain moderate lengths of 35–58 pages on average, while guidebooks extend to approximately 78 pages. Brochures and administrative files remain relatively compact, whereas financial reports again emerge as the longest category."
                                }
                            ]
                        }
                    ],
                    "index": 8
                },
                {
                    "bbox": [
                        104,
                        501,
                        507,
                        536
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                501,
                                507,
                                536
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        501,
                                        507,
                                        536
                                    ],
                                    "type": "text",
                                    "content": "Collectively, these two benchmarks provide comprehensive coverage ranging from brief news articles to extensive technical and financial documentation. This establishes diverse and challenging evaluation contexts for multimodal document understanding tasks."
                                }
                            ]
                        }
                    ],
                    "index": 9
                },
                {
                    "bbox": [
                        105,
                        548,
                        257,
                        559
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                548,
                                257,
                                559
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        548,
                                        257,
                                        559
                                    ],
                                    "type": "text",
                                    "content": "A.2 ADDITIONAL CASE STUDIES"
                                }
                            ]
                        }
                    ],
                    "index": 10
                },
                {
                    "type": "image",
                    "bbox": [
                        106,
                        569,
                        279,
                        666
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                569,
                                279,
                                666
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        569,
                                        279,
                                        666
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                569,
                                                279,
                                                666
                                            ],
                                            "type": "image",
                                            "image_path": "cfa0548ca603857a89bc882628fb87b65132d8a0a78cae8bba5a64e041d34f60.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 11,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                104,
                                670,
                                506,
                                693
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        670,
                                        506,
                                        693
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                670,
                                                506,
                                                693
                                            ],
                                            "type": "text",
                                            "content": "Figure 5: Visual reasoning case. RAG-Anything correctly identifies \"-S-A\" as the lowest accuracy configuration, while baselines misinterpret spatial relationships."
                                        }
                                    ]
                                }
                            ],
                            "index": 16,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        285,
                        574,
                        355,
                        610
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                285,
                                574,
                                355,
                                610
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        285,
                                        574,
                                        355,
                                        610
                                    ],
                                    "type": "text",
                                    "content": "GPT-4o-mini: The removal of the dual coattention mechanism from the GCAN sub-model resulted in the lowest accuracy for Twitter15."
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        285,
                        622,
                        386,
                        657
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                285,
                                622,
                                386,
                                657
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        285,
                                        622,
                                        386,
                                        657
                                    ],
                                    "type": "text",
                                    "content": "MMGraphRAG: The removal of the source tweet from the GCAN model resulted in the lowest accuracy for Twitter15, as indicated by significant accuracy drops in the ablation analysis in Figure 4."
                                }
                            ]
                        }
                    ],
                    "index": 13
                },
                {
                    "bbox": [
                        389,
                        575,
                        499,
                        616
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                389,
                                575,
                                499,
                                616
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        389,
                                        575,
                                        499,
                                        616
                                    ],
                                    "type": "text",
                                    "content": "LightRAG: Removing the source tweet from the GCAN model resulted in the lowest accuracy for Twitter15, as indicated by a significant drop in performance when using the sub-model that excluded both source tweet embeddings and dual co-attention."
                                }
                            ]
                        }
                    ],
                    "index": 14
                },
                {
                    "bbox": [
                        389,
                        621,
                        494,
                        659
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                389,
                                621,
                                494,
                                659
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        389,
                                        621,
                                        494,
                                        659
                                    ],
                                    "type": "text",
                                    "content": "RAG-Anything(Correct): The removal of the source tweet embeddings and dual co-attention, indicated as model configuration \"S-A,\" resulted in the lowest accuracy for Twitter15."
                                }
                            ]
                        }
                    ],
                    "index": 15
                },
                {
                    "bbox": [
                        104,
                        698,
                        507,
                        734
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                698,
                                507,
                                734
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        698,
                                        507,
                                        734
                                    ],
                                    "type": "text",
                                    "content": "- Demonstrating Visual Reasoning Capabilities. Figure 5 illustrates how RAG-Anything handles complex visual reasoning tasks involving chart interpretation. The query asks which GCAN sub-model component removal yields the lowest accuracy on Twitter15. Traditional approaches struggle"
                                }
                            ]
                        }
                    ],
                    "index": 17
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        340,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                340,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        340,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        299,
                        750,
                        311,
                        761
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                299,
                                750,
                                311,
                                761
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        299,
                                        750,
                                        311,
                                        761
                                    ],
                                    "type": "text",
                                    "content": "13"
                                }
                            ]
                        }
                    ],
                    "index": 18
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 12
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        104,
                        82,
                        504,
                        117
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                82,
                                504,
                                117
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        82,
                                        504,
                                        117
                                    ],
                                    "type": "text",
                                    "content": "with spatial relationships between visual elements. RAG-Anything addresses this challenge by constructing a structured graph representation of the bar plot. Bars, axis labels, and legends become interconnected nodes. These are linked by semantic relations such as bar-of and label-applies-to."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        104,
                        121,
                        506,
                        188
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                121,
                                506,
                                188
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        121,
                                        506,
                                        188
                                    ],
                                    "type": "text",
                                    "content": "This graph-based approach enables precise alignment between visual and textual elements. The system correctly identifies the bar labeled \"-S-A\" (removing source tweet embeddings and dual co-attention) and its corresponding accuracy value as the lowest performer. Baseline methods that flatten visual information often misinterpret spatial relationships. They frequently conflate nearby components. RAG-Anything's structured representation preserves critical visual-textual associations. This leads to accurate query resolution and proper attribution of performance drops to \"-S-A\"."
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "type": "image",
                    "bbox": [
                        106,
                        208,
                        261,
                        293
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                208,
                                261,
                                293
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        208,
                                        261,
                                        293
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                208,
                                                261,
                                                293
                                            ],
                                            "type": "image",
                                            "image_path": "d53d5f010b3509d6f1e8c9e0c888f94af69668846221e2a3fbd92e38ee0a8e1e.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 3,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                104,
                                301,
                                504,
                                324
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        301,
                                        504,
                                        324
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                301,
                                                504,
                                                324
                                            ],
                                            "type": "text",
                                            "content": "Figure 6: Tabular navigation case. RAG-Anything locates the highest AUPRC value (0.506), while the compared approaches struggle with structural ambiguity."
                                        }
                                    ]
                                }
                            ],
                            "index": 8,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 3
                },
                {
                    "type": "image",
                    "bbox": [
                        264,
                        213,
                        376,
                        243
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                264,
                                213,
                                376,
                                243
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        264,
                                        213,
                                        376,
                                        243
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                264,
                                                213,
                                                376,
                                                243
                                            ],
                                            "type": "image",
                                            "image_path": "6c0bd9fbf06a20153dcb8c3dbe74800147dab2e5473394b918442bece356fcfe.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 4
                },
                {
                    "type": "image",
                    "bbox": [
                        264,
                        251,
                        380,
                        285
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                264,
                                251,
                                380,
                                285
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        264,
                                        251,
                                        380,
                                        285
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                264,
                                                251,
                                                380,
                                                285
                                            ],
                                            "type": "image",
                                            "image_path": "8b5d650838a0be291a112adb1f94ae0fa96c354b14af0052b7c8c34d04b3bdbb.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 5,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 5
                },
                {
                    "type": "image",
                    "bbox": [
                        384,
                        213,
                        497,
                        242
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                384,
                                213,
                                497,
                                242
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        384,
                                        213,
                                        497,
                                        242
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                384,
                                                213,
                                                497,
                                                242
                                            ],
                                            "type": "image",
                                            "image_path": "49f670bf5d7dcfd4a94f7042d1716e53453712f5b3c1ba4da921d9d045960de1.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 6,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 6
                },
                {
                    "type": "image",
                    "bbox": [
                        384,
                        251,
                        497,
                        289
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                384,
                                251,
                                497,
                                289
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        384,
                                        251,
                                        497,
                                        289
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                384,
                                                251,
                                                497,
                                                289
                                            ],
                                            "type": "image",
                                            "image_path": "9fa77e628c8067bd981cc79c269ff7e69ec7dabba1b578b2c7df81976a9482f4.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 7,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        339,
                        506,
                        396
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                339,
                                506,
                                396
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        339,
                                        506,
                                        396
                                    ],
                                    "type": "text",
                                    "content": "- Handling Complex Tabular Structures. Figure 6 showcases RAG-Anything's ability to navigate intricate tabular data where structural disambiguation is crucial. The query seeks the model combination achieving the highest AUPRC value for the Evidence Inference dataset—a task complicated by repeated row labels across multiple datasets within the same table. This scenario highlights a fundamental limitation of conventional approaches that struggle with structural ambiguity in data."
                                }
                            ]
                        }
                    ],
                    "index": 9
                },
                {
                    "bbox": [
                        104,
                        400,
                        506,
                        467
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                400,
                                506,
                                467
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        400,
                                        506,
                                        467
                                    ],
                                    "type": "text",
                                    "content": "RAG-Anything overcomes this by parsing the table into a comprehensive relational graph where headers and data cells become nodes connected through explicit row-of and column-of relationships. This structured representation enables the system to correctly isolate the Evidence Inference dataset context and identify \"GloVe + LSTM - Attention\" with a score of 0.506 as the optimal configuration. By explicitly preserving hierarchical table constraints that other methods often collapse or misinterpret, RAG-Anything ensures reliable reasoning across complex multi-dataset tabular structures."
                                }
                            ]
                        }
                    ],
                    "index": 10
                },
                {
                    "bbox": [
                        105,
                        491,
                        327,
                        502
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                491,
                                327,
                                502
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        491,
                                        327,
                                        502
                                    ],
                                    "type": "text",
                                    "content": "A.3 CONTEXT-AWARE MULTIMODAL PROMPTING"
                                }
                            ]
                        }
                    ],
                    "index": 11
                },
                {
                    "bbox": [
                        104,
                        516,
                        506,
                        550
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                516,
                                506,
                                550
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        516,
                                        506,
                                        550
                                    ],
                                    "type": "text",
                                    "content": "These three prompts orchestrate structured, context-aware multimodal analysis with JSON-formatted outputs. They systematically guide the model to extract comprehensive descriptions of visual, tabular, and mathematical content while maintaining explicit alignment with surrounding information."
                                }
                            ]
                        }
                    ],
                    "index": 12
                },
                {
                    "bbox": [
                        104,
                        555,
                        506,
                        612
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                555,
                                506,
                                612
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        555,
                                        506,
                                        612
                                    ],
                                    "type": "text",
                                    "content": "Vision Analysis Prompt. Figure 7 orchestrates comprehensive image-context integration. The prompt directs the model to systematically capture compositional elements, object relationships, visual attributes, stylistic features, dynamic actions, and technical components (e.g., charts), while establishing explicit connections to accompanying text. This approach transcends superficial description, enabling contextually-grounded interpretations that enhance knowledge retrieval and substantiation."
                                }
                            ]
                        }
                    ],
                    "index": 13
                },
                {
                    "bbox": [
                        104,
                        616,
                        506,
                        672
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                616,
                                506,
                                672
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        616,
                                        506,
                                        672
                                    ],
                                    "type": "text",
                                    "content": "Table Analysis Prompt. Figure 8 structures systematic tabular content decomposition across multiple analytical dimensions: structural organization, column semantics, critical values, statistical patterns, and contextual relevance. Through precise terminology and numerical accuracy requirements, the prompt eliminates ambiguous generalizations and ensures faithful preservation of key indicators while maintaining coherent alignment with surrounding discourse."
                                }
                            ]
                        }
                    ],
                    "index": 14
                },
                {
                    "bbox": [
                        104,
                        677,
                        506,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                677,
                                506,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        677,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "Equation Analysis Prompt. Figure 9 prioritizes semantic interpretation over syntactic restatement of mathematical expressions. The prompt instructs comprehensive analysis of variable definitions, operational logic, theoretical foundations, inter-formula relationships, and practical applications. This methodology ensures mathematical content becomes integral to broader argumentative frameworks, supporting enhanced retrieval accuracy, analytical traceability, and reasoning coherence."
                                }
                            ]
                        }
                    ],
                    "index": 15
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        105,
                        26,
                        338,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                26,
                                338,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        26,
                                        338,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        299,
                        750,
                        311,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                299,
                                750,
                                311,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        299,
                                        750,
                                        311,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "14"
                                }
                            ]
                        }
                    ],
                    "index": 16
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 13
        },
        {
            "para_blocks": [
                {
                    "type": "code",
                    "bbox": [
                        118,
                        103,
                        492,
                        349
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                118,
                                103,
                                492,
                                349
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        118,
                                        103,
                                        492,
                                        349
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                118,
                                                103,
                                                492,
                                                349
                                            ],
                                            "type": "text",
                                            "content": "1 Please analyze this image in detail, considering the surrounding context. Provide a JSON response with the   \n2 following structure:   \n3   \n4 {   \n5 \"detailed_description\": \"A comprehensive and detailed visual description of the image following these   \n6 guidelines:   \n7 - Describe the overall composition and layout   \n8 - Identify all objects, people, text, and visual elements   \n9 - Explain relationships between elements and how they relate to the surrounding context   \n10 - Note colors, lighting, and visual style   \n11 - Describe any actions or activities shown   \n12 - Include technical details if relevant (charts, diagrams, etc.)   \n13 - Reference connections to the surrounding content when relevant   \n14 - Always use specific names instead of pronouns\",   \n15 \"entity_info\": {   \n16 \"entity_name\": {\"entity_name\"},   \n17 \"entity_type\": \"image\",   \n18 \"summary\": \"concise summary of the image content, its significance, and relationship to surrounding content   \n19 (max 100 words)\"   \n20 }   \n21 }   \n22   \n23 Context from surrounding content:   \n24 {context}   \n25   \n26 Image details:   \n27 - Image Path: {image_path}   \n28 - Captions: {captions}   \n29 - Footnotes: {footnotes}   \n30   \n31 Focus on providing accurate, detailed visual analysis that incorporates the context and would be useful for   \n32 knowledge retrieval."
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "code_body"
                        },
                        {
                            "bbox": [
                                116,
                                91,
                                221,
                                101
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        116,
                                        91,
                                        221,
                                        101
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                116,
                                                91,
                                                221,
                                                101
                                            ],
                                            "type": "text",
                                            "content": "Vision Analysis Prompt"
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "code_caption"
                        }
                    ],
                    "index": 2,
                    "sub_type": "code",
                    "guess_lang": "txt"
                },
                {
                    "bbox": [
                        105,
                        365,
                        503,
                        377
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                365,
                                503,
                                377
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        365,
                                        503,
                                        377
                                    ],
                                    "type": "text",
                                    "content": "Figure 7: Vision analysis prompt for context-aware image interpretation and knowledge extraction."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "type": "code",
                    "bbox": [
                        118,
                        424,
                        492,
                        668
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                118,
                                424,
                                492,
                                668
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        118,
                                        424,
                                        492,
                                        668
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                118,
                                                424,
                                                492,
                                                668
                                            ],
                                            "type": "text",
                                            "content": "table_analysis_prompt.png\n1 Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:\n2 following structure:\n3\n4 {\n5 \"detailed_description\": \"A comprehensive analysis of the table including:\n6 - Table structure and organization\n7 - Column headers and their meanings\n8 - Key data points and patterns\n9 - Statistical insights and trends\n10 - Relationships between data elements\n11 - Significance of the data presented in relation to surrounding context\n12 - How the table supports or illustrates concepts from the surrounding content\n13 Always use specific names and values instead of general references.\", \n14 \"entity_info\": {\n15 \"entity_name\": \"{entity_name}\", \n16 \"entity_type\": \"table\",\n17 \"summary\": \"concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)\"\n18 }\n19 }\n20 }\n21 }\n22 Context from surrounding content:\n23 {context}\n24 }\n25 Table Information:\n26 Image Path: {table_img_path}\n27 Caption: {table_caption}\n28 Body: {table_body}\n29 Footnotes: {table_footnote}\n30\n31 Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content."
                                        }
                                    ]
                                }
                            ],
                            "index": 5,
                            "angle": 0,
                            "type": "code_body"
                        },
                        {
                            "bbox": [
                                116,
                                411,
                                216,
                                421
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        116,
                                        411,
                                        216,
                                        421
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                116,
                                                411,
                                                216,
                                                421
                                            ],
                                            "type": "text",
                                            "content": "Table Analysis Prompt"
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "code_caption"
                        }
                    ],
                    "index": 5,
                    "sub_type": "code",
                    "guess_lang": "txt"
                },
                {
                    "bbox": [
                        106,
                        685,
                        501,
                        696
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                685,
                                501,
                                696
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        685,
                                        501,
                                        696
                                    ],
                                    "type": "text",
                                    "content": "Figure 8: Table analysis prompt for structured content decomposition and semantic understanding."
                                }
                            ]
                        }
                    ],
                    "index": 6
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        300,
                        750,
                        310,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                300,
                                750,
                                310,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        300,
                                        750,
                                        310,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "15"
                                }
                            ]
                        }
                    ],
                    "index": 7
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 14
        },
        {
            "para_blocks": [
                {
                    "type": "code",
                    "bbox": [
                        118,
                        103,
                        492,
                        342
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                118,
                                103,
                                492,
                                342
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        118,
                                        103,
                                        492,
                                        342
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                118,
                                                103,
                                                492,
                                                342
                                            ],
                                            "type": "text",
                                            "content": "equation_analysis_prompt.png   \n1 Please analyze this mathematical equation considering the surrounding context, and provide a JSON response   \n2 with the following structure:   \n3   \n4 {   \n5 \"detailed_description\": \"A comprehensive analysis of the equation including:   \n6 - Mathematical meaning and interpretation   \n7 - Variables and their definitions in the context of surrounding content   \n8 - Mathematical operations and functions used   \n9 - Application domain and context based on surrounding material   \n10 - Physical or theoretical significance   \n11 - Relationship to other mathematical concepts mentioned in the context   \n12 - Practical applications or use cases   \n13 - How the equation relates to the broader discussion or framework   \n14 Always use specific mathematical terminology.\",   \n15 \"entity_info\": {   \n16 \"entity_name\": \"{entity_name}\",   \n17 \"entity_type\": \"equation\",   \n18 \"summary\": \"concise summary of the equation's purpose, significance, and role in the surrounding context (max   \n19 100 words)\"   \n20 }   \n21 }   \n22   \nContext from surrounding content:   \n24 {context}   \n25   \nEquation Information:   \n27 Equation: {equation_text}   \n28 Format: {equation_format}   \n29   \nFocus on providing mathematical insights and explaining the equation's significance within the broader   \n31 context."
                                        }
                                    ]
                                }
                            ],
                            "index": 2,
                            "angle": 0,
                            "type": "code_body"
                        },
                        {
                            "bbox": [
                                116,
                                91,
                                231,
                                101
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        116,
                                        91,
                                        231,
                                        101
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                116,
                                                91,
                                                231,
                                                101
                                            ],
                                            "type": "text",
                                            "content": "Equation Analysis Prompt"
                                        }
                                    ]
                                }
                            ],
                            "index": 1,
                            "angle": 0,
                            "type": "code_caption"
                        }
                    ],
                    "index": 2,
                    "sub_type": "code",
                    "guess_lang": "txt"
                },
                {
                    "bbox": [
                        114,
                        354,
                        495,
                        367
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                114,
                                354,
                                495,
                                367
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        114,
                                        354,
                                        495,
                                        367
                                    ],
                                    "type": "text",
                                    "content": "Figure 9: Equation analysis prompt for mathematical expression interpretation and integration."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "type": "code",
                    "bbox": [
                        118,
                        388,
                        492,
                        642
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                118,
                                388,
                                492,
                                642
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        118,
                                        388,
                                        492,
                                        642
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                118,
                                                388,
                                                492,
                                                642
                                            ],
                                            "type": "text",
                                            "content": "accuracy_evaluation_prompt.png   \n1 You are an expert evaluator tasked with assessing the accuracy of answers generated by a RAG   \n2 (Retrieval-Augmented Generation) system.   \n3   \n4 \\*\\*Task\\*: Evaluate whether the generated answer correctly responds to the given question based on the expected   \n5 answer.   \n6   \n7 \\*\\*Question\\*: {question}   \n8   \n9 \\*\\*Expected Answer\\*: {expected_answer}   \n10   \n11 \\*\\*Generated Answer\\*: {generated_answer}   \n12   \n13   \n14 \\*\\*Evaluation Criteria\\*:   \n15 1. \\*\\*Accuracy (0 or 1)\\*: Does the generated answer match the factual content of the expected answer?   \n16 - 1: The generated answer is factually correct and aligns with the expected answer   \n17 - 0: The generated answer is factually incorrect or contradicts the expected answer   \n18   \n19 \\*\\*Instructions\\*:   \n20 - Focus on factual correctness, not writing style or format   \n21 - Consider partial matches: if the generated answer contains the correct information but includes additional   \n22 context, it should still be considered accurate   \n23 - For numerical answers, check if the values match or are equivalent   \n24 - For list answers, check if all key elements are present   \n25 - If the expected answer is \"Not answerable\" and the generated answer indicates inability to answer, consider   \n26 it accurate   \n27   \n28 \\*\\*Output Format\\*:   \n29 Please respond with a JSON object containing only:   \n30 {   \n31 \"accuracy\": 0 or 1,   \n32 \"reasoning\": \"Brief explanation of your evaluation\"   \n33 }"
                                        }
                                    ]
                                }
                            ],
                            "index": 5,
                            "angle": 0,
                            "type": "code_body"
                        },
                        {
                            "bbox": [
                                115,
                                374,
                                240,
                                385
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        115,
                                        374,
                                        240,
                                        385
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                115,
                                                374,
                                                240,
                                                385
                                            ],
                                            "type": "text",
                                            "content": "Accuracy Evaluation Prompt"
                                        }
                                    ]
                                }
                            ],
                            "index": 4,
                            "angle": 0,
                            "type": "code_caption"
                        }
                    ],
                    "index": 5,
                    "sub_type": "code",
                    "guess_lang": "txt"
                },
                {
                    "bbox": [
                        113,
                        656,
                        495,
                        669
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                113,
                                656,
                                495,
                                669
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        113,
                                        656,
                                        495,
                                        669
                                    ],
                                    "type": "text",
                                    "content": "Figure 10: Accuracy evaluation prompt for consistent factual assessment across question types."
                                }
                            ]
                        }
                    ],
                    "index": 6
                },
                {
                    "bbox": [
                        105,
                        677,
                        316,
                        688
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                677,
                                316,
                                688
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        677,
                                        316,
                                        688
                                    ],
                                    "type": "text",
                                    "content": "A.4 ACCURACY EVALUATION PROMPT DESIGN"
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "bbox": [
                        104,
                        698,
                        506,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                698,
                                506,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        698,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "Figure 10 presents the standardized prompt specifically designed for systematic factual accuracy assessment of generated responses across multiple domains. The prompt establishes explicit evaluation criteria that prioritize content correctness over stylistic considerations, producing binary accuracy"
                                }
                            ]
                        }
                    ],
                    "index": 8
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        300,
                        751,
                        311,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                300,
                                751,
                                311,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        300,
                                        751,
                                        311,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "16"
                                }
                            ]
                        }
                    ],
                    "index": 9
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 15
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        104,
                        82,
                        504,
                        118
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                82,
                                504,
                                118
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        82,
                                        504,
                                        118
                                    ],
                                    "type": "text",
                                    "content": "classifications accompanied by concise analytical justifications. All accuracy evaluations throughout our comprehensive experimental framework were conducted using GPT-4o-mini, ensuring consistent and reliable assessment standards across diverse question categories and specialized domains."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        105,
                        129,
                        418,
                        140
                    ],
                    "type": "title",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                105,
                                129,
                                418,
                                140
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        105,
                                        129,
                                        418,
                                        140
                                    ],
                                    "type": "text",
                                    "content": "A.5 CHALLENGES AND FUTURE DIRECTIONS FOR MULTI-MODAL RAG"
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        104,
                        149,
                        504,
                        228
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                149,
                                504,
                                228
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        149,
                                        504,
                                        228
                                    ],
                                    "type": "text",
                                    "content": "While current multimodal RAG systems demonstrate promising capabilities, their limitations emerge most clearly through systematic analysis of failure cases. Understanding where and why these systems break down is crucial for advancing the field beyond current performance plateaus. Examining failure patterns helps identify fundamental architectural bottlenecks and design principles for more robust multimodal systems. Our investigation reveals two critical failure patterns exposing deeper systemic issues in multimodal RAG architectures. These patterns are not merely edge cases but reflect fundamental challenges in cross-modal information integration and structural reasoning:"
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        104,
                        236,
                        506,
                        288
                    ],
                    "type": "list",
                    "angle": 0,
                    "index": 6,
                    "blocks": [
                        {
                            "bbox": [
                                104,
                                236,
                                504,
                                259
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        236,
                                        504,
                                        259
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                236,
                                                504,
                                                259
                                            ],
                                            "type": "text",
                                            "content": "- Text-Centric Retrieval Bias: Systems exhibit strong preference for textual sources, even when queries explicitly demand visual information. This reveals inadequate cross-modal attention."
                                        }
                                    ]
                                }
                            ],
                            "index": 4
                        },
                        {
                            "bbox": [
                                104,
                                264,
                                506,
                                288
                            ],
                            "type": "text",
                            "angle": 0,
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        264,
                                        506,
                                        288
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                264,
                                                506,
                                                288
                                            ],
                                            "type": "text",
                                            "content": "- Document Structure Processing Challenges: Systems struggle with complex layouts and nonlinear information flows. This exposes limitations in spatial reasoning and contextual understanding."
                                        }
                                    ]
                                }
                            ],
                            "index": 5
                        }
                    ],
                    "sub_type": "text"
                },
                {
                    "bbox": [
                        104,
                        292,
                        504,
                        316
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                292,
                                504,
                                316
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        292,
                                        504,
                                        316
                                    ],
                                    "type": "text",
                                    "content": "These failure modes illuminate key insights about current multimodal AI. They provide concrete directions for architectural innovations that could substantially improve system robustness."
                                }
                            ]
                        }
                    ],
                    "index": 7
                },
                {
                    "type": "image",
                    "bbox": [
                        106,
                        327,
                        245,
                        417
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                327,
                                245,
                                417
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        327,
                                        245,
                                        417
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                327,
                                                245,
                                                417
                                            ],
                                            "type": "image",
                                            "image_path": "2662a2f4905577d7008213b838cf2d581b9314741c3a427344d3a573e7e6dd62.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 8,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                104,
                                427,
                                504,
                                450
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        427,
                                        504,
                                        450
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                427,
                                                504,
                                                450
                                            ],
                                            "type": "text",
                                            "content": "Figure 11: Cross-modal noise case. All methods fail to retrieve the correct answer from the specified image, instead retrieving noisy textual evidence that misaligns with the structured visual content."
                                        }
                                    ]
                                }
                            ],
                            "index": 13,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 8
                },
                {
                    "type": "image",
                    "bbox": [
                        255,
                        331,
                        374,
                        369
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                255,
                                331,
                                374,
                                369
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        255,
                                        331,
                                        374,
                                        369
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                255,
                                                331,
                                                374,
                                                369
                                            ],
                                            "type": "image",
                                            "image_path": "a1eddd099be8c89a9af58a85909b9bdbe17781731b1ad431b6471b243fe79a34.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 9,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 9
                },
                {
                    "type": "image",
                    "bbox": [
                        380,
                        331,
                        497,
                        368
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                380,
                                331,
                                497,
                                368
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        380,
                                        331,
                                        497,
                                        368
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                380,
                                                331,
                                                497,
                                                368
                                            ],
                                            "type": "image",
                                            "image_path": "44bd605f5c7c20b07569ddb12093ec4efb91635f589511657f56d4777996c2de.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 10,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 10
                },
                {
                    "type": "image",
                    "bbox": [
                        256,
                        374,
                        373,
                        411
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                256,
                                374,
                                373,
                                411
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        256,
                                        374,
                                        373,
                                        411
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                256,
                                                374,
                                                373,
                                                411
                                            ],
                                            "type": "image",
                                            "image_path": "08a20d1b8a4423b84204e9918992c60e371320ee6cb4dd98129e2f96fc66ce1c.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 11,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 11
                },
                {
                    "type": "image",
                    "bbox": [
                        380,
                        373,
                        497,
                        414
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                380,
                                373,
                                497,
                                414
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        380,
                                        373,
                                        497,
                                        414
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                380,
                                                373,
                                                497,
                                                414
                                            ],
                                            "type": "image",
                                            "image_path": "1fd6ad16e77f39ee91d2b96f9a0f90f9c63a12368f3624b03db05af5900fb28a.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 12,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 12
                },
                {
                    "type": "image",
                    "bbox": [
                        106,
                        460,
                        274,
                        548
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                106,
                                460,
                                274,
                                548
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        106,
                                        460,
                                        274,
                                        548
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                106,
                                                460,
                                                274,
                                                548
                                            ],
                                            "type": "image",
                                            "image_path": "08776b90c3bf0686a0b14489df82d4a28ed1d2c697cc2fa65daf3d62b10e32b6.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 14,
                            "angle": 0,
                            "type": "image_body"
                        },
                        {
                            "bbox": [
                                104,
                                560,
                                504,
                                584
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        104,
                                        560,
                                        504,
                                        584
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                104,
                                                560,
                                                504,
                                                584
                                            ],
                                            "type": "text",
                                            "content": "Figure 12: Ambiguous table structure case. All methods fail to correctly parse the confusing table layout with merged cells and unclear column boundaries, leading to incorrect data extraction."
                                        }
                                    ]
                                }
                            ],
                            "index": 19,
                            "angle": 0,
                            "type": "image_caption"
                        }
                    ],
                    "index": 14
                },
                {
                    "type": "image",
                    "bbox": [
                        279,
                        466,
                        386,
                        498
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                279,
                                466,
                                386,
                                498
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        279,
                                        466,
                                        386,
                                        498
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                279,
                                                466,
                                                386,
                                                498
                                            ],
                                            "type": "image",
                                            "image_path": "3a2a789e40e2e34f5061cbad9d0e9d34fe10a4e8861595dbc883ceeb1d588b04.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 15,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 15
                },
                {
                    "type": "image",
                    "bbox": [
                        391,
                        466,
                        498,
                        499
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                391,
                                466,
                                498,
                                499
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        391,
                                        466,
                                        498,
                                        499
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                391,
                                                466,
                                                498,
                                                499
                                            ],
                                            "type": "image",
                                            "image_path": "e463aea5b04288ae18185bd159ad748512f9137f6d84994db730778cc5eaa5c8.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 16,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 16
                },
                {
                    "type": "image",
                    "bbox": [
                        279,
                        505,
                        384,
                        538
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                279,
                                505,
                                384,
                                538
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        279,
                                        505,
                                        384,
                                        538
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                279,
                                                505,
                                                384,
                                                538
                                            ],
                                            "type": "image",
                                            "image_path": "9426e9c961e50d8baa2f4e8a63b0f2ff4f2d082a920320a7d624a0764fa14385.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 17,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 17
                },
                {
                    "type": "image",
                    "bbox": [
                        391,
                        504,
                        500,
                        548
                    ],
                    "blocks": [
                        {
                            "bbox": [
                                391,
                                504,
                                500,
                                548
                            ],
                            "lines": [
                                {
                                    "bbox": [
                                        391,
                                        504,
                                        500,
                                        548
                                    ],
                                    "spans": [
                                        {
                                            "bbox": [
                                                391,
                                                504,
                                                500,
                                                548
                                            ],
                                            "type": "image",
                                            "image_path": "60b72bbba99fbb664cc00f82b739910746c362c9a9d7c65a7dbffe5d1a69520b.jpg"
                                        }
                                    ]
                                }
                            ],
                            "index": 18,
                            "angle": 0,
                            "type": "image_body"
                        }
                    ],
                    "index": 18
                },
                {
                    "bbox": [
                        104,
                        587,
                        504,
                        655
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                587,
                                504,
                                655
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        587,
                                        504,
                                        655
                                    ],
                                    "type": "text",
                                    "content": "Case 1: Cross-Modal Misalignment. Figure 11 presents a particularly revealing failure scenario where all evaluated methods consistently produce incorrect answers despite having access to the necessary information. This universal failure across different architectures suggests fundamental limitations in how current systems handle noisy, heterogeneous multimodal data—a critical challenge as real-world applications inevitably involve imperfect, inconsistent information sources. The failure exposes two interconnected systemic issues that compound each other:"
                                }
                            ]
                        }
                    ],
                    "index": 20
                },
                {
                    "bbox": [
                        104,
                        659,
                        504,
                        704
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                659,
                                504,
                                704
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        659,
                                        504,
                                        704
                                    ],
                                    "type": "text",
                                    "content": "Issue 1: Retrieval Bias Toward Text. Current RAG systems demonstrate pronounced bias toward textual passages. This occurs particularly when visual content lacks exact keyword matches. The bias persists even when queries contain explicit instructions to prioritize visual sources. This reveals a fundamental weakness in cross-modal attention mechanisms."
                                }
                            ]
                        }
                    ],
                    "index": 21
                },
                {
                    "bbox": [
                        104,
                        709,
                        506,
                        733
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                104,
                                709,
                                506,
                                733
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        104,
                                        709,
                                        506,
                                        733
                                    ],
                                    "type": "text",
                                    "content": "The retrieved textual information, while topically related, often operates at a different granularity level than visual content. Images may contain precise, structured data such as specific numerical values,"
                                }
                            ]
                        }
                    ],
                    "index": 22
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        337,
                        37
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                337,
                                37
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        337,
                                        37
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        299,
                        750,
                        310,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                299,
                                750,
                                310,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        299,
                                        750,
                                        310,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "17"
                                }
                            ]
                        }
                    ],
                    "index": 23
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 16
        },
        {
            "para_blocks": [
                {
                    "bbox": [
                        107,
                        82,
                        504,
                        116
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                82,
                                504,
                                116
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        82,
                                        504,
                                        116
                                    ],
                                    "type": "text",
                                    "content": "detailed diagrams, or exact spatial relationships. Corresponding text typically provides general, conceptual descriptions. This semantic misalignment introduces noise that actively misleads the reasoning process. The system attempts to reconcile incompatible levels of detail and specificity."
                                }
                            ]
                        }
                    ],
                    "index": 1
                },
                {
                    "bbox": [
                        107,
                        121,
                        504,
                        209
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                121,
                                504,
                                209
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        121,
                                        504,
                                        209
                                    ],
                                    "type": "text",
                                    "content": "Issue 2: Rigid Spatial Processing Patterns. Current visual processing models exhibit fundamental rigidity in spatial interpretation. Most systems default to sequential scanning patterns—top-to-bottom and left-to-right—that mirror natural reading conventions. While effective for simple text documents, this approach creates systematic failures with structurally complex real-world content. Many documents require non-conventional processing strategies. Tables demand column-wise interpretation, technical diagrams follow specific directional flows, and scientific figures embed critical information in unexpectedly positioned annotations. These structural variations are prevalent in professional documents, making adaptive spatial reasoning essential."
                                }
                            ]
                        }
                    ],
                    "index": 2
                },
                {
                    "bbox": [
                        107,
                        214,
                        504,
                        292
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                214,
                                504,
                                292
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        214,
                                        504,
                                        292
                                    ],
                                    "type": "text",
                                    "content": "In the observed failure case, the correct answer required integrating visual elements in reverse order from the model's default processing sequence. The system's inability to recognize and adapt to this structural requirement led to systematic misinterpretation. This represents a fundamental architectural limitation where spatial reasoning remains static regardless of document context or query intent. When spatial processing patterns are misaligned with document structure, the extracted information becomes not merely incomplete but actively misleading. This structural noise compounds other processing errors and can lead to confident but entirely incorrect conclusions."
                                }
                            ]
                        }
                    ],
                    "index": 3
                },
                {
                    "bbox": [
                        107,
                        297,
                        504,
                        396
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                297,
                                504,
                                396
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        297,
                                        504,
                                        396
                                    ],
                                    "type": "text",
                                    "content": "Case 2: Structural Noise in Ambiguous Table Layouts. As shown in Figure 12, all methods failed when confronted with a structurally ambiguous table. The primary failure stems from the table's confusing design: the GEM row lacks dedicated cell boundaries, and the \"Joint\" and \"Slot\" columns merge without clear separation. These structural irregularities create parsing ambiguities that systematically mislead extraction algorithms. This failure pattern reveals a critical vulnerability in current RAG systems. When table structures deviate from standard formatting conventions—through merged cells, unclear boundaries, or non-standard layouts—extraction methods consistently misinterpret cell relationships and conflate distinct data values. This exposes the brittleness of current approaches when faced with real-world document variations that deviate from clean, structured formats."
                                }
                            ]
                        }
                    ],
                    "index": 4
                },
                {
                    "bbox": [
                        107,
                        401,
                        504,
                        457
                    ],
                    "type": "text",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                107,
                                401,
                                504,
                                457
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        107,
                                        401,
                                        504,
                                        457
                                    ],
                                    "type": "text",
                                    "content": "The case highlights two essential directions for enhancing robustness. RAG systems require layout-aware parsing mechanisms that can recognize and adapt to structural irregularities rather than imposing rigid formatting assumptions. Additionally, integrating visual processing capabilities could significantly improve noise resilience, as visual models can leverage spatial relationships and contextual design cues that are lost in purely structural representations."
                                }
                            ]
                        }
                    ],
                    "index": 5
                }
            ],
            "discarded_blocks": [
                {
                    "bbox": [
                        106,
                        26,
                        337,
                        36
                    ],
                    "type": "header",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                106,
                                26,
                                337,
                                36
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        106,
                                        26,
                                        337,
                                        36
                                    ],
                                    "type": "text",
                                    "content": "RAG-ANYTHING: ALL-IN-ONE RAG FRAMEWORK"
                                }
                            ]
                        }
                    ],
                    "index": 0
                },
                {
                    "bbox": [
                        300,
                        750,
                        310,
                        760
                    ],
                    "type": "page_number",
                    "angle": 0,
                    "lines": [
                        {
                            "bbox": [
                                300,
                                750,
                                310,
                                760
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        300,
                                        750,
                                        310,
                                        760
                                    ],
                                    "type": "text",
                                    "content": "18"
                                }
                            ]
                        }
                    ],
                    "index": 6
                }
            ],
            "page_size": [
                612,
                792
            ],
            "page_idx": 17
        }
    ],
    "_backend": "vlm",
    "_version_name": "2.5.3"
}