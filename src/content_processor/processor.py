

from dataclasses import dataclass
import numpy as np 
from src.document_parsing.sample_data import sample_multi_model_chunks_with_llm_description, sample_textual_knowledge_units
from utils import doc_id, document_title, Milvus_client, openai_embeddings, num_tokens_from_string
from src.document_parsing.sample_data import list_of_vector_embeddings
from dataclasses import dataclass
from pymilvus import DataType
from typing import Any
import os
from dotenv import load_dotenv
load_dotenv()

perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Here is the document ID
doc_id_generated = doc_id()
doc_title = document_title()

    # Lets define the dataclass to pass the parameters to processor_storage class
@dataclass
class Config:
        
        """
        This class contains the parameters that can be very likely change to test program with different
        parameters.
        This class requires to keep the flexibility with creation of collections so, we decided to provide
        that through config_class rather than hard-coding everytime we need to make any changes.
 
        """ 
        textual_collection_VDB: str = "Textual_VDB_collection"
        multi_model_collection_VDB: str = "Multi_modal_VDB_collection"
        token_limit: int = 5000
 
 
class processor_storage():
    """
    This class handles the tasks related to the storage of the data in vector & Graph database.

    """


    def __init__(self, multi_model_chunks_with_llm_description, textual_knowledge_units, config:Config = None):

        self.document_id = doc_id_generated
        self.docum_title = doc_title
        self.textual_units = textual_knowledge_units
        self.multi_model_units = multi_model_chunks_with_llm_description
        self.config = config or Config()
        self.client = None
        self.embedding_function = None


    def __run_processor_storage__(self):
         
        """
        It orchestrate the implementation of the processor storage class, starts from generating the payload chunks,
        ends with push data to vector database.           
        """


        lists_of_units = [self.textual_units, self.multi_model_units]
        text_payload_insertion_list = []
        multi_model_payload_insertion_list = []

        current_item_no_textual = 0
        current_item_no_multimodal = 0
        list_no = 0

        for list_of_units in lists_of_units:
            list_no += 1
            for unit in list_of_units:
                  
                if list_no == 1: 
                    current_item_no_textual += 1
                    text_chunk_insertion_data = self.chunk_payload_prep(
                        current_item=unit, 
                        current_item_number=current_item_no_textual
                    )
                    text_payload_insertion_list.append(text_chunk_insertion_data)
                elif list_no == 2:
                    current_item_no_multimodal += 1
                    multimodal_chunk_insertion_data = self.chunk_payload_prep(
                        current_item= unit,
                        current_item_number=current_item_no_multimodal
                    )
                    multi_model_payload_insertion_list.append(multimodal_chunk_insertion_data)
        

        # Add the vectors to the payload
        textual_vectorized_payload_insertion_list = self.generate_embeddings_for_payload_text(payloads_list=text_payload_insertion_list)
        multi_modal_vectorized_payload_insertion_list = self.generate_embeddings_for_payload_text(payloads_list=multi_model_payload_insertion_list) 

        vectorized_payloads = [textual_vectorized_payload_insertion_list, multi_modal_vectorized_payload_insertion_list]

        """
        1) Provide the list of collections TEXT and MULTI-MODEL collections that we want to generate. (in-main)
        2) Pass the payloads to the method to insert the data in the respective collections. (in-class method)

        Design choice: It will allow us to create new collections without hard-coding the inner method. 
        """

        
        collections_to_be_generated = [self.config.textual_collection_VDB, self.config.multi_model_collection_VDB]
        for collection in collections_to_be_generated:
            coll_signal, coll_list = self.create_VDB_collection(content_collection_name=collection)
        
        for payload, collection_created in zip(vectorized_payloads, collections_to_be_generated):
            collection_data_view = self.VDB_data_insertion_task(payloads=payload,collection=collection_created)
        
                
        return    collection_data_view
        


    def chunk_payload_prep(self,current_item,current_item_number):

        """
        ##  Prepares the payload which includes the content & metadata that needs to be pushed to Milvus database.   ##
        It takes the current_unit, and generates the document-id and chunk-id for it, and insert both in that along
        with the document title in order to prepare final payload of the textual units that can be stored in the
        Vector database. (Embeddings for the chunk will be generated separately)

        **Args:**
        Textual_knowledge_units (list[dict]): It is the list of dictionaries that contains textual chunk & placement information.
        
        **Returns:**
        3)      Prepare the payload of the Textual Unit item that will be pushed to the VDB after embeddings generation of its text.

        """


        # Get a document id
        doc_id = self.document_id
        # Get a doc title
        doc_title = self.docum_title

        # Get a chunk id
        current_item_number = str(current_item_number)
        chunk_id = doc_id[0:8] + "-" + "chunk-" + current_item_number

        # Add doc_id, chunk_id in the item to prepare it for the chunk-payload
        current_item["doc_id"] = doc_id
        current_item["chunk_id"] = chunk_id
        current_item["document_title"] = doc_title

        # Get the JSON of Metadata
        metadata = {
            "page_no.": current_item["page_no."],
            "index_on_page": current_item["index_on_page"],
            "content_type": current_item["content_type"],
            "document_title": current_item["document_title"]
        }

        # Using metadata and current item fields, we need to create insertion payload that aligns with schema
        self.chunk_insertion_data = {
            "doc_id": current_item["doc_id"],
            "chunk_id": current_item["chunk_id"],
            "raw_content": current_item["raw_content"],
            "metadata": metadata
        }

        return self.chunk_insertion_data
    
    def generate_embeddings_for_payload_text(self,payloads_list:list[dict]):

        """
        It calls the openai pre-trained model to generate the vector embeddings for the payload texts. It 
        generates the high-dimensionality embeddings with dim of 1536.
        It takes the text from the payload one by one, generate embeddings and then add vector embeddings back
        to the payloads respectively.

        **Args:**
        payload_insertion_list (list[dict]): It is the list of payloads.

        **Returns:**
        vectorized_payloads_insertion_list  (list[dict]): It is the list of text payloads containing vector embeddings of content.
        """
        # lets get the embedding model object
        #openai_embedding_model = openai_embeddings()
        
        vectorized_payloads_insertion_list = []

        for payload in payloads_list:
            content = payload.get("raw_content","")

            # lets get the token counts for the text
            tokens_count = num_tokens_from_string(string=content, encoding_name="cl100k_base")
            """
            Check if chunk is exceeding the token limit or not.
            1- Check the number of tokens in the chunk.
            2- If less than limit then no need of processing.
            3- If greater than limit, then reduce the size of the chunk.
            CAN OPTIMIZE TI BY APPLYING SENTENCE LEVEL TOKEN LIMITING.
            """
                        # FOR TESTING PURPOSE - USING SAMPLE VECTORS 
            float_vectors_list = list_of_vector_embeddings
            #float_vectors = openai_embedding_model.encode_documents(content)
            #float_vectors_list = np.array(float_vectors[0], dtype=float).tolist()
            

            if tokens_count <= self.config.token_limit:
                payload["Vectors"] = float_vectors_list            
            else:
                reduced_tokens = float_vectors_list[:tokens_count]
                payload["Vectors"] = reduced_tokens
            
            vectorized_payloads_insertion_list.append(payload)

        return vectorized_payloads_insertion_list


    def create_VDB_collection(self,content_collection_name:str):
        """
        It initializes the milvus client and defines the schema of the collection and load the collection using that 
        schema, also defines the indexing strategy for chunks vectors.

        1) It initializes the Milvus Client 
        2) It defines the Schema of the "Milvus Collection"
        3) Index parameter for defining indexing type & metric type for Vector field

        **Returns:** (str): It returns the confirmation abuot the collection loading 

        """

        # Initialize milvus client
        self.client = Milvus_client()
        # Initialize the schema creation 
        text_vectors_schema = self.client.create_schema(
                                                    auto_id = False,
                                                    )                          

        # Fields of schema
        text_vectors_schema.add_field(
            field_name= "doc_id",
            datatype=DataType.VARCHAR,
            max_length = 50
        )
        text_vectors_schema.add_field(
            field_name= "chunk_id",
            datatype= DataType.VARCHAR,
            is_primary = True,
            max_length = 50
        )
        text_vectors_schema.add_field(
            field_name= "raw_content",
            datatype= DataType.VARCHAR,
            max_length = 50000
        )
        text_vectors_schema.add_field(
            field_name= "metadata",
            datatype= DataType.JSON
        )
        text_vectors_schema.add_field(
            field_name= "Vectors",
            datatype= DataType.FLOAT_VECTOR, dim = 1536
        )
        
        """
        For prioritising the high recall & high QPS, we will stick with HNSW approach which uses graph to map the 
        data, params (M, ef) can be tuned based on the outcome quality.
        """
        


        # Indexing for vectors field is must - in order to perform vector search. Currently, we will stick with the
        vector_index_params = self.client.prepare_index_params()
        vector_index_params.add_index(
            field_name="Vectors",
            index_name= "dense_vectors_index",
            # As TopK in our case gonna be low, we will choose Graph based approach (HNSW)
            index_type="HNSW",
            metric_type = "COSINE",
            params = {
                "M": 20,
                "efConstruction": 35
            }
        )

        
        # Let's create/Load the collection
        collection_signal = None
        list_of_collections = self.client.list_collections()

        if content_collection_name not in list_of_collections:

                # Let's create the collection which we will use to store the data
            self.client.create_collection(
                collection_name=content_collection_name,
                schema= text_vectors_schema,
                index_params= vector_index_params)
            collection_signal = "Collection created"
        

        else:
            # confirmation of collection creation
            self.client.load_collection(
                collection_name=content_collection_name
                )
            collection_signal = "Collection loaded!"
        
        
        list_of_collections = self.client.list_collections()

        """
        Client.drop_collection(
            collection_name= "Textual_collection_2"
        )
        """

        return collection_signal, list_of_collections
    
    def VDB_data_insertion_task(self, payloads: Any = None, collection: Any = None):
        """
        This function takes the textual and multi-modal payloads and simply insert them into collections. 
        """


        collection_data = self.client.insert(
            collection_name= collection,
            data= payloads
        )

        return collection_data






if __name__ == "__main__":

    db_storage = processor_storage(
        multi_model_chunks_with_llm_description=sample_multi_model_chunks_with_llm_description,
        textual_knowledge_units= sample_textual_knowledge_units)

    results = db_storage.__run_processor_storage__()

    print(results)


    



        
