

from dataclasses import dataclass

from src.document_parsing.data_extraction import MinerU_Parser
from src.document_parsing.sample_data import combined_knowledge_units, current_multimodel_unit
from src.content_processor.prompt import TABLE_CONTENT_WITH_CONTEXT_PROMPT
from src.content_processor.schemas import table_description_schema
from utils import doc_id, units_splitter, document_title, Milvus_client

from perplexity import Perplexity
from pymilvus import MilvusClient, DataType
import perplexity
import os
import base64
from dotenv import load_dotenv
load_dotenv()

perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

# Here is the document ID
doc_id_generated = doc_id()
doc_title = document_title()


"""
2- Context extractor for multi-modal content

"""



                        ####  2-  Context Extractor  ####


class Context_Extractor():
    """
    It contains two components required for driving the context around the multi-model context. 
    1) Context Extractor 
    2) multi-model Prcoessor

    1) Context extractor takes the current chunk (multi-model chunk), fetches the text from the surrounding of the current chunk.

    2) multi-model Processor: It takes the image of multi-model content, and text fetched by the context extractor & get the 
        description of the multi-model content & details of entity name & other details required to store it in Knowledge Graph.

    """
    def __init__(self,all_knowledge_units):

        self.all_knowledge_units:list[dict[str]] = combined_knowledge_units

    
    def multi_model_extractor(self,current_multi_model_unit:list[dict[str]]):
        """
        It takes the current unit, and fetches its placement details to identify the surrouding chunks in the documents and then 
        place them in the chunks for context extraction list.

        Here is the workflow:
        1- Find out the page of current chunk. Using that, find out previous page and next page. (Done)
        2- Access all chunks of current page, extract their index numbers and store in a list in their hierarchical order. (Done)
        3- Access all chunks of the next page, and of the previous page. Extract their index numbers and store separately. (Done)
        4- Put them together in a single list (Done)
        5- Fetch the next two & previous two chunks of the current chunk from this list.
            (a) loop over the list to find out the current chunk (Done)
            (b) Findout the index of the current chunk in the list (Done)
            (c) Fetch the previous two chunks and next two chunks from the units (Done)
            (d) Fetch text from the shortlisted surrounding chunks using chunk-context-window
        
        Detect source type --> Source Handler --> Windowing --> Extract --> Truncate
                
        """

        # Input variables
        all_knowledge_units = self.all_knowledge_units
        current_item = current_multimodel_unit

        # As it is list of chunk because of figure & caption unit so, we need to find out only figure unit as reference for placement of current chunk
        unit_of_figure = None
        # Placement details of the current chunk
        for unit in current_item:
            if "table_image_path" in unit:
                page_of_current_unit = unit.get("page_no.","")
                page_index_of_current_unit = unit.get("index_on_page","")
                content_of_current_unit = unit.get("table_image_path","")
                unit_of_figure = unit
        
        surrounding_pages_units = []


        # Previous page & Next page 
        previous_page = page_of_current_unit - 1
        next_page = page_of_current_unit + 1
        pages_relvant_for_context = [previous_page,page_of_current_unit,next_page]
        # Fetch all the chunks from previous page, current page, and next page (In this hierarchical order)
        for page in pages_relvant_for_context:
            for unit in combined_knowledge_units:
                if unit.get("page_no.") == page:
                    surrounding_pages_units.append(unit)

        # Fetch the index of the current chunk in the list of surrounding chunks
        chunk_window = 2
        index_of_current_unit = surrounding_pages_units.index(unit_of_figure)
        start_index = max(0,index_of_current_unit - chunk_window)
        end_index = min(len(surrounding_pages_units), index_of_current_unit + chunk_window + 1)
        
        # Fetch previous two chunks
        range_of_surrounding_chunks = list(range(start_index, end_index))
        list_of_context_chunks = [surrounding_pages_units[i] for i in range_of_surrounding_chunks if i != index_of_current_unit]


        """
        Imp Note: For multi-model content context extraction, we do not need to store previous & next chunks separately. It is because
        placement of image does not break the continuity of the text chunks, it just enhances the semantic meaning of it. 
        """
        # Fetch the content out of chunks
        context_chunks_text = []
        for lcc in list_of_context_chunks:
            if "raw_content" in lcc:
                context_chunks_text.append(lcc.get("raw_content",""))
            if "table_caption" in lcc:
                context_chunks_text.append(lcc.get("table_caption",""))

        return content_of_current_unit,context_chunks_text


class ContentProcessor():
      
      """
        It is the processor that takes the contextual text from the surroudning chunks & address of the image of the multi-model chunk,
        calls the LLM to give the summary of the content and also entity description which will include following things:
        1- Name of entity
        2- Entity type
        3- Description for Graph DB 
      """

      def __init__(self,context_chunks_text,content_of_current_chunk,table_content_schema):
          
          self.context_chunks_text = context_chunks_text
          self.content_of_current_chunk = content_of_current_chunk
          self.table_content_schema = table_content_schema
          
        
      def Information_generation_processor(self):
          
        """
        **Args:**
        context_chunks_text (list[str]): It is the text of the surrouding context, that gives the context of the document narrative around
                                         the multi-model content.
        content_of_current_chunk (str): It is the local address of the image of multi-model content.
        
        **Returns:**
        Content_description (list[str]): It is the detailed description of the multi-model content
        Entity_summary (dict): It contains the information which we need to include table-description as node in knowledge Graph. It contains
                                entity name, entity type, entity description for knowledge Graph.
        
        """
        

        # Get the input variables
        address_of_content = self.content_of_current_chunk
        contextual_text = self.context_chunks_text
        table_output_schema = self.table_content_schema

        # lets convert the address in base64 mode
        try:
            with open(address_of_content,"rb") as file_path:
                base_format_address = base64.b64encode(file_path.read()).decode("utf-8")
                file_uri = f"data:image/jpg;base64,{base_format_address}"
        except FileNotFoundError:
            print("Error: Image file not found.")
            exit()

        """
        1- Ensure limit runs with maximum retries
        2- Handle errors
        3- Control creativity
        """
        
        # Get the prompt for the LLM
        prompt = TABLE_CONTENT_WITH_CONTEXT_PROMPT.format(
            address_of_content = base_format_address,
            contextual_text = contextual_text
        )

        # Initialize the client for perplexity
        client = Perplexity(api_key=perplexity_api_key, 
                            max_retries=1, 
                            )
        try:
            # Generate perplexity response
            llm_content_description = client.chat.completions.create(
                messages= 
                        [
                            {
                            "role": "user", 
                            "content" :  [ 
                                            {
                                            "type" : "text",
                                            "text": prompt
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": file_uri
                                                }
                                            }  
                                        ]
                            }
                        ],
                model= "sonar",
                response_format= {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": table_output_schema.model_json_schema()
                    }

                }

            )

            # convert the json into dict to reuse its fields
            llm_structured_output = table_output_schema.model_validate_json(llm_content_description.choices[0].message.content)
            llm_structured_output = llm_structured_output.model_dump()
            table_description = llm_structured_output.get("content_description","")
            entity_summary = llm_structured_output.get("entity_summary","")

            return table_description, entity_summary, llm_structured_output
        except perplexity.BadRequestError as e:
            print(f"Invalid request parameters: {e}")
        except perplexity.RateLimitError as e:
            print("Rate limit exceeded, please retry later")
        except perplexity.APIStatusError as e:
            print(f"API error: {e.status_code}")




class processor_storage():
    """
    This class handles the tasks related to the storage of the data in vector & Graph databases.
    
    Below is the blueprint of the class:

    Config class: (Check if required or not.)   ---> Can only confirm after initial implementation. If we come across storing configurable parameters then we can create a configdata class for them.
    Main Class: (Finalize the class instances and object instances) ---> 
    Input handler-1: (Finalize the format that we want to pass onto the formatter-1)
    Input handler-2: (Finalize the format that we want to pass onto the formatter-2)
    Formatter for IH-1: (Finalize the final format required to pass onto the textual Vector database)
    Format for IH-2: (Finalize the final format required to pass onto the multi-mode Vector database)
    Processor for storing data: (Finalize the process the we need to implement for storing data in Vector DB)
    Output confirmation: (Finalize the confirmation required for the successful storage of the data)

    """

    # Document ID will remain same across the chunks.
    document_id = doc_id_generated
    docum_title = doc_title 


    def __init__(self):
        pass


    def textual_chunk_payload_prep(self,current_item,current_item_number):

        """
        ##  Textual Units Handler   ##

        **Args:**
        Textual_knowledge_units (list[dict]): It is the list of dictionaries that contains textual chunk & placement information.
        
        **Returns:**
        1)      UUID: It is the ID that we generate universally using UUID, and assign it to the document chunks as Doc ID.
        2)      Chunk-ID: It is the ID that will be designated to the Chunk to be stored in VDB.
        3)      Prepare the Package of the Textual Unit item that will be pushed to the VDB.

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

        self.current_item = current_item
        return self.current_item
    
    def textual_chunks_vector_payload(self):
        """
        It takes the data from the current item and prepares the payload for textual chunks by including the
        vector representation of data in it.

        **Args:**
        current_item (dict): It contains the datapoints that are useful for building the payload.

        **Returns:**
        chunk_payload (dict): It is the payload of the chunk that includes vector representation and 
                              will be pushed to the vector database.

        """

        # Get the current item
        current_item = self.current_item

        # Initialize milvus client
        Client = Milvus_client()
        # Initialize the schema creation 
        text_vectors_schema = Client.create_schema(
                                                    auto_id = False,
                                                    )
        
        # Fields of schema
        text_vectors_schema.add_field(
            field_name= "doc_id",
            datatype= DataType.VARCHAR,
            is_primary = True,
            max_length = 50
        )
        text_vectors_schema.add_field(
            field_name= "chunk_id",
            datatype=DataType.VARCHAR,
            max_length = 50
        )
        text_vectors_schema.add_field(
            field_name= "document_title",
            datatype= DataType.VARCHAR,
            max_length = 500
        )
        text_vectors_schema.add_field(
            field_name= "page_no",
            datatype= DataType.INT64
        )
        text_vectors_schema.add_field(
            field_name= "index_on_page",
            datatype= DataType.INT64
        )
        text_vectors_schema.add_field(
            field_name= "Vectors",
            datatype= DataType.FLOAT_VECTOR, dim = 768
        )

        # Indexing for vectors field is must - in order to perform vector search. Currently, we will stick with the
        vector_index = Client.prepare_index_params()
        vector_index.add_index(
            field_name="Vectors",
            index_name= "dense_vectors_index"
            index_type="AUTOINDEX",
            metric_type = "COSINE"
        )



        return Client





        
















if __name__ == "__main__":

    multi_model_knowledge_units, textual_knowledge_units = units_splitter(knowledge_units_list=combined_knowledge_units)

    #extractor = Context_Extractor(all_knowledge_units=combined_knowledge_units)
    #address_of_table, context_chunks_text = extractor.multi_model_extractor(current_multi_model_unit=current_multimodel_unit)
    
    #run_processor = ContentProcessor(context_chunks_text=context_chunks_text,
    #                                content_of_current_chunk = address_of_table,
    #                                table_content_schema= table_description_schema)
    #llm_response = run_processor.Information_generation_processor()
    
    db_storage = processor_storage()


    current_item_number = 0
    for current_item in [textual_knowledge_units[0]]:
        current_item_number += 1
        new = db_storage.textual_chunk_payload_prep(current_item, current_item_number)
        new_1 = db_storage.textual_chunks_vector_payload()
    print(new_1)





        
