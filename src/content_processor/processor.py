

from dataclasses import dataclass

from src.document_parsing.data_extraction import MinerU_Parser
from src.document_parsing.sample_data import combined_knowledge_units, current_multimodel_unit
from src.content_processor.prompt import TABLE_CONTENT_WITH_CONTEXT_PROMPT

from perplexity import Perplexity
import perplexity
import os
import base64
from dotenv import load_dotenv
load_dotenv()

perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")


"""
Utils Functions:

1- Knowledge units splitter
2- Context extractor for multi-modal content

"""
                        ##  1-  Knowledge units splitter  ##

# Define the function to split the knowledge units into textual and non-textual units

def units_splitter(knowledge_units_list:list):
    """
    This function takes the list of the knowledge units created in the parsing process from minerU output, and 
    filter out the textual and non-textual knowledge units on the basis of their content type, into two different
    objects. 

    **Args:**
    knowledge_units_list (list): It is the list of the knowledge units - combined textual and non-textual units.

    **Returns:**
    textual_knowledge_units (list): It is the list of the textual knowledge units.
    multi-model_knowledge_units (list): It is the list of the non-textual knowledge units.

    **Raises:**

    Implementation workflow:
    
    1- Initiate the two lists to store respective type of units separately.
    2- Iterate over the units using for loop 
    3- If content type is in ["title","text"]: append the list for textual knowledge units
    4- If content type == "table": append the list for non-textual knowledge units
    5- Return the textual_knowledge_units, non_textual_knowledge_units
    
    """

    # Initialize the minerU parser
    #### FREEZED FOR TESTING PURPOSE ####
    #init_minerU = MinerU_Parser(data_file_path=knowledge_units_list)
    #knowledge_units_list = init_minerU.format_minerU_output()

    complete_knowledge_units = knowledge_units_list

    # Initialize the lists for textual and non-textual units separately
    multi_model_units = []
    textual_units = []

    for unit in complete_knowledge_units:
        unit = dict(unit)

        # Fetch the textual units
        content_type = unit.get("content_type")
        if content_type in ["text","title"]:
            textual_units.append(unit)
        
        # Fetch the non-textual units
        elif content_type == "table":
            multi_model_units.append(unit)

    return multi_model_units,textual_units





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

      def __init__(self,context_chunks_text,content_of_current_chunk):
          
          self.context_chunks_text = context_chunks_text
          self.content_of_current_chunk = content_of_current_chunk
          
        
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
                model= "sonar"

            )

            return llm_content_description.choices[0].message.content
        except perplexity.BadRequestError as e:
            print(f"Invalid request parameters: {e}")
        except perplexity.RateLimitError as e:
            print("Rate limit exceeded, please retry later")
        except perplexity.APIStatusError as e:
            print(f"API error: {e.status_code}")




if __name__ == "__main__":

    multi_model_knowledge_units, textual_knowledge_units = units_splitter(knowledge_units_list=combined_knowledge_units)

    extractor = Context_Extractor(all_knowledge_units=combined_knowledge_units)
    address_of_table, context_chunks_text = extractor.multi_model_extractor(current_multi_model_unit=current_multimodel_unit)

    print(address_of_table)
    print(context_chunks_text)
    
    run_processor = ContentProcessor(context_chunks_text=context_chunks_text,
                                    content_of_current_chunk = address_of_table)
    llm_response = run_processor.Information_generation_processor()
    print(llm_response)
