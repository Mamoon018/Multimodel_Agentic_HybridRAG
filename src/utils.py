

from dataclasses import dataclass

from src.document_parsing.data_extraction import MinerU_Parser
from src.document_parsing.sample_data import combined_knowledge_units, current_multimodel_unit



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


class context_extractor():
    """
    It contains two components required for driving the context around the multi-model context. 
    1) Context Extractor 
    2) Image Prcoessor

    1) Context extractor takes the current chunk (multi-model chunk), fetches the text from the surrounding of the current chunk.

    2) Image Processor: It takes the image of multi-model content, and text fetched by the context extractor & get the 
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

        return print(content_of_current_unit)




if __name__ == "__main__":

    multi_model_knowledge_units, textual_knowledge_units = units_splitter(knowledge_units_list=combined_knowledge_units)

    extractor = context_extractor(all_knowledge_units=combined_knowledge_units)
    extractor.multi_model_extractor(current_multi_model_unit=current_multimodel_unit)

