
import uuid
from src.document_parsing.sample_data import combined_knowledge_units
from pymilvus import MilvusClient
import os


milvus_api_key = os.getenv("MILVUS_API_KEY")




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





## Function to create the Doc ID
def doc_id():

    """
    It simply generates the document ID that will remain same across all the chunks.
    """

    document_id = uuid.uuid4()
    document_id = str(document_id) # It is the DB friendly format.

    return document_id

## Function to fetch the document title (Report type)
def document_title():
    """
    It fetches the title of the document that will be used as metadata in the chunk payload of the document
    """

    multi_model_knowledge_units, textual_knowledge_units = units_splitter(knowledge_units_list=combined_knowledge_units)

    title_text_chunk = textual_knowledge_units[0]
    doc_title = title_text_chunk["raw_content"] 


    return doc_title



## Initialize the Milvus Client

def Milvus_client():
    """
    It initializes the connection with the Milvus server using the API
    """
    Client = MilvusClient(
                            uri= "https://in03-83d8e6e72c3248f.serverless.aws-eu-central-1.cloud.zilliz.com",
                            token= milvus_api_key
                        )

    return Client

if __name__ == "__main__":

    Milvus_client()

