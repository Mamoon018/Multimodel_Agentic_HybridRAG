
# Schema for the Perplexity structured output
from pydantic import BaseModel, Field


class table_entity_summary_datapoints(BaseModel):
    """
    It contains the datapoints that needs to be collected by the LLM for compiling the entity summary
    """

    entity_name: str = Field(description="It is the name of the entity for which llm is generating description, it is the name of the table.")
    entity_type: str = Field(description="It is the type of the entity which is table")
    related_entities: str = Field(description="It is related data elements that are being discussed in the table, and relationship between them")
    entity_summary: str = Field(description= "concise summary of table's purpose and key insights (max 100 words)")

class table_description_schema(BaseModel):
    """
    It defines the fields and datatypes that we want to get in the output of the LLM.
    """

    content_description: str = Field(description= "It contains the description of the table that needs to be interpreted.")

    entity_summary: list[table_entity_summary_datapoints] = Field(description= "It contains the datapoints related to the entity summary")

