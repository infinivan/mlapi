
# %%
from fastapi.params import Body
from pydantic import BaseModel
from typing import Dict, List, Optional
from spacy.language import Language
from typing import Dict, List
import uvicorn
import srsly
from starlette.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
import os
from collections import defaultdict
import spacy
import en_core_web_sm


# doc = nlp("But Virginia also tends to punish the party in the White House. In the last 12 Virginia gubernatorial elections, the president’s party has won only once, by McAulliffe in 2013")

# for ent in doc.ents:
#     print(ent.text, ent.label_)


class SpacyExtractor:
    """class SpacyExtractor encapsulates logic to pipe Records with an id and text body
    through a spacy model and return entities separated by Entity Type
    """

    def __init__(
        self, nlp: Language, input_id_col: str = "id", input_text_col: str = "text"
    ):
        """Initialize the SpacyExtractor pipeline.

        nlp (spacy.language.Language): pre-loaded spacy language model
        input_text_col (str): property on each document to run the model on
        input_id_col (str): property on each document to correlate with request
        RETURNS (EntityRecognizer): The newly constructed object.
        """
        self.nlp = nlp
        self.input_id_col = input_id_col
        self.input_text_col = input_text_col

    def _name_to_id(self, text: str):
        """Utility function to do a messy normalization of an entity name
        text (str): text to create "id" from
        """
        return "-".join([s.lower() for s in text.split()])

    def extract_entities(self, records: List[Dict[str, str]]):
        """Apply the pre-trained model to a batch of records

        records (list): The list of "document" dictionaries each with an
            `id` and `text` property

        RETURNS (list): List of responses containing the id of 
            the correlating document and a list of entities.
        """
        ids = (doc[self.input_id_col] for doc in records)
        texts = (doc[self.input_text_col] for doc in records)

        res = []

        for doc_id, spacy_doc in zip(ids, self.nlp.pipe(texts)):
            entities = {}
            for ent in spacy_doc.ents:
                ent_id = ent.kb_id
                if not ent_id:
                    ent_id = ent.ent_id
                if not ent_id:
                    ent_id = self._name_to_id(ent.text)

                if ent_id not in entities:
                    if ent.text.lower() == ent.text:
                        ent_name = ent.text.capitalize()
                    else:
                        ent_name = ent.text
                    entities[ent_id] = {
                        "name": ent_name,
                        "label": ent.label_,
                        "matches": [],
                    }
                entities[ent_id]["matches"].append(
                    {"start": ent.start_char, "end": ent.end_char, "text": ent.text}
                )

            res.append({"id": doc_id, "entities": list(entities.values())})
        return res
# %%
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# load_dotenv(find_dotenv())
# prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")


ENT_PROP_MAP = {
    "CARDINAL": "cardinals",
    "DATE": "dates",
    "EVENT": "events",
    "FAC": "facilities",
    "GPE": "gpes",
    "LANGUAGE": "languages",
    "LAW": "laws",
    "LOC": "locations",
    "MONEY": "money",
    "NORP": "norps",
    "ORDINAL": "ordinals",
    "ORG": "organizations",
    "PERCENT": "percentages",
    "PERSON": "people",
    "PRODUCT": "products",
    "QUANTITY": "quanities",
    "TIME": "times",
    "WORK_OF_ART": "worksOfArt",
}


class RecordDataRequest(BaseModel):
    text: str
    language: str = "en"


class RecordRequest(BaseModel):
    recordId: str
    data: RecordDataRequest


class RecordsRequest(BaseModel):
    values: List[RecordRequest]


class RecordDataResponse(BaseModel):
    entities: List


class Message(BaseModel):
    message: str


class RecordResponse(BaseModel):
    recordId: str
    data: RecordDataResponse
    errors: Optional[List[Message]]
    warnings: Optional[List[Message]]


class RecordsResponse(BaseModel):
    values: List[RecordResponse]


class RecordEntitiesByTypeResponse(BaseModel):
    recordId: str
    data: Dict[str, List[str]]


class RecordsEntitiesByTypeResponse(BaseModel):
    values: List[RecordEntitiesByTypeResponse]


app = FastAPI(
    # title="{{cookiecutter.project_name}}",
    # version="1.0",
    # description="{{cookiecutter.project_short_description}}",
    # openapi_prefix=prefix,
)

example_request = {
    "values": [
        {
            "recordId": "a1",
            "data": {
                "text": "But Google is starting from behind. The company made a late push into hardware, and Apple's Siri, available on iPhones, and Amazon's Alexa software, which runs on its Echo and Dot devices, have clear leads in consumer adoption.",
                "language": "en"
            }
        }
    ]
}

nlp = en_core_web_sm.load()
extractor = SpacyExtractor(nlp)


@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"/docs")


@app.post("/entities", response_model=RecordsResponse, tags=["NER"])
async def extract_entities(body: RecordsRequest = Body(..., example=example_request)):
    """Extract Named Entities from a batch of Records."""

    res = []
    documents = []

    for val in body.values:
        documents.append({"id": val.recordId, "text": val.data.text})

    entities_res = extractor.extract_entities(documents)

    res = [
        {"recordId": er["id"], "data": {"entities": er["entities"]}}
        for er in entities_res
    ]

    return {"values": res}


@app.post(
    "/entities_by_type", response_model=RecordsEntitiesByTypeResponse, tags=["NER"]
)
async def extract_entities_by_type(body: RecordsRequest = Body(..., example=example_request)):
    """Extract Named Entities from a batch of Records separated by entity label.
        This route can be used directly as a Cognitive Skill in Azure Search
        For Documentation on integration with Azure Search, see here:
        https://docs.microsoft.com/en-us/azure/search/cognitive-search-custom-skill-interface"""

    res = []
    documents = []

    for val in body.values:
        documents.append({"id": val.recordId, "text": val.data.text})

    entities_res = extractor.extract_entities(documents)
    res = []

    for er in entities_res:
        groupby = defaultdict(list)
        for ent in er["entities"]:
            ent_prop = ENT_PROP_MAP[ent["label"]]
            groupby[ent_prop].append(ent["name"])
        record = {"recordId": er["id"], "data": groupby}
        res.append(record)

    return {"values": res}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5002,
                log_level="info", reload=True)
