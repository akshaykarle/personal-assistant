import os
import pprint
from unstructured.partition.auto import partition
from unstructured.documents import elements
from unstructured.staging.base import convert_to_dict

from database import Database

db = Database()
# splits = partition(filename="../../data/notion_db/Task List e18d5b59652e44f6a02ef3cadb322b85/9 May 2022 fade00f99daa416282357d4867b78c41.md")

splits = partition(filename="../../data/notion_db/Task List e18d5b59652e44f6a02ef3cadb322b85/Springer Nature - 4 July 2023 eab44115e8784e068ca75e4f5d01f21c.md")

for s in splits:
    metadata = s.metadata.to_dict()
    metadata.pop('languages')
    db.store_document(s.text, metadata, s.id)


pprint.pprint(db.retrieve_documents("What did I do on 4th July 2023?"))
