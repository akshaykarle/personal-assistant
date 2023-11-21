import logging
import json
import os
import pprint
from haystack import document_stores
from haystack.nodes import retriever
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack import Pipeline
from haystack.nodes import TextConverter, FileTypeClassifier, MarkdownConverter, DocxToTextConverter, PreProcessor
from haystack.nodes.retriever import EmbeddingRetriever
from haystack.nodes.reader import FARMReader
from haystack.utils import print_answers


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


document_store = InMemoryDocumentStore()
document_store.delete_documents()

document_index = "document"

# get all files including those in subdirs
def get_all_files_by_type(root_dir):
    files_by_type = {}
    for path, subdirs, filenames in os.walk(root_dir):
        for filename in filenames:
            extension = filename.split(".")[-1]
            if extension in ["txt", "md", "docx"]:
                if extension not in files_by_type: files_by_type[extension] = []
                files_by_type[extension].append(os.path.join(path, filename))
    return files_by_type

data_dir = "../../data"
files_to_index = get_all_files_by_type(data_dir)
pprint.pprint(len(files_to_index))


file_type_classifier = FileTypeClassifier(supported_types=['txt', 'md', 'docx'])

text_converter = TextConverter()
md_converter = MarkdownConverter()
docx_converter = DocxToTextConverter()
preprocessor = PreProcessor()

indexing_pipeline = Pipeline()

indexing_pipeline.add_node(component=file_type_classifier, name="FileTypeClassifier", inputs=["File"])

indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["FileTypeClassifier.output_1"])
indexing_pipeline.add_node(component=md_converter, name="MarkdownConverter", inputs=["FileTypeClassifier.output_2"])
indexing_pipeline.add_node(component=docx_converter, name="DocxConverter", inputs=["FileTypeClassifier.output_3"])

indexing_pipeline.add_node(
    component=preprocessor,
    name="Preprocessor",
    inputs=[
        "TextConverter",
        "MarkdownConverter",
        "DocxConverter",
            ],
)

indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Preprocessor"])

for file_paths in files_to_index.values():
    indexing_pipeline.run_batch(file_paths=file_paths)



retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/all-mpnet-base-v2-table")
document_store.update_embeddings(retriever, update_existing_embeddings=True)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

qa_pipeline = Pipeline()
qa_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
qa_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])


query = "What was the interview process with Deliveroo?"
result = qa_pipeline.run(query, params={"Retriever": {"top_k": 2}})
print_answers(results=result, details="minimum")
