from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    GPTSQLStructStoreIndex,
    SQLDatabase,
    WikipediaReader
)
# define pinecone index
import pinecone
import os

api_key = os.environ['PINECONE_API_KEY']
pinecone.init(api_key=api_key, environment="us-east1-gcp")

# dimensions are for text-embedding-ada-002
# pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index("quickstart")

# OPTIONAL: delete all
# pinecone_index.delete(deleteAll=True)

from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import ServiceContext, LLMPredictor
from llama_index.storage import StorageContext
from llama_index.vector_stores import PineconeVectorStore
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI

# define node parser and LLM
chunk_size = 1024
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True))
service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm_predictor=llm_predictor)
text_splitter = TokenTextSplitter(chunk_size=chunk_size)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

# define pinecone vector index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='wiki_cities')
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = GPTVectorStoreIndex([], storage_context=storage_context)

# create database schema and test data
# here we introduce a toy scenario where there are 100 tables (too big to fit into the prompt)

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column

engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)

# print tables
print(metadata_obj.tables.keys())

# we introduce some test data into the city_stats table

from sqlalchemy import insert
rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
        connection.commit()
with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())

# Load Data
# We first show how to convert a Document into a set of nodes, and insert into a DocumentStore

cities = ['Toronto', 'Berlin', 'Tokyo']
wiki_docs = WikipediaReader().load_data(pages=cities)

# Build SQL Index
sql_database = SQLDatabase(engine, include_tables=["city_stats"])

sql_index = GPTSQLStructStoreIndex.from_documents(
    [],
    sql_database=sql_database,
    table_name="city_stats",
)

from llama_index.indices.struct_store.sql_query import BaseSQLTableQueryEngine
sql_query_engine = BaseSQLTableQueryEngine([], sql_database, table_name="city_stats")



# Build Vector Index

# Insert documents into vector index
# Each document has metadata of the city attached
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = node_parser.get_nodes_from_documents([wiki_doc])
    # add metadata to each node
    for node in nodes:
        node.metadata = {"title": city}
    vector_index.insert_nodes(nodes)

# Define Query Engines, Set as Tools

from llama_index.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.vector_store import VectorIndexAutoRetriever
from llama_index.indices.struct_store.sql_query import (
    BaseSQLTableQueryEngine,
)

# sql_query_engine = sql_index.as_query_engine(synthesize_response=True)

from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

vector_store_info = VectorStoreInfo(
    content_info='articles about different cities',
    metadata_info=[
        MetadataInfo(
            name='title',
            type='str',
            description='The name of the city',
        )
    ]
)
vector_auto_retriever = VectorIndexAutoRetriever(vector_index, vector_store_info=vector_store_info)

retriever_query_engine = RetrieverQueryEngine.from_args(
    vector_auto_retriever, service_context=service_context
)

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        'Useful for translating a natural language query into a SQL query over a table containing: '
        'city_stats, containing the population/country of each city'
    )
)

print(type(sql_tool.query_engine))

vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=f'Useful for answering semantic questions about different cities',
)

# Define SQLAutoVectorQueryEngine

query_engine = SQLAutoVectorQueryEngine(
    sql_tool,
    vector_tool,
    service_context=service_context
)

response = query_engine.query('Tell me about the arts and culture of the city with the highest population')
print(response)