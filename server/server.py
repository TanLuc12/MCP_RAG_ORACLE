import oracledb
import os
import json
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

ORACLE_HOST = os.getenv("ORACLE_HOST")
ORACLE_PORT = os.getenv("ORACLE_PORT")
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE")
ORACLE_USERNAME = os.getenv("ORACLE_USERNAME")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
api_key = os.getenv("OPEN_API_KEY")
# print(ORACLE_HOST)
dsn = f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"


class OracleVectorSearch:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key= api_key)

        self.vector_store = OracleVS(
                client=oracledb.connect(
                    user=ORACLE_USERNAME,
                    password=ORACLE_PASSWORD,
                    dsn=f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"
                ),
                embedding_function=self.embeddings,
                table_name="N8N_VECTORS",
                distance_strategy="COSINE"
            )
            
    def vector_search(self, prompt: str) -> str:
        """Search for the most similar documents in Oracle database using vector search.
        
        Args:
            prompt: The prompt to search for related documents using similarity search
            
        Returns:
            JSON string of search results with content, metadata, and similarity scores
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=prompt,
                k=10 
            )
            
            if not results:
                return json.dumps({
                    "message": "No results found for the given prompt.",
                    "results": []
                })

            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    "name": doc.metadata.get("name", ""),
                    "description": doc.metadata.get("description", doc.page_content),
                    "score": float(score)
                }
                formatted_results.append(result)
                
            # Trả về JSON string thay vì list
            return json.dumps({
                "message": f"Found {len(formatted_results)} results",
                "results": formatted_results
            },ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "error": f"Search failed: {str(e)}",
                "results": []
            })


oracle_search = OracleVectorSearch()

# Initialize MCP server
mcp = FastMCP("mcp-oracle-vector-search")

@mcp.tool()
async def vector_search_oracle(prompt: str) -> str:
    """Search for the most similar nodes in the Oracle database using vector search.
    
    Args:
        prompt: The prompt to search for related nodes using similarity search
        
    Returns:
        JSON string of search results with name, description, and score
    """
    return oracle_search.vector_search(prompt)

def main() -> None:
    try:
        oracledb.init_oracle_client()
        mcp.run(transport="stdio")
        
    except Exception as e:
        raise (f"Failed to start server: {str(e)}")
        

if __name__ == "__main__":
    main()