import warnings
warnings.filterwarnings("ignore")

import asyncio
import sys
from client import MCPClient

async def test_weather_llm():
    """Test weather tools using LLM with natural language queries"""
    if len(sys.argv) < 2:
        print("Usage: python test_weather_llm.py <URL of SSE MCP server>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        
        # Test queries
        test_queries = [
            "What's the weather forecast for Los Angeles?",
            "Are there any weather alerts in California?",
            "Get the weather forecast for New York City (40.7128, -74.0060)",
            "Show me weather alerts for Texas"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*50}")
            print(f"Test {i}: {query}")
            print(f"{'='*50}")
            
            try:
                response = await client.process_query(query)
                print(f"Response:\n{response}")
            except Exception as e:
                print(f"Error: {str(e)}")
            
            print()
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(test_weather_llm()) 