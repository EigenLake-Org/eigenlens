"""OpenAI agent tool: semantic Qdrant query → DBSCAN clusters → saved plot."""
import asyncio
from contextlib import closing
import json
import os
from pathlib import Path
import uuid
from agents import Agent, Runner, function_tool, RunConfig
from qdrant_client import QdrantClient
from vector_tools.config import Config
from examples.banking77 import read_records
from examples.banking77_qdrant import DEFAULT_QUERY, query_and_plot


async def main():
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('OPENAI_MODEL'):
        raise SystemExit('Set OPENAI_API_KEY and OPENAI_MODEL. First run examples.banking77_qdrant demo to index data.')
    config = Config('config/banking77.json')
    _, manifest = read_records(config)
    # Connection, model revision and file destinations stay in trusted host code.
    @function_tool
    async def search_banking_clusters(query: str, top_k: int = 2000) -> str:
        """Search indexed BANKING77 banking queries; return DBSCAN groups and a local interactive plot path.

        Args:
            query: Banking support topic to search for.
            top_k: Number of nearest records to cluster, from 3 to 2000.
        """
        if not 3 <= top_k <= 2000:
            return 'top_k must be between 3 and 2000.'
        output = Path('demo-output/banking77-agent') / uuid.uuid4().hex
        def compute():
            with closing(QdrantClient(url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
                                     api_key=os.getenv('QDRANT_API_KEY'), timeout=120)) as client:
                return query_and_plot(client, 'banking77_minilm', manifest, query, top_k, output, config.limits)
        report = await asyncio.to_thread(compute)
        return json.dumps({k:v for k,v in report.items() if k != 'hits'})

    agent = Agent(name='Banking Query Analyst', model=os.environ['OPENAI_MODEL'], tools=[search_banking_clusters],
                  instructions='Use search_banking_clusters for banking cluster requests. Report the actual retrieved count, cluster sizes, noise, and plot path. Explain that groups describe the query matches, not the entire corpus. If no_valid_candidate, report that no suitable clustering was found. Do not invent cluster labels or quality scores. Make one search unless the user requests another.')
    result = await Runner.run(agent, os.getenv('BANKING_QUERY', DEFAULT_QUERY), max_turns=4,
                              run_config=RunConfig(tracing_disabled=True))
    print(result.final_output)


if __name__ == '__main__':
    asyncio.run(main())
