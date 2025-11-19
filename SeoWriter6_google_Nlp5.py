import os
import requests
import logging
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from datetime import datetime

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Data Classes and Config ---
@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.5-pro"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

@dataclass
class BlogInput:
    """Input data structure for blog writing."""
    brand: str
    title: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    target_audience: str = "General Audience"
    tone: str = "Informative"
    content_type: str = "Blog Post" 
    
    def __post_init__(self):
        if not self.brand or not self.title: raise ValueError("Brand and title are required")
        if not self.primary_keywords: raise ValueError("At least one primary keyword is required")

# --- Utility Classes (APIKeyManager, GoogleSearchAPI) ---
class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        try:
            key_path = Path(filepath);
            if not key_path.exists(): logger.error(f"API key file not found: '{filepath}'"); return None
            with open(key_path, 'r', encoding='utf-8') as f: api_key = f.read().strip()
            if not api_key or len(api_key) < 10: logger.error(f"Invalid API key in {filepath}"); return None
            return api_key
        except Exception as e: logger.error(f"Error reading API key: {e}"); return None

class GoogleSearchAPI:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        if not self.api_key: raise ValueError("Google Search API key not found")
        if not self.search_engine_id: raise ValueError("Google Search Engine ID not found")

    def search(self, query: str, num_results: int = 3, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        try:
            params = {'key': self.api_key, 'cx': self.search_engine_id, 'q': query, 'num': num_results}
            log_msg = f"Executing Fact-Check Search for: {query}"
            if sort_by_date:
                params['sort'] = 'date'
                log_msg = f"Executing News Search (sorted by date) for: {query}"
            if self.config.verbose: logger.info(log_msg)
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except Exception as e: logger.error(f"Google Search API error for query '{query}': {e}"); return []

# --- LLM-Powered Query Generator ---
class QueryGenerator:
    """Uses an LLM to generate a set of search queries."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=APIKeyManager.get_api_key(config.api_key_file))

    def _build_prompt(self, blog_input: BlogInput) -> str:
        # --- MODIFICATION: Added current date to the prompt ---
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"""
You are a world-class research assistant. Your task is to generate a list of Google search queries to gather information for a blog post.

**Research Context:**
- **Current Date:** {current_date}
- **Blog Post Title:** "{blog_input.title}"
- **Primary Keywords:** {blog_input.primary_keywords}
- **Target Audience:** {blog_input.target_audience}

Generate two types of queries based on the current date and blog topic:
1.  **news_queries:** To find the absolute latest news, announcements, and trends. These should be highly timely.
2.  **fact_queries:** To find foundational, evergreen information like definitions, explanations, and core data.

**CRITICAL:** Respond with ONLY a valid JSON object containing the two keys "news_queries" and "fact_queries". Do not include any other text, comments, or markdown.

Example format:
{{
  "news_queries": ["latest advancements in {{"keyword"}}", "{{"keyword"}} announcements {datetime.now().year}"],
  "fact_queries": ["what is {{"keyword"}}", "how does {{"keyword"}} work"]
}}
"""

    def generate_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        logger.info("Generating search queries with LLM...")
        prompt = self._build_prompt(blog_input)
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            queries = json.loads(cleaned_response)
            logger.info(f"Successfully generated {len(queries.get('news_queries',[]))} news and {len(queries.get('fact_queries',[]))} fact queries with LLM.")
            return queries
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}. Falling back to default queries.")
            return {
                "news_queries": [f'"{k}" latest news {datetime.now().year}' for k in blog_input.primary_keywords],
                "fact_queries": [f"what is {k}" for k in blog_input.primary_keywords]
            }

# --- Researcher using the QueryGenerator ---
class NewsAndDataResearcher:
    """Gathers news and facts using LLM-generated queries."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        self.query_generator = QueryGenerator(config)

    def gather_news_and_facts(self, blog_input: BlogInput) -> List[str]:
        all_snippets = []
        query_dict = self.query_generator.generate_queries(blog_input)
        news_queries = query_dict.get("news_queries", [])
        fact_queries = query_dict.get("fact_queries", [])

        logger.info("--- Starting News-Focused Search Phase ---")
        for query in news_queries[:4]:
            results = self.google_search.search(query, sort_by_date=True)
            for result in results:
                if result.get('snippet'): all_snippets.append(result['snippet'].strip())
            sleep(1)

        logger.info("--- Starting Foundational Fact-Check Phase ---")
        for query in fact_queries[:3]:
            results = self.google_search.search(query, sort_by_date=False)
            for result in results:
                if result.get('snippet'): all_snippets.append(result['snippet'].strip())
            sleep(1)

        unique_snippets = list(set(all_snippets))
        logger.info(f"Collected {len(unique_snippets)} unique snippets.")
        return unique_snippets

# --- Main SEOBlogWriter and supporting classes ---
class SEOBlogWriter:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.researcher = NewsAndDataResearcher(config)

    def _setup_llm(self):
        return ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=APIKeyManager.get_api_key(self.config.api_key_file))

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing process for '{blog_input.title}'")
        try:
            logger.info("[Phase 1/3] Generating search queries and gathering research...")
            news_and_facts = self.researcher.gather_news_and_facts(blog_input)
            
            logger.info("[Phase 2/3] Building prompt for content generation...")
            prompt = self._build_generation_prompt(blog_input, news_and_facts)
            
            logger.info("[Phase 3/3] Generating final blog content...")
            llm = self._setup_llm()
            response = llm.invoke(prompt)
            blog_content = response.content

            return {"success": True, "blog_content": blog_content}
        except Exception as e:
            logger.error(f"Error in blog writing process: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_generation_prompt(self, blog_input: BlogInput, research_snippets: List[str]) -> str:
        research_summary = "\n".join([f'- "{snippet}"' for snippet in research_snippets]) if research_snippets else "No research gathered."
        return f"""
You are an expert tech journalist and real estate analyst. Your primary goal is to write a blog post that is extremely current, data-driven, and well-researched.

**CRITICAL INSTRUCTIONS:**
1.  **Prioritize Freshness:** Use the `LATEST NEWS & FACTUAL RESEARCH` to build your article. The content must feel like it was written this week.
2.  **Data and Tables:** If the user's tone requests tables (like "{blog_input.tone}"), you must include them in valid Markdown format. Use the research data to populate them.
3.  **Output in Markdown:** The entire output must be valid Markdown.

---
**BLOG POST REQUIREMENTS:**
- **Title**: {blog_input.title}
- **Tone**: {blog_input.tone}

---
**LATEST NEWS & FACTUAL RESEARCH (Your Primary Source of Truth):**
{research_summary}

---
**YOUR TASK:**
Write the full blog post now. Start with the most recent and exciting information from the research. Explain what's new, why it matters, and provide background context. Adhere strictly to the requested tone.

**BEGIN BLOG POST (MARKDOWN):**
"""

# --- Main execution block ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts using LLM-generated queries.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file with blog input data.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file.')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    return parser.parse_args()

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        return [BlogInput(**item) for item in data]
    except Exception as e:
        logger.error(f"Error loading config file {filepath}: {e}"); raise

def save_blog_to_file(blog_content: str, blog_input: BlogInput):
    safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')
    filename = Path(f"{safe_title[:100]}.md")
    filename.write_text(blog_content, encoding='utf-8')
    logger.info(f"Blog post saved to: {filename.resolve()}")

def main():
    args = parse_arguments()
    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            verbose=args.verbose
        )
        writer = SEOBlogWriter(config)
        for blog_input in blog_inputs:
            result = writer.write_blog(blog_input)
            if result and result.get('success'):
                save_blog_to_file(result['blog_content'], blog_input)
                print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' complete.\n")
            else:
                print(f"\n❌ ERROR: Could not generate blog post for '{blog_input.title}'.\n")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()
