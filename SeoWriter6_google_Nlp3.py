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
    max_concurrent_requests: int = 5
    competitor_analysis_count: int = 5 # Reduced for news-focus
    min_word_count: int = 1500
    max_word_count: int = 3000
    output_format: str = "markdown"

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

class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'"); return None
            with open(key_path, 'r', encoding='utf-8') as f: api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logger.error(f"Invalid API key in {filepath}"); return None
            logger.info(f"API key loaded successfully from {filepath}")
            return api_key
        except Exception as e:
            logger.error(f"Error reading API key from {filepath}: {e}"); return None

class GoogleSearchAPI:
    """Google Custom Search API client, now with date sorting capabilities."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        if not self.api_key: raise ValueError(f"Google Search API key not found in {config.google_search_api_file}")
        if not self.search_engine_id: raise ValueError(f"Google Search Engine ID not found in {config.google_cx_file}")
        logger.info("Google Custom Search API initialized successfully")
    
    def search(self, query: str, num_results: int = 10, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        """Performs a Google search, with an option to sort by date for freshness."""
        try:
            params = {
                'key': self.api_key, 'cx': self.search_engine_id, 'q': query,
                'num': min(num_results, 10), 'fields': 'items(title,link,snippet,displayLink)'
            }
            if sort_by_date:
                params['sort'] = 'date'
                log_msg = f"Executing news-focused search (sorted by date) for: {query}"
            else:
                log_msg = f"Executing standard search for: {query}"
            
            if self.config.verbose: logger.info(log_msg)

            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            results = [{'title': i.get('title',''), 'url': i.get('link',''), 'snippet': i.get('snippet',''), 'domain': i.get('displayLink','')} for i in data.get('items', [])]
            if self.config.verbose: logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API request failed: {e}"); return []
        except Exception as e:
            logger.error(f"Google Search API error: {e}"); return []

class NewsAndDataResearcher:
    """Gathers the latest news and foundational facts to create fresh, accurate content."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        
    def gather_news_and_facts(self, blog_input: BlogInput) -> List[str]:
        logger.info("Gathering latest news and foundational facts...")
        all_snippets = []

        # 1. News-Focused Queries
        news_queries = set()
        for keyword in blog_input.primary_keywords:
            news_queries.add(f'"{keyword}" latest news OR update')
            news_queries.add(f'"{keyword}" announcement {datetime.now().year}')

        logger.info("--- Starting News-Focused Search Phase ---")
        for query in list(news_queries)[:3]: # Limit queries to avoid API overuse
            results = self.google_search.search(query, num_results=3, sort_by_date=True)
            for result in results:
                if result.get('snippet'):
                    all_snippets.append(result['snippet'].replace("\n", " ").strip())
            sleep(1)

        # 2. Foundational Fact Queries
        fact_queries = set()
        for keyword in blog_input.primary_keywords:
            fact_queries.add(f"what is {keyword}")
        
        logger.info("--- Starting Foundational Fact Search Phase ---")
        for query in list(fact_queries)[:2]:
            results = self.google_search.search(query, num_results=2, sort_by_date=False)
            for result in results:
                if result.get('snippet'):
                    all_snippets.append(result['snippet'].replace("\n", " ").strip())
            sleep(1)

        unique_snippets = list(set(all_snippets))
        logger.info(f"Collected {len(unique_snippets)} unique snippets from news and fact searches.")
        return unique_snippets

class CompetitorAnalyzer:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
    
    def search_competitors(self, keywords: List[str]) -> List[Dict[str, Any]]:
        return [] # Disabling for a pure news focus

    def analyze_competitor_content(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Competitor analysis is disabled for this news-focused run.")
        return {}

class MarkdownFormatter:
    @staticmethod
    def format_blog_output(result: Dict[str, Any], blog_input: BlogInput) -> str:
        # Using a more detailed version from your original script for better output
        markdown_content = ["---"]
        markdown_content.append(f"title: \"{blog_input.title}\"")
        markdown_content.append(f"brand: {blog_input.brand}")
        markdown_content.append(f"created: {result['metadata']['created_at']}")
        markdown_content.append(f"word_count: {result['metadata']['word_count']}")
        markdown_content.append("primary_keywords:")
        for keyword in blog_input.primary_keywords:
            markdown_content.append(f"  - \"{keyword}\"")
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append(result["blog_content"])
        return "\n".join(markdown_content)

class SEOBlogWriter:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.researcher = NewsAndDataResearcher(config)
        self.formatter = MarkdownFormatter()
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        api_key = APIKeyManager.get_api_key(self.config.api_key_file)
        if not api_key: raise ValueError("Gemini API key is required")
        return ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key, convert_system_message_to_human=True)
            
    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing process for title: '{blog_input.title}'")
        try:
            logger.info("[Phase 1/4] Competitor Analysis (Skipped for news focus)")
            analysis_results = self.analyzer.analyze_competitor_content([])
            
            logger.info("[Phase 2/4] Gathering Latest News and Factual Data...")
            news_and_facts = self.researcher.gather_news_and_facts(blog_input)
            
            logger.info("[Phase 3/4] Building prompt for content generation...")
            prompt = self._build_generation_prompt(blog_input, analysis_results, news_and_facts)
            
            logger.info("[Phase 4/4] Generating blog content with LLM...")
            response = self.llm.invoke(prompt)
            blog_content = response.content
            
            word_count = len(blog_content.split())
            result = {
                "success": True, "blog_content": blog_content,
                "metadata": { "word_count": word_count, "created_at": datetime.now().isoformat() }
            }
            logger.info(f"Successfully finished writing process for '{blog_input.title}' ({word_count} words)")
            return result
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing '{blog_input.title}': {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_generation_prompt(self, blog_input: BlogInput, analysis: Dict[str, Any], news_and_facts: List[str]) -> str:
        research_summary = "\n".join([f'- "{snippet}"' for snippet in news_and_facts]) if news_and_facts else "No specific news or facts were gathered. Rely on general knowledge."
        return f"""
You are an expert tech journalist and SEO content writer. Your primary goal is to write a blog post that feels extremely current and up-to-date.

**CRITICAL INSTRUCTIONS:**
1.  **Prioritize Freshness:** The most important instruction is to use the `LATEST NEWS & RESEARCH` section to build your article. Start with the latest news, announcements, or data. The content must feel like it was written this week.
2.  **Use Foundational Facts for Context:** Use the more general facts from the research section to explain the "why" behind the news.
3.  **Output in Markdown:** The entire output must be valid Markdown, starting with a `#` title.
4.  **Integrate Keywords Naturally:** Weave in keywords where they make sense.

---
**BLOG POST REQUIREMENTS:**
- **Title**: {blog_input.title}
- **Primary Keywords**: {', '.join(blog_input.primary_keywords)}
- **Target Audience**: {blog_input.target_audience}
- **Tone**: {blog_input.tone}, with an emphasis on being current and journalistic.

---
**LATEST NEWS & FACTUAL RESEARCH (Your Primary Source of Truth):**
This is a mix of the latest news and foundational facts. **You must build your article around this information.**
{research_summary}

---
**YOUR TASK:**
Write the full blog post now. Start with the most recent and exciting information you found. Explain what's new, why it matters, and then use the foundational facts to give the reader the necessary background. Ensure the post is insightful, accurate, and incredibly fresh.

**BEGIN BLOG POST (MARKDOWN):**
"""
    
    def save_blog_to_file(self, result: Dict[str, Any], blog_input: BlogInput) -> str:
        safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')
        filename = f"{safe_title[:100]}.md"
        markdown_content = self.formatter.format_blog_output(result, blog_input)
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f: f.write(markdown_content)
        logger.info(f"Content for '{blog_input.title}' saved to: {output_path.resolve()}")
        return str(output_path.resolve())

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    try:
        logger.info(f"Loading blog inputs from config file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        if not isinstance(data, list): raise TypeError("JSON must contain a list of objects.")
        return [BlogInput(**item) for item in data]
    except FileNotFoundError: logger.error(f"Config file not found: {filepath}"); raise
    except json.JSONDecodeError: logger.error(f"Error decoding JSON: {filepath}"); raise
    except TypeError as e: logger.error(f"Mismatch between JSON keys and BlogInput fields: {e}"); raise

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts with a focus on fresh news.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file with blog input data.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file.')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    return parser.parse_args()

def main():
    """Main function to run the SEO blog writer."""
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
        for i, blog_input in enumerate(blog_inputs):
            logger.info(f"--- Starting job {i+1} of {len(blog_inputs)}: '{blog_input.title}' ---")
            for attempt in range(config.max_retries):
                try:
                    result = writer.write_blog(blog_input)
                    if result and result.get('success'):
                        saved_file = writer.save_blog_to_file(result, blog_input)
                        print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' saved to: {saved_file}\n")
                        break
                    else:
                        logger.error(f"Failed to generate blog post. Reason: {result.get('error', 'Unknown')}")
                        break
                except ResourceExhausted as e:
                    if attempt < config.max_retries - 1:
                        logger.warning(f"API quota exceeded. Retrying in {config.retry_delay}s...")
                        sleep(config.retry_delay)
                    else:
                        logger.error("API quota exceeded and max retries reached. Aborting.")
                        break
                except Exception as e:
                    logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                    break
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Could not start blog generation due to a configuration error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()

