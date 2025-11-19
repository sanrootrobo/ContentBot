import os
import requests
import logging
import argparse
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
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
    max_content_length: int = 10000
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    max_concurrent_requests: int = 5
    competitor_analysis_count: int = 10
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
        """Validate input data."""
        if not self.brand or not self.title:
            raise ValueError("Brand and title are required")
        if not self.primary_keywords:
            raise ValueError("At least one primary keyword is required")

class APIKeyManager:
    """Enhanced API key management."""
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10:
                logger.error(f"Invalid API key in {filepath}")
                return None
            logger.info(f"API key loaded successfully from {filepath}")
            return api_key
        except Exception as e:
            logger.error(f"Error reading API key from {filepath}: {e}")
            return None

class GoogleSearchAPI:
    """Google Custom Search API client."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        if not self.api_key:
            raise ValueError(f"Google Search API key not found or invalid in {config.google_search_api_file}")
        if not self.search_engine_id:
            raise ValueError(f"Google Search Engine ID not found or invalid in {config.google_cx_file}")
        logger.info("Google Custom Search API initialized successfully")
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        try:
            params = {
                'key': self.api_key, 'cx': self.search_engine_id, 'q': query,
                'num': min(num_results, 10), 'fields': 'items(title,link,snippet,displayLink)'
            }
            if self.config.verbose:
                logger.info(f"Executing Google Search for query: {query}")
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            results = [{'title': i.get('title',''), 'url': i.get('link',''), 'snippet': i.get('snippet',''), 'domain': i.get('displayLink','')} for i in data.get('items', [])]
            if self.config.verbose:
                logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Google Search API error: {e}")
            return []

class FactualResearcher:
    """Gathers factual snippets from the web to ground the LLM."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        
    def gather_facts(self, blog_input: BlogInput, competitor_analysis: Dict[str, Any]) -> List[str]:
        logger.info("Gathering factual information to support content generation.")
        queries = set()
        for keyword in blog_input.primary_keywords:
            queries.add(f"what is {keyword}")
            queries.add(f"{keyword} statistics 2024")
        common_topics = competitor_analysis.get('common_topics', {})
        for topic in list(common_topics.keys())[:3]:
             queries.add(f"latest trends in {topic}")
        
        factual_snippets = []
        for query in list(queries)[:5]:
            try:
                results = self.google_search.search(query, num_results=3)
                for result in results:
                    if result.get('snippet'):
                        factual_snippets.append(result['snippet'].replace("\n", " ").strip())
                sleep(1)
            except Exception as e:
                logger.error(f"Error during factual search for query '{query}': {e}")
        logger.info(f"Collected {len(factual_snippets)} unique factual snippets.")
        return list(set(factual_snippets))

class CompetitorAnalyzer:
    """Analyzes competitor content for SEO insights."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
    
    def search_competitors(self, keywords: List[str]) -> List[Dict[str, Any]]:
        # Implementation from your original script is sound
        competitor_data = []
        for keyword in keywords[:3]: # Limit initial search to top 3 primary keywords
            try:
                search_query = f"{keyword} blog article"
                results = self.google_search.search(search_query, num_results=10)
                for i, result in enumerate(results[:5]):
                    domain = urlparse(result['url']).netloc.lower()
                    exclude_domains = ['google.com', 'facebook.com', 'twitter.com', 'youtube.com', 'instagram.com', 'wikipedia.org']
                    if not any(exc in domain for exc in exclude_domains):
                        competitor_data.append({
                            'url': result['url'], 'title': result['title'], 'snippet': result['snippet'],
                            'domain': result['domain'], 'keyword': keyword, 'search_position': i + 1
                        })
                sleep(1)
            except Exception as e:
                logger.error(f"Error searching for keyword '{keyword}': {e}")
        
        seen_urls = set()
        unique_competitors = []
        for comp in competitor_data:
            if comp['url'] not in seen_urls:
                seen_urls.add(comp['url'])
                unique_competitors.append(comp)
        
        logger.info(f"Found {len(unique_competitors)} unique competitor pages to analyze.")
        return unique_competitors[:self.config.competitor_analysis_count]

    def analyze_competitor_content(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Implementation from your original script is sound
        analysis_results = {'total_analyzed': 0, 'successful_analyses': 0, 'common_topics': {}}
        successful_analyses = []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            future_to_comp = {executor.submit(self._analyze_single_competitor, comp): comp for comp in competitors}
            for future in as_completed(future_to_comp):
                try:
                    result = future.result()
                    if result and result.get('success'):
                        successful_analyses.append(result)
                        analysis_results['successful_analyses'] += 1
                except Exception as e:
                    logger.error(f"Error processing competitor analysis result: {e}")
                analysis_results['total_analyzed'] += 1
        
        if successful_analyses:
            analysis_results.update(self._process_competitor_analyses(successful_analyses))
        logger.info(f"Completed competitor analysis: {analysis_results['successful_analyses']}/{analysis_results['total_analyzed']} pages analyzed.")
        return analysis_results

    def _analyze_single_competitor(self, competitor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = competitor['url']
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            title = soup.find('title').get_text().strip() if soup.find('title') else ''
            main_content = soup.find('main') or soup.find('article') or soup.body
            if not main_content: return None
            content_text = main_content.get_text(separator=' ', strip=True)
            return {'success': True, 'content': content_text, 'word_count': len(content_text.split())}
        except Exception as e:
            logger.warning(f"Failed to analyze competitor {url}: {e}")
            return None

    def _process_competitor_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        word_counts = [a['word_count'] for a in analyses if a.get('word_count', 0) > 0]
        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
        all_content = ' '.join([a['content'].lower() for a in analyses])
        words = re.findall(r'\b\w{5,15}\b', all_content)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        common_topics = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:20])
        return {'avg_word_count': int(avg_word_count), 'common_topics': common_topics}

class MarkdownFormatter:
    # Your original formatter is good.
    @staticmethod
    def format_blog_output(result: Dict[str, Any], blog_input: BlogInput) -> str:
        # This can be expanded as needed.
        return f"# {blog_input.title}\n\n{result['blog_content']}"

class SEOBlogWriter:
    """Main SEO blog writing agent."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.researcher = FactualResearcher(config)
        self.formatter = MarkdownFormatter()
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        api_key = APIKeyManager.get_api_key(self.config.api_key_file)
        if not api_key:
            raise ValueError("Gemini API key is required")
        return ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key, convert_system_message_to_human=True)
            
    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing process for title: '{blog_input.title}'")
        try:
            logger.info(f"[Phase 1/5] Analyzing Competitors...")
            competitors = self.analyzer.search_competitors(blog_input.primary_keywords)
            analysis_results = self.analyzer.analyze_competitor_content(competitors)
            
            logger.info(f"[Phase 2/5] Gathering Factual Information...")
            factual_snippets = self.researcher.gather_facts(blog_input, analysis_results)
            
            logger.info(f"[Phase 3/5] Building prompt for content generation...")
            prompt = self._build_generation_prompt(blog_input, analysis_results, factual_snippets)
            
            logger.info(f"[Phase 4/5] Generating blog content with LLM...")
            response = self.llm.invoke(prompt)
            blog_content = response.content
            
            logger.info(f"[Phase 5/5] Finalizing content.")
            word_count = len(blog_content.split())
            
            result = {"success": True, "blog_content": blog_content, "metadata": {"word_count": word_count}}
            logger.info(f"Successfully finished writing process for '{blog_input.title}' ({word_count} words)")
            return result
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing '{blog_input.title}': {e}", exc_info=True)
            return {"success": False, "error": str(e), "blog_content": "", "metadata": {}}

    def _build_generation_prompt(self, blog_input: BlogInput, analysis: Dict[str, Any], factual_snippets: List[str]) -> str:
        analysis_summary = json.dumps({
            "avg_word_count": analysis.get('avg_word_count', 'N/A'),
            "common_topics": list(analysis.get('common_topics', {}).keys())
        }, indent=2)

        factual_summary = "\n".join([f'- "{snippet}"' for snippet in factual_snippets]) if factual_snippets else "No specific facts gathered. Rely on general knowledge."

        return f"""
You are an expert SEO content writer and subject matter expert. Your task is to write a comprehensive, high-quality, and factually accurate blog post.

**CRITICAL INSTRUCTIONS:**
1.  **Output MUST be in valid Markdown format**, starting with the main title (`#`).
2.  **Ground your writing in the provided factual research**. Use these snippets to ensure accuracy. Do not invent statistics or data.
3.  **Use the competitor analysis** to understand common topics and appropriate length, but create a superior, more insightful article.
4.  **Integrate keywords naturally**. Avoid stuffing.

---
**BLOG POST REQUIREMENTS:**
- **Title**: {blog_input.title}
- **Primary Keywords**: {', '.join(blog_input.primary_keywords)}
- **Target Audience**: {blog_input.target_audience}
- **Tone**: {blog_input.tone}
- **Target Word Count**: Approx. {self.config.min_word_count}-{self.config.max_word_count} words.

---
**COMPETITOR ANALYSIS SUMMARY:**
This is what top-ranking content discusses.
{analysis_summary}

---
**FACTUAL RESEARCH SUMMARY:**
Use this information to ensure the blog post is accurate and trustworthy.
{factual_summary}

---
**YOUR TASK:**
Write the full blog post now. Adhere strictly to all instructions.

**BEGIN BLOG POST (MARKDOWN):**
"""
    
    def save_blog_to_file(self, result: Dict[str, Any], blog_input: BlogInput) -> str:
        safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')
        filename = f"{safe_title[:100]}.md"
        markdown_content = self.formatter.format_blog_output(result, blog_input)
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Content for '{blog_input.title}' saved to: {output_path.resolve()}")
        return str(output_path.resolve())

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    # Your original loader is good.
    try:
        logger.info(f"Loading blog inputs from config file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError("The JSON file must contain a list of blog post objects.")
        return [BlogInput(**item) for item in data]
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the configuration file: {filepath}")
        raise
    except TypeError as e:
        logger.error(f"Mismatch between JSON keys and BlogInput fields: {e}")
        raise

def parse_arguments():
    # Your original argument parser is excellent.
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts from a JSON file.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file containing blog input data.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file.')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
    parser.add_argument('--min-words', type=int, default=1500, help='Minimum word count.')
    parser.add_argument('--max-words', type=int, default=3000, help='Maximum word count.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    return parser.parse_args()

def main():
    """Main function to run the SEO blog writer, using your original robust structure."""
    args = parse_arguments()
    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            min_word_count=args.min_words,
            max_word_count=args.max_words,
            verbose=args.verbose
        )
        writer = SEOBlogWriter(config)
        for i, blog_input in enumerate(blog_inputs):
            logger.info(f"--- Starting job {i+1} of {len(blog_inputs)}: '{blog_input.title}' ---")
            # This retry loop is excellent for handling transient API issues.
            for attempt in range(config.max_retries):
                try:
                    result = writer.write_blog(blog_input)
                    if result and result.get('success'):
                        saved_file = writer.save_blog_to_file(result, blog_input)
                        print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' saved to: {saved_file}\n")
                        break
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        logger.error(f"Failed to generate blog post for '{blog_input.title}'. Reason: {error_msg}")
                        break
                except ResourceExhausted as e:
                    if attempt < config.max_retries - 1:
                        logger.warning(f"API quota exceeded. Retrying in {config.retry_delay}s... (Attempt {attempt + 2}/{config.max_retries})")
                        sleep(config.retry_delay)
                    else:
                        logger.error("API quota exceeded and max retries reached. Aborting.")
                        print(f"\n❌ ERROR: Could not generate blog post for '{blog_input.title}' due to API rate limits.\n")
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
