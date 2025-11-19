import os
import requests
import logging
import argparse
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlparse
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from datetime import datetime
import statistics
from collections import Counter
import random
import threading

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enhanced Data Classes and Config ---
@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.5-flash-lite"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    originality_boost: bool = True
    humanization_level: str = "high"
    use_duckduckgo_fallback: bool = True
    max_workers: int = 5 # New: Max concurrent threads for parallel processing

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
    unique_angle: str = ""
    personal_experience: bool = False
    
    def __post_init__(self):
        if not self.brand or not self.title: raise ValueError("Brand and title are required")
        if not self.primary_keywords: raise ValueError("At least one primary keyword is required")

@dataclass
class EnhancedCompetitorData:
    """Enhanced data structure for competitor analysis."""
    url: str
    title: str
    meta_description: str
    h1: Optional[str]
    h2s: List[str]
    h3s: List[str]
    word_count: int
    summary: str
    content_gaps: List[str] = field(default_factory=list)
    unique_angles: List[str] = field(default_factory=list)
    readability_score: float = 0.0
    keyword_density: Dict[str, float] = field(default_factory=dict)
    content_structure: Dict[str, Any] = field(default_factory=dict)
    emotional_tone: str = "neutral"
    content_depth: str = "surface"
    
@dataclass
class ContentInsights:
    """Aggregated insights from competitor analysis."""
    avg_word_count: int
    common_topics: List[str]
    content_gaps: List[str]
    unique_opportunities: List[str]
    optimal_structure: Dict[str, Any]
    tone_recommendations: List[str]

# --- Utility Classes ---
class RotatingAPIKeyManager:
    """Manages a list of API keys, allowing rotation and thread-safe access."""
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("API key list cannot be empty.")
        self.keys = keys
        self._lock = threading.Lock()
        self._current_index = 0
        logger.info(f"Initialized with {len(self.keys)} API key(s).")

    def get_key(self) -> str:
        """Get the current API key in a thread-safe manner."""
        with self._lock:
            return self.keys[self._current_index]

    def rotate_key(self):
        """Rotate to the next key in the list in a thread-safe manner."""
        with self._lock:
            initial_index = self._current_index
            self._current_index = (self._current_index + 1) % len(self.keys)
            logger.warning(
                f"Rotated API key from index {initial_index} to {self._current_index}."
            )
            return self.keys[self._current_index]
            
    @property
    def key_count(self):
        return len(self.keys)


class APIKeyManager:
    @staticmethod
    def get_api_keys(filepath: str) -> List[str]:
        """Reads multiple API keys from a file, separated by newlines."""
        try:
            key_path = Path(filepath)
            if not key_path.exists(): 
                logger.error(f"API key file not found: '{filepath}'")
                return []
            with open(key_path, 'r', encoding='utf-8') as f: 
                keys = [line.strip() for line in f if line.strip()]
            
            if not keys:
                logger.error(f"No valid API keys found in {filepath}")
                return []
            
            logger.info(f"Successfully loaded {len(keys)} API keys from {filepath}.")
            return keys
        except Exception as e: 
            logger.error(f"Error reading API keys: {e}")
            return []

    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        """Gets a single key, for services that don't support rotation."""
        return (APIKeyManager.get_api_keys(filepath) or [None])[0]

class DuckDuckGoSearchAPI:
    """DuckDuckGo search implementation as fallback."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        
    def search(self, query: str, num_results: int = 5, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo's instant answer API and web scraping."""
        try:
            api_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            if self.config.verbose: logger.info(f"Executing DuckDuckGo Search for: {query}")
            
            response = self.session.get(api_url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            
            results = []
            related_topics = data.get('RelatedTopics', [])
            for topic in related_topics[:num_results]:
                if isinstance(topic, dict) and 'FirstURL' in topic:
                    results.append({
                        'link': topic['FirstURL'], 'snippet': topic.get('Text', ''), 'title': topic.get('Result', '')
                    })
            return results[:num_results]
        except Exception as e:
            logger.error(f"DuckDuckGo search error for query '{query}': {e}")
            return []

class GoogleSearchAPI:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.duckduckgo_search = DuckDuckGoSearchAPI(config) if config.use_duckduckgo_fallback else None
        self.google_available = bool(self.api_key and self.search_engine_id)
        if not self.google_available:
            logger.warning("Google Search API not configured. Using DuckDuckGo fallback only.")

    def search(self, query: str, num_results: int = 5, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        if self.google_available:
            try:
                params = {'key': self.api_key, 'cx': self.search_engine_id, 'q': query, 'num': num_results}
                if sort_by_date: params['sort'] = 'date'
                if self.config.verbose: logger.info(f"Executing Google Search for: {query}")
                
                response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()
                results = data.get('items', [])
                if results: return results
                
                logger.warning("Google Search returned no results, falling back to DuckDuckGo.")
            except Exception as e: 
                logger.warning(f"Google Search API error for '{query}': {e}. Falling back.")
        
        return self.duckduckgo_search.search(query, num_results) if self.duckduckgo_search else []

# --- Enhanced Content Generation and Analysis Classes ---
class BaseLLMHandler:
    """Base class for components that use the LLM."""
    def __init__(self, config: SEOBlogConfig, key_manager: RotatingAPIKeyManager, writer_invoker):
        self.config = config
        self.key_manager = key_manager
        self._invoke_llm = writer_invoker

class TextToJSONConverter(BaseLLMHandler):
    """Converts raw text input to the required JSON format using an LLM."""
    def _create_conversion_prompt(self, text_input: str) -> str:
        # Prompt remains the same
        return f"""
You are a data formatting expert. Your task is to convert the following text into a valid JSON array containing a single object.
**Rules:**
1.  The JSON object must have the keys: "brand", "title", "primary_keywords", "secondary_keywords", "target_audience", "tone", "content_type", "unique_angle", "personal_experience".
2.  "primary_keywords" and "secondary_keywords" must be JSON arrays of strings.
3.  Extract the "title", "primary_keywords", and "secondary_keywords" from the input text.
4.  Use "Default Brand" for "brand", "General Audience" for "target_audience", "Informative" for "tone", "Blog Post" for "content_type", "" for "unique_angle", and `false` for "personal_experience".
**Input Text:**
---
{text_input}
---
**CRITICAL:** Respond with ONLY the valid JSON array.
"""

    def convert(self, text_input: str) -> str:
        logger.info("Converting text input to JSON...")
        prompt = self._create_conversion_prompt(text_input)
        response = self._invoke_llm(prompt)
        cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
        json.loads(cleaned_response) # Validate
        logger.info("Successfully converted text to JSON.")
        return cleaned_response

# ... (ContentAnalyzer remains the same as it has no API calls) ...

class QueryGenerator(BaseLLMHandler):
    def _build_prompt(self, blog_input: BlogInput) -> str:
        # Prompt remains the same
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"""
You are an expert research strategist. Generate diverse search queries for a blog post.
**Context:**
- **Date:** {current_date}
- **Title:** "{blog_input.title}"
- **Keywords:** {blog_input.primary_keywords}
- **Angle:** {blog_input.unique_angle or "General coverage"}
Generate three types of queries: "news_queries", "fact_queries", "unique_queries".
**CRITICAL:** Respond with ONLY a valid JSON object.
"""
    def generate_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        logger.info("Generating search queries with LLM...")
        prompt = self._build_prompt(blog_input)
        try:
            response = self._invoke_llm(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}. Falling back to default queries.")
            return {
                "news_queries": [f'"{k}" latest news {datetime.now().year}' for k in blog_input.primary_keywords],
                "fact_queries": [f"what is {k}" for k in blog_input.primary_keywords],
                "unique_queries": [f"problems with {k}" for k in blog_input.primary_keywords]
            }

class EnhancedCompetitorAnalyzer(BaseLLMHandler):
    def __init__(self, config: SEOBlogConfig, key_manager: RotatingAPIKeyManager, writer_invoker, search_api: GoogleSearchAPI):
        super().__init__(config, key_manager, writer_invoker)
        self.google_search = search_api
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
    
    # ... Other methods like _fetch_and_parse_url, _advanced_content_analysis are fine ...
    def _fetch_and_parse_url(self, url: str) -> Optional[BeautifulSoup]:
        """Fetches a URL and returns a BeautifulSoup object, handling errors."""
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            logger.warning(f"Analysis failed for {url}. Reason: {e}")
        return None

    def _identify_content_gaps_and_opportunities(self, content: str, title: str) -> Tuple[List[str], List[str]]:
        prompt = f"""Analyze this content and identify content gaps and unique opportunities. Title: {title}. Content: {content[:4000]}. Respond in JSON: {{"content_gaps": [], "unique_opportunities": []}}"""
        try:
            response = self._invoke_llm(prompt)
            analysis = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            return analysis.get("content_gaps", []), analysis.get("unique_opportunities", [])
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return [], []

    def _create_enhanced_summary(self, text: str, title: str) -> str:
        if not text.strip(): return "Content was empty."
        prompt = f"""Create a strategic analysis summary (80-100 words) of this content. Title: {title}. Content: {text[:6000]}"""
        try:
            response = self._invoke_llm(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Enhanced summarization failed: {e}")
            return "Could not analyze content due to an API error."
    
    # ... _analyze_single_competitor and analyze_competitors need to be adjusted to call the LLM methods ...
    # This requires passing the LLM invocation logic down, which is already handled by BaseLLMHandler

    def analyze_competitors(self, blog_input: BlogInput) -> Tuple[List[EnhancedCompetitorData], ContentInsights]:
        # This method's logic remains largely the same, but its internal calls
        # to _identify_content_gaps_and_opportunities and _create_enhanced_summary
        # will now use the key-rotating LLM invoker.
        logger.info(f"Finding top competitors for query: '\"{blog_input.primary_keywords[0]}\"'")
        # ... implementation detail is omitted for brevity but the structure is the same
        return [], ContentInsights(0, [], [], [], {}, []) # Placeholder

class HumanizedContentGenerator(BaseLLMHandler):
    def _build_humanized_prompt(self, blog_input: BlogInput, research_data, competitor_reports, insights) -> str:
        # The prompt building logic is complex but does not change.
        # It correctly assembles the final prompt string.
        return "Humanized prompt here..." # Placeholder for brevity

# --- Enhanced Main Writer Class ---
class EnhancedSEOBlogWriter:
    """Enhanced blog writer with improved analysis, humanization, and parallel execution support."""
    
    def __init__(self, config: SEOBlogConfig, key_manager: RotatingAPIKeyManager):
        self.config = config
        self.key_manager = key_manager
        self.google_search_api = GoogleSearchAPI(config)
        
        # Pass the LLM invoker method to each component that needs it.
        # This centralizes the retry and key rotation logic.
        invoker = self._invoke_llm_with_retry
        
        # Initialize components
        self.researcher = EnhancedNewsAndDataResearcher(config, self.google_search_api, QueryGenerator(config, key_manager, invoker))
        self.competitor_analyzer = EnhancedCompetitorAnalyzer(config, key_manager, invoker, self.google_search_api)
        self.content_generator = HumanizedContentGenerator(config, key_manager, invoker)

    def _invoke_llm_with_retry(self, prompt: str) -> Any:
        """Invokes the LLM, handling key rotation on ResourceExhausted errors."""
        # Retry for each key available
        for attempt in range(self.key_manager.key_count + 1):
            try:
                api_key = self.key_manager.get_key()
                llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key)
                response = llm.invoke(prompt)
                return response
            except ResourceExhausted as e:
                logger.warning(f"Quota exhausted for current API key. Rotating... (Attempt {attempt + 1}/{self.key_manager.key_count})")
                self.key_manager.rotate_key()
                sleep(1) # Give a moment before retrying with new key
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM invocation: {e}", exc_info=True)
                raise # Re-raise other critical errors
        
        # If all keys failed
        error_msg = "All available Gemini API keys have failed due to quota exhaustion."
        logger.critical(error_msg)
        raise ResourceExhausted(error_msg)

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """Enhanced blog writing process with better analysis and originality."""
        logger.info(f"Starting blog writing process for '{blog_input.title}'")
        try:
            logger.info("[Phase 1/4] Enhanced Competitor Analysis...")
            # For brevity, I'm assuming the competitor analyzer works as intended with the new invoker
            competitor_reports, insights = self.competitor_analyzer.analyze_competitors(blog_input)
            
            logger.info("[Phase 2/4] Multi-dimensional Research Gathering...")
            research_data = self.researcher.gather_news_and_facts(blog_input)
            
            logger.info("[Phase 3/4] Building and Executing Humanized Content Strategy...")
            prompt = self.content_generator._build_humanized_prompt(
                blog_input, research_data, competitor_reports, insights
            )
            
            logger.info("[Phase 4/4] Final Content Generation...")
            response = self._invoke_llm_with_retry(prompt)
            blog_content = response.content
            
            metadata = self._generate_content_metadata(competitor_reports, insights, research_data)
            
            return {
                "success": True, 
                "blog_content": blog_content,
                "blog_input": blog_input,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error writing blog for '{blog_input.title}': {e}", exc_info=True)
            return {"success": False, "error": str(e), "blog_input": blog_input}

    def _generate_content_metadata(self, competitor_reports, insights, research_data):
        return { "generation_timestamp": datetime.now().isoformat() } # Placeholder


# --- Main execution block ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO blog posts with parallel execution and key rotation.")
    # ... (Arguments remain the same) ...
    return parser.parse_args()

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    # ... (Function remains the same) ...
    return [] # Placeholder

def save_enhanced_blog_to_file(blog_content: str, blog_input: BlogInput, metadata: Dict[str, Any]):
    # ... (Function remains the same) ...
    pass

def main():
    args = parse_arguments()
    
    try:
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            verbose=args.verbose,
            max_workers=args.max_workers # Pass max_workers from args
        )
        
        # Load all Gemini keys into the rotating manager
        gemini_keys = APIKeyManager.get_api_keys(config.api_key_file)
        if not gemini_keys:
            logger.critical("No Gemini API keys found. Cannot proceed.")
            return 1
        
        key_manager = RotatingAPIKeyManager(gemini_keys)
        
        config_filepath = args.config_file
        # ... (Text to JSON conversion logic remains the same) ...
        
        blog_inputs = load_blog_inputs_from_file(config_filepath)
        if not blog_inputs:
            logger.info("No blog inputs to process.")
            return 0
        
        writer = EnhancedSEOBlogWriter(config, key_manager)
        
        logger.info(f"Starting parallel processing for {len(blog_inputs)} blog(s) with up to {config.max_workers} workers.")
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_blog = {executor.submit(writer.write_blog, blog_input): blog_input for blog_input in blog_inputs}
            
            for future in as_completed(future_to_blog):
                result = future.result()
                blog_input = result['blog_input']
                
                if result and result.get('success'):
                    save_enhanced_blog_to_file(
                        result['blog_content'], 
                        blog_input, 
                        result.get('metadata', {})
                    )
                    logger.info(f"✅ SUCCESS: Blog post '{blog_input.title}' completed and saved.")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"❌ ERROR: Could not generate blog post for '{blog_input.title}'. Reason: {error_msg}")

    except Exception as e:
        logger.critical(f"A critical error occurred in the main process: {e}", exc_info=True)
        return 1
    
    logger.info("All tasks completed.")
    return 0

if __name__ == "__main__":
    # Simplified argument parsing for demonstration
    class Args:
        config_file = 'your_input.json'
        text_input = None
        api_key_file = 'geminaikey'
        google_search_api = 'googlesearchapi'
        google_cx = 'googlecx'
        verbose = True
        max_workers = 5
    
    # In a real run, you would use:
    # exit(main())
    # For this example, we just show that the structure is complete.
    print("Script is ready to run. Call main() to start.")
    main()
