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
import itertools
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
    model_name: str = "gemini-2.5-pro"
    # NEW: Specific model for lightweight tasks like query generation
    flash_model_name: str = "gemini-2.5-flash"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    originality_boost: bool = True 
    use_duckduckgo_fallback: bool = True
    max_parallel_blogs: int = 3

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
class APIKeyManager:
    """Manages single and multiple API key loading."""
    @staticmethod
    def load_api_keys(filepath: str) -> List[str]:
        """Retrieve and validate multiple API keys from a file (one key per line)."""
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'")
                return []
            
            with open(key_path, 'r', encoding='utf-8') as f:
                keys = [line.strip() for line in f if line.strip()]
            
            valid_keys = [key for key in keys if len(key) >= 10]
            
            if not valid_keys:
                logger.error(f"No valid API keys found in {filepath}.")
                return []

            logger.info(f"Loaded {len(valid_keys)} API keys successfully from {filepath}")
            return valid_keys
            
        except Exception as e:
            logger.error(f"Error reading multiple API keys from {filepath}: {e}")
            return []

    @staticmethod
    def get_single_api_key(filepath: str) -> Optional[str]:
        """Retrieves the first key from a file, suitable for non-Gemini services."""
        try:
            key_path = Path(filepath)
            if not key_path.exists(): 
                logger.error(f"API key file not found: '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f: 
                api_key = f.read().strip().splitlines()[0]
            if not api_key or len(api_key) < 10: 
                logger.error(f"Invalid API key in {filepath}")
                return None
            return api_key
        except Exception as e: 
            logger.error(f"Error reading single API key from {filepath}: {e}")
            return None

class DuckDuckGoSearchAPI:
    """DuckDuckGo search implementation as fallback."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo's HTML page."""
        try:
            if self.config.verbose:
                logger.info(f"Executing DuckDuckGo Search for: {query}")
            
            search_url = "https://duckduckgo.com/html/"
            params = {'q': query}
            
            response = self.session.get(search_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            result_divs = soup.find_all('div', class_='result')[:num_results]
            
            for div in result_divs:
                link_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('a', class_='result__snippet')
                
                if link_elem and snippet_elem:
                    results.append({
                        'link': link_elem['href'],
                        'title': link_elem.get_text(strip=True),
                        'snippet': snippet_elem.get_text(strip=True)
                    })
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error for query '{query}': {e}")
            return []

class GoogleSearchAPI:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_single_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_single_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.duckduckgo_search = DuckDuckGoSearchAPI(config) if config.use_duckduckgo_fallback else None
        self.google_available = bool(self.api_key and self.search_engine_id)
        if not self.google_available:
            logger.warning("Google Search API not configured. Will use DuckDuckGo search only.")

    def search(self, query: str, num_results: int = 5, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        if self.google_available:
            try:
                params = {'key': self.api_key, 'cx': self.search_engine_id, 'q': query, 'num': min(num_results, 10)}
                if sort_by_date: params['sort'] = 'date'
                if self.config.verbose: logger.info(f"Executing Google Search for: {query}")
                
                response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()
                results = [{'link': item.get('link'), 'title': item.get('title'), 'snippet': item.get('snippet')} for item in data.get('items', [])]

                if results: return results
                logger.warning("Google Search returned no results, falling back to DuckDuckGo")
            except Exception as e: 
                logger.warning(f"Google Search API error for '{query}': {e}. Falling back to DuckDuckGo.")
        
        if self.duckduckgo_search:
            return self.duckduckgo_search.search(query, num_results)
        
        logger.error("No search engines available.")
        return []

# --- Enhanced Content Analysis Tools ---
class ContentAnalyzer:
    """Advanced content analysis for competitor insights."""
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        if not text.strip(): return 0.0
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences: return 0.0
        words = re.findall(r'\w+', text)
        if not words: return 0.0
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        return max(0, 100 - (avg_sentence_length * 2) - (avg_word_length * 10))
    
    @staticmethod
    def analyze_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
        text_lower = text.lower()
        word_count = len(re.findall(r'\w+', text))
        if word_count == 0: return {}
        density = {kw: (text_lower.count(kw.lower()) / word_count) * 100 for kw in keywords}
        return density
    
    @staticmethod
    def detect_emotional_tone(text: str) -> str:
        positive_words = ['amazing', 'excellent', 'great', 'wonderful', 'fantastic']
        negative_words = ['terrible', 'awful', 'bad', 'worst', 'horrible']
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        if pos_count > neg_count * 1.5: return "positive"
        if neg_count > pos_count * 1.5: return "negative"
        return "neutral"
    
    @staticmethod
    def assess_content_depth(text: str, h2_count: int, h3_count: int) -> str:
        word_count = len(re.findall(r'\w+', text))
        score = 0
        if word_count > 2000: score += 3
        elif word_count > 1000: score += 2
        if h2_count > 5: score += 2
        elif h2_count > 3: score += 1
        if h3_count > 3: score += 1
        if score >= 5: return "deep"
        if score >= 3: return "moderate"
        return "surface"

# --- LLM-Powered Tools ---
# MODIFIED: This class now initializes its own lightweight LLM.
class QueryGenerator:
    """Generates diverse search queries using a lightweight model."""
    def __init__(self, config: SEOBlogConfig, api_key: str):
        self.config = config
        logger.info(f"Initializing QueryGenerator with lightweight model: {config.flash_model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=config.flash_model_name, 
            google_api_key=api_key
        )

    def generate_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        logger.info("Generating enhanced search queries with LLM...")
        prompt = f"""
Generate diverse, high-quality search queries for a blog post titled "{blog_input.title}" with primary keywords {blog_input.primary_keywords}. Create three types of queries:
1. **news_queries:** For latest trends and news.
2. **fact_queries:** For foundational info and stats.
3. **unique_queries:** For unexplored angles and contrarian views.
Respond with ONLY a valid JSON object with these three keys.
"""
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            queries = json.loads(cleaned_response)
            return queries
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}. Falling back to default queries.")
            return {
                "news_queries": [f'"{k}" latest news {datetime.now().year}' for k in blog_input.primary_keywords],
                "fact_queries": [f"what is {k}" for k in blog_input.primary_keywords],
                "unique_queries": [f"problems with {k}" for k in blog_input.primary_keywords]
            }

class EnhancedCompetitorAnalyzer:
    """Advanced competitor analysis with deeper insights."""
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI, llm_instance: ChatGoogleGenerativeAI):
        self.config = config
        self.google_search = search_api
        self.llm = llm_instance
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        self.content_analyzer = ContentAnalyzer()

    def _fetch_and_parse_url(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            logger.warning(f"Analysis failed for {url}. Reason: {e}")
            return None

    def _identify_gaps_and_opportunities(self, content: str, title: str) -> Tuple[List[str], List[str]]:
        prompt = f"""
Analyze this competitor content. Identify content gaps (missing topics) and unique opportunities (uncovered angles).
Title: {title}
Content (first 4000 chars): {content[:4000]}
Respond in JSON: {{"content_gaps": ["gap1"], "unique_opportunities": ["opportunity1"]}}
"""
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            analysis = json.loads(cleaned_response)
            return analysis.get("content_gaps", []), analysis.get("unique_opportunities", [])
        except Exception:
            return [], []

    def _create_strategic_summary(self, text: str, title: str) -> str:
        prompt = f"""
Create a strategic analysis summary (80-100 words) of this competitor content. Focus on their strategy, strengths, weaknesses, and unique value.
Title: {title}
Content (first 6000 chars): {text[:6000]}
Strategic Summary:
"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception:
            return "Could not generate strategic summary."

    def _analyze_single_competitor(self, url: str, keywords: List[str]) -> Optional[EnhancedCompetitorData]:
        logger.info(f"Analyzing competitor: {url}")
        soup = self._fetch_and_parse_url(url)
        if not soup: return None

        title = soup.title.string.strip() if soup.title else "No Title"
        main_content_element = soup.find('main') or soup.find('article') or soup.body
        content_text = ' '.join(main_content_element.get_text(strip=True).split())
        
        h2s = [h.get_text(strip=True) for h in soup.find_all('h2')]
        h3s = [h.get_text(strip=True) for h in soup.find_all('h3')]
        
        gaps, opportunities = self._identify_gaps_and_opportunities(content_text, title)

        return EnhancedCompetitorData(
            url=url, title=title,
            meta_description=(soup.find('meta', attrs={'name': 'description'}) or {}).get('content', "N/A"),
            h1=(soup.find('h1') or {}).get_text(strip=True) or "N/A",
            h2s=h2s[:10], h3s=h3s[:15],
            word_count=len(re.findall(r'\w+', content_text)),
            summary=self._create_strategic_summary(content_text, title),
            content_gaps=gaps, unique_angles=opportunities,
            readability_score=self.content_analyzer.calculate_readability_score(content_text),
            keyword_density=self.content_analyzer.analyze_keyword_density(content_text, keywords),
            emotional_tone=self.content_analyzer.detect_emotional_tone(content_text),
            content_depth=self.content_analyzer.assess_content_depth(content_text, len(h2s), len(h3s)),
            content_structure={"has_lists": len(soup.find_all(['ul', 'ol'])) > 0, "has_images": len(soup.find_all('img')) > 0}
        )

    def analyze_competitors(self, blog_input: BlogInput) -> Tuple[List[EnhancedCompetitorData], ContentInsights]:
        search_query = f'"{blog_input.primary_keywords[0]}"'
        urls = [r['link'] for r in self.google_search.search(search_query, num_results=7) if r.get('link')]
        if not urls: return [], self._generate_content_insights([])

        reports = []
        all_keywords = blog_input.primary_keywords + blog_input.secondary_keywords
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._analyze_single_competitor, url, all_keywords): url for url in urls}
            for future in as_completed(future_to_url):
                if result := future.result(): reports.append(result)
        
        return reports, self._generate_content_insights(reports)

    def _generate_content_insights(self, reports: List[EnhancedCompetitorData]) -> ContentInsights:
        if not reports: return ContentInsights(1500, [], [], [], {}, [])
        
        word_counts = [r.word_count for r in reports]
        avg_word_count = int(statistics.median(word_counts) if word_counts else 1500)
        
        all_h2s = [h2.lower() for r in reports for h2 in r.h2s]
        common_topics = [topic for topic, count in Counter(all_h2s).most_common(10)]
        
        all_gaps = list(set([gap for r in reports for gap in r.content_gaps]))
        all_opportunities = list(set([opp for r in reports for opp in r.unique_angles]))
        
        optimal_structure = {
            "recommended_word_count": max(avg_word_count + 500, 1500),
            "recommended_h2_count": max(int(statistics.median([len(r.h2s) for r in reports if r.h2s])) + 2, 5),
            "should_include_images": any(r.content_structure.get("has_images") for r in reports),
        }
        
        return ContentInsights(avg_word_count, common_topics, all_gaps[:10], all_opportunities[:10], optimal_structure, [])

class EnhancedNewsAndDataResearcher:
    """Gathers categorized research data."""
    def __init__(self, search_api: GoogleSearchAPI, query_generator: QueryGenerator):
        self.google_search = search_api
        self.query_generator = query_generator

    def gather_news_and_facts(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        query_dict = self.query_generator.generate_queries(blog_input)
        research_data = {"news_snippets": [], "fact_snippets": [], "unique_snippets": []}

        for i, (key, sort_by_date) in enumerate([("news_queries", True), ("fact_queries", False), ("unique_queries", False)]):
            cat_name = key.replace("_queries", "_snippets")
            logger.info(f"--- Starting Research Phase: {cat_name.replace('_', ' ').title()} ---")
            for query in query_dict.get(key, [])[:3-i]:
                results = self.google_search.search(query, sort_by_date=sort_by_date)
                research_data[cat_name].extend([r['snippet'].strip() for r in results if r.get('snippet')])
                sleep(0.5)

        for key in research_data: research_data[key] = list(set(research_data[key]))
        return research_data

class OriginalContentGenerator:
    """Specialized class for generating original, high-quality content."""
    def __init__(self, llm_instance: ChatGoogleGenerativeAI):
        self.llm = llm_instance
    
    def _build_original_content_prompt(self, blog_input: BlogInput, research_data: Dict[str, List[str]], competitor_reports: List[EnhancedCompetitorData], insights: ContentInsights) -> str:
        research_summary = "\n\n".join([f"**{cat.replace('_', ' ').title()}:**\n" + "\n".join([f"• {s}" for s in snippets[:5]]) for cat, snippets in research_data.items() if snippets])
        competitor_summary = "\n".join([f"**Competitor {i+1}**: {r.title}\n- **Approach**: {r.content_depth} coverage, {r.emotional_tone} tone.\n- **Content Gaps**: {', '.join(r.content_gaps[:3]) or 'N/A'}" for i, r in enumerate(competitor_reports[:3])])
        strategy_section = f"""
**STRATEGIC INTELLIGENCE:**
- **Target Word Count**: {insights.optimal_structure.get('recommended_word_count', 1500)}+ words
- **Content Gaps to Fill**: {', '.join(insights.content_gaps[:5]) or 'N/A'}
- **Unique Opportunities**: {', '.join(insights.unique_opportunities[:3]) or 'N/A'}
"""
        originality_instructions = """
**CRITICAL WRITING REQUIREMENTS:**

**1. ORIGINALITY & AI-RESISTANCE (100% Priority):**
- Write from a unique perspective that competitors have not taken.
- Include unexpected insights, contrarian viewpoints, or lesser-known facts from the research.
- Use specific examples, case studies, and real-world applications.
- Vary sentence structure: mix short, punchy sentences with longer, explanatory ones.
- Include rhetorical questions to engage readers.
- Use active voice predominantly.

**2. STRICTLY FORBIDDEN ELEMENTS:**
- Never use these AI-typical phrases: "delve into", "tapestry", "landscape" (as metaphor), "navigate the complexities", "it's worth noting", "comprehensive guide", "in today's digital age".
- Avoid starting consecutive paragraphs with "Additionally", "Furthermore", "Moreover".
- Avoid generic, empty statements like "XYZ is important."

**3. CONTENT STRUCTURE:**
- Start with a compelling hook (a question, surprising fact, or bold statement).
- Include at least 2 unique angles not covered by competitors.
- Add practical, actionable advice.
- End with a memorable conclusion that reinforces key takeaways.
- Include 4+ relevant FAQs before the conclusion.
"""
        return f"""
You are a world-class content strategist specializing in creating original, engaging content that outperforms typical AI articles.
{originality_instructions}

**BLOG POST SPECIFICATIONS:**
- **Title**: {blog_input.title}
- **Primary Keywords**: {blog_input.primary_keywords}
- **Target Audience**: {blog_input.target_audience}
- **Tone**: {blog_input.tone}
- **Unique Angle**: {blog_input.unique_angle or "Find a fresh perspective"}

{strategy_section}

**COMPETITOR INTELLIGENCE:**
{competitor_summary}

**RESEARCH FOUNDATION:**
{research_summary}

**YOUR MISSION:** Create a blog post that is more comprehensive, original, and valuable than any competitor.

**OUTPUT FORMAT:** Complete blog post in Markdown format, starting with the H1 title.

**BEGIN BLOG POST:**
"""

    def generate(self, blog_input: BlogInput, research_data: Dict[str, List[str]], competitor_reports: List[EnhancedCompetitorData], insights: ContentInsights) -> str:
        prompt = self._build_original_content_prompt(blog_input, research_data, competitor_reports, insights)
        response = self.llm.invoke(prompt)
        return response.content
        
# --- Main Writer Class with Key Rotation ---
class EnhancedSEOBlogWriter:
    """Orchestrates blog writing with advanced analysis and API key rotation."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search_api = GoogleSearchAPI(config)
        self.api_keys = APIKeyManager.load_api_keys(self.config.api_key_file)
        if not self.api_keys:
            raise ValueError("No Gemini API keys found. Please check the geminaikey file.")
        self.key_cycler = itertools.cycle(self.api_keys)
        self.key_lock = threading.Lock()

    def _get_next_key(self) -> str:
        with self.key_lock:
            return next(self.key_cycler)

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """Enhanced blog writing process with retries using different API keys."""
        logger.info(f"Starting enhanced blog writing process for '{blog_input.title}'")
        
        for attempt in range(len(self.api_keys)):
            api_key = self._get_next_key()
            key_id = f"{api_key[:4]}...{api_key[-4:]}"
            logger.info(f"Attempt {attempt + 1}/{len(self.api_keys)} for '{blog_input.title}' using key {key_id}")
            
            try:
                # MODIFIED: Initialize two different LLM clients.
                # One for heavy tasks (Pro model) and one for light tasks (Flash model).
                
                # LLM for complex analysis and generation
                pro_llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key)
                
                # Initialize components with the correct LLM or API key
                # QueryGenerator now creates its own lightweight (Flash) LLM instance
                query_generator = QueryGenerator(self.config, api_key)
                
                # Other components use the powerful (Pro) LLM instance
                researcher = EnhancedNewsAndDataResearcher(self.google_search_api, query_generator)
                competitor_analyzer = EnhancedCompetitorAnalyzer(self.config, self.google_search_api, pro_llm)
                content_generator = OriginalContentGenerator(pro_llm)

                logger.info("[Phase 1/4] Enhanced Competitor Analysis...")
                competitor_reports, insights = competitor_analyzer.analyze_competitors(blog_input)
                
                logger.info("[Phase 2/4] Multi-dimensional Research Gathering...")
                research_data = researcher.gather_news_and_facts(blog_input)
                
                logger.info("[Phase 3/4] Building Original Content Strategy...")
                sleep(random.uniform(1, 2))
                
                logger.info("[Phase 4/4] Generating Original Content...")
                blog_content = content_generator.generate(blog_input, research_data, competitor_reports, insights)
                
                metadata = self._generate_content_metadata(competitor_reports, insights, research_data, key_id)
                
                logger.info(f"Successfully generated content for '{blog_input.title}'")
                return {
                    "success": True, "blog_content": blog_content, "metadata": metadata
                }
            
            except ResourceExhausted as e:
                logger.warning(f"API quota exceeded for key {key_id} on '{blog_input.title}'. Trying next key. Error: {e}")
                if attempt + 1 >= len(self.api_keys):
                    logger.error(f"All {len(self.api_keys)} API keys have failed for '{blog_input.title}'.")
                    return {"success": False, "error": f"All {len(self.api_keys)} API keys are rate-limited."}
                sleep(2)
            
            except Exception as e:
                logger.error(f"A critical error occurred while processing '{blog_input.title}' with key {key_id}: {e}", exc_info=True)
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Failed to generate blog for '{blog_input.title}' after trying all keys."}

    def _generate_content_metadata(self, reports: List[EnhancedCompetitorData], insights: ContentInsights, research: Dict, key_id: str) -> Dict[str, Any]:
        return {
            "competitors_analyzed": len(reports),
            "avg_competitor_word_count": insights.avg_word_count,
            "research_snippets_used": sum(len(s) for s in research.values()),
            "content_gaps_addressed": len(insights.content_gaps),
            "unique_opportunities_leveraged": len(insights.unique_opportunities),
            "generation_timestamp": datetime.now().isoformat(),
            "search_engine_used": "Google+DuckDuckGo" if self.google_search_api.google_available else "DuckDuckGo",
            "gemini_key_used": key_id
        }

# --- Main Functions with Parallel Processing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate enhanced, SEO-optimized blog posts in parallel.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to JSON file with blog inputs.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file (one key per line).')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
    parser.add_argument('--parallel-jobs', type=int, default=3, help='Max number of blogs to process in parallel.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--no-duckduckgo-fallback', action='store_true', help='Disable DuckDuckGo fallback.')
    return parser.parse_args()

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        return [BlogInput(**item) for item in data]
    except Exception as e:
        logger.error(f"Error loading config file {filepath}: {e}")
        raise

def save_enhanced_blog_to_file(blog_content: str, blog_input: BlogInput, metadata: Dict[str, Any]):
    safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')[:100]
    Path(f"{safe_title}.md").write_text(blog_content, encoding='utf-8')
    Path(f"{safe_title}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"Blog post and metadata saved for: {blog_input.title}")
    return Path(f"{safe_title}.md").resolve()

def process_blog_job(writer: EnhancedSEOBlogWriter, blog_input: BlogInput) -> Tuple[str, Optional[str]]:
    """Manages the lifecycle of a single blog post generation for parallel execution."""
    try:
        result = writer.write_blog(blog_input)
        if result and result.get('success'):
            saved_file = save_enhanced_blog_to_file(result['blog_content'], blog_input, result.get('metadata', {}))
            return blog_input.title, str(saved_file)
        else:
            reason = result.get('error', 'Unknown error') if result else "No result"
            logger.error(f"Failed to generate blog '{blog_input.title}'. Reason: {reason}")
            return blog_input.title, None
    except Exception as e:
        logger.error(f"A critical error occurred processing job '{blog_input.title}': {e}", exc_info=True)
        return blog_input.title, None

def main():
    """Enhanced main function with parallel processing."""
    args = parse_arguments()
    
    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            verbose=args.verbose,
            use_duckduckgo_fallback=not args.no_duckduckgo_fallback,
            max_parallel_blogs=args.parallel_jobs
        )
        
        writer = EnhancedSEOBlogWriter(config)
        
        num_jobs = len(blog_inputs)
        max_workers = min(config.max_parallel_blogs, num_jobs, len(writer.api_keys))
        
        print(f"\n🚀 Starting parallel generation of {num_jobs} blogs with up to {max_workers} workers.")
        print(f"🔑 Found {len(writer.api_keys)} Gemini API keys for rotation.")
        
        success_count, failure_count = 0, 0
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='BlogWriter') as executor:
            future_to_blog = {executor.submit(process_blog_job, writer, blog_input): blog_input for blog_input in blog_inputs}
            
            for future in as_completed(future_to_blog):
                title, saved_file = future.result()
                if saved_file:
                    print(f"\n✅ SUCCESS: Blog '{title}' complete. Saved to: {saved_file}\n")
                    success_count += 1
                else:
                    print(f"\n❌ FAILED: Blog '{title}' could not be generated. See logs for details.\n")
                    failure_count += 1
        
        print("--- All Jobs Completed ---")
        print(f"Total Successful: {success_count}")
        print(f"Total Failed: {failure_count}")

    except Exception as e:
        logger.error(f"Critical error in main process: {e}", exc_info=True)
        print(f"\n💥 CRITICAL ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
