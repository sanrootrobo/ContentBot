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
import threading
import queue
import time
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
    flash_model_name: str = "gemini-2.5-flash"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    originality_boost: bool = True 
    use_duckduckgo_fallback: bool = True
    max_parallel_blogs: int = 3
    api_key_retry_delay: float = 2.0
    max_api_key_retries: int = 2

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
        if not self.brand or not self.title: 
            raise ValueError("Brand and title are required")
        if not self.primary_keywords: 
            raise ValueError("At least one primary keyword is required")

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

# --- Enhanced API Key Manager with Pool ---
class APIKeyPool:
    """Thread-safe API key pool with rate limiting and health tracking."""
    def __init__(self, api_keys: List[str], config: SEOBlogConfig):
        self.config = config
        self.available_keys = queue.Queue()
        self.rate_limited_keys = {}  # key -> timestamp when it can be used again
        self.failed_keys = set()
        self.lock = threading.Lock()
        self.key_usage_count = {}
        # Initialize pool
        for key in api_keys:
            self.available_keys.put(key)
            self.key_usage_count[key] = 0

    def get_key(self, timeout: float = 30.0) -> Optional[str]:
        """Get an available API key, waiting if necessary."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if any rate-limited keys are now available
            with self.lock:
                current_time = time.time()
                recovered_keys = []
                for key, limited_until in list(self.rate_limited_keys.items()):
                    if current_time >= limited_until:
                        recovered_keys.append(key)
                        del self.rate_limited_keys[key]
                for key in recovered_keys:
                    if key not in self.failed_keys:
                        self.available_keys.put(key)
                        logger.info(f"API key {key[:8]}... recovered from rate limiting")
            try:
                key = self.available_keys.get(timeout=1.0)
                with self.lock:
                    self.key_usage_count[key] += 1
                return key
            except queue.Empty:
                continue
        logger.warning("No API keys available after timeout")
        return None

    def return_key(self, key: str, success: bool = True):
        """Return a key to the pool or mark it as rate-limited/failed."""
        with self.lock:
            if success:
                self.available_keys.put(key)
            else:
                # Don't return failed keys immediately
                pass

    def mark_rate_limited(self, key: str, retry_after: float = None):
        """Mark a key as rate-limited."""
        if retry_after is None:
            retry_after = self.config.retry_delay
        with self.lock:
            self.rate_limited_keys[key] = time.time() + retry_after
            logger.warning(f"API key {key[:8]}... marked as rate-limited for {retry_after}s")

    def mark_failed(self, key: str):
        """Mark a key as permanently failed."""
        with self.lock:
            self.failed_keys.add(key)
            logger.error(f"API key {key[:8]}... marked as failed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                "available_keys": self.available_keys.qsize(),
                "rate_limited_keys": len(self.rate_limited_keys),
                "failed_keys": len(self.failed_keys),
                "key_usage_count": dict(self.key_usage_count)
            }

class APIKeyManager:
    """Manages API key loading and pool creation."""
    @staticmethod
    def load_api_keys(filepath: str) -> List[str]:
        """Retrieve and validate multiple API keys from a file."""
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
            logger.error(f"Error reading API keys from {filepath}: {e}")
            return []

    @staticmethod
    def get_single_api_key(filepath: str) -> Optional[str]:
        """Retrieves the first key from a file."""
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
            logger.error(f"Error reading API key from {filepath}: {e}")
            return None

# --- Search API Classes ---
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
                params = {
                    'key': self.api_key, 
                    'cx': self.search_engine_id, 
                    'q': query, 
                    'num': min(num_results, 10)
                }
                if sort_by_date: 
                    params['sort'] = 'date'
                if self.config.verbose: 
                    logger.info(f"Executing Google Search for: {query}")
                response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()
                results = [
                    {
                        'link': item.get('link'), 
                        'title': item.get('title'), 
                        'snippet': item.get('snippet')
                    } 
                    for item in data.get('items', [])
                ]
                if results: 
                    return results
                logger.warning("Google Search returned no results, falling back to DuckDuckGo")
            except Exception as e: 
                logger.warning(f"Google Search API error for '{query}': {e}. Falling back to DuckDuckGo.")
        if self.duckduckgo_search:
            return self.duckduckgo_search.search(query, num_results)
        logger.error("No search engines available.")
        return []

# --- Content Analysis Tools ---
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

# --- Enhanced LLM-Powered Tools ---
class QueryGenerator:
    """Generates diverse search queries using a lightweight model."""
    def __init__(self, config: SEOBlogConfig, api_key_pool: APIKeyPool):
        self.config = config
        self.api_key_pool = api_key_pool
        logger.info(f"Initializing QueryGenerator with lightweight model: {config.flash_model_name}")

    def generate_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        logger.info("Generating enhanced search queries with LLM...")
        api_key = self.api_key_pool.get_key()
        if not api_key:
            logger.warning("No API key available for query generation, using fallback queries")
            return self._get_fallback_queries(blog_input)
        try:
            llm = ChatGoogleGenerativeAI(
                model=self.config.flash_model_name, 
                google_api_key=api_key
            )
            prompt = f"""
Generate diverse, high-quality search queries for a blog post titled "{blog_input.title}" with primary keywords {blog_input.primary_keywords}. Create three types of queries:
1. **news_queries:** For latest trends and news.
2. **fact_queries:** For foundational info and stats.
3. **unique_queries:** For unexplored angles and contrarian views.
Respond with ONLY a valid JSON object with these three keys.
"""
            response = llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            queries = json.loads(cleaned_response)
            self.api_key_pool.return_key(api_key, success=True)
            return queries
        except ResourceExhausted as e:
            logger.warning(f"API quota exceeded during query generation: {e}")
            self.api_key_pool.mark_rate_limited(api_key)
            return self._get_fallback_queries(blog_input)
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}. Falling back to default queries.")
            self.api_key_pool.return_key(api_key, success=False)
            return self._get_fallback_queries(blog_input)

    def _get_fallback_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        """Fallback queries when LLM is unavailable."""
        return {
            "news_queries": [f'"{k}" latest news {datetime.now().year}' for k in blog_input.primary_keywords],
            "fact_queries": [f"what is {k}" for k in blog_input.primary_keywords],
            "unique_queries": [f"problems with {k}" for k in blog_input.primary_keywords]
        }

class EnhancedCompetitorAnalyzer:
    """Advanced competitor analysis with deeper insights."""
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI, api_key_pool: APIKeyPool):
        self.config = config
        self.google_search = search_api
        self.api_key_pool = api_key_pool
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
        api_key = self.api_key_pool.get_key()
        if not api_key:
            return [], []
        try:
            llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key)
            prompt = f"""
Analyze this competitor content. Identify content gaps (missing topics) and unique opportunities (uncovered angles).
Title: {title}
Content (first 4000 chars): {content[:4000]}
Respond in JSON: {{"content_gaps": ["gap1"], "unique_opportunities": ["opportunity1"]}}
"""
            response = llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            analysis = json.loads(cleaned_response)
            self.api_key_pool.return_key(api_key, success=True)
            return analysis.get("content_gaps", []), analysis.get("unique_opportunities", [])
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            self.api_key_pool.return_key(api_key, success=False)
            return [], []

    def _create_strategic_summary(self, text: str, title: str) -> str:
        api_key = self.api_key_pool.get_key()
        if not api_key:
            return "Could not generate strategic summary - no API key available."
        try:
            llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key)
            prompt = f"""
Create a strategic analysis summary (80-100 words) of this competitor content. Focus on their strategy, strengths, weaknesses, and unique value.
Title: {title}
Content (first 6000 chars): {text[:6000]}
Strategic Summary:
"""
            response = llm.invoke(prompt)
            self.api_key_pool.return_key(api_key, success=True)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Strategic summary generation failed: {e}")
            self.api_key_pool.return_key(api_key, success=False)
            return "Could not generate strategic summary."

    def _analyze_single_competitor(self, url: str, keywords: List[str]) -> Optional[EnhancedCompetitorData]:
        logger.info(f"Analyzing competitor: {url}")
        soup = self._fetch_and_parse_url(url)
        if not soup: 
            return None
        title = soup.title.string.strip() if soup.title else "No Title"
        main_content_element = soup.find('main') or soup.find('article') or soup.body
        content_text = ' '.join(main_content_element.get_text(strip=True).split()) if main_content_element else ""
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
            content_structure={
                "has_lists": len(soup.find_all(['ul', 'ol'])) > 0, 
                "has_images": len(soup.find_all('img')) > 0
            }
        )

    def analyze_competitors(self, blog_input: BlogInput) -> Tuple[List[EnhancedCompetitorData], ContentInsights]:
        search_query = f'"{blog_input.primary_keywords[0]}"'
        urls = [r['link'] for r in self.google_search.search(search_query, num_results=7) if r.get('link')]
        if not urls: 
            return [], self._generate_content_insights([])
        reports = []
        all_keywords = blog_input.primary_keywords + blog_input.secondary_keywords
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(self._analyze_single_competitor, url, all_keywords): url 
                for url in urls
            }
            for future in as_completed(future_to_url):
                if result := future.result(): 
                    reports.append(result)
        return reports, self._generate_content_insights(reports)

    def _generate_content_insights(self, reports: List[EnhancedCompetitorData]) -> ContentInsights:
        if not reports: 
            return ContentInsights(1500, [], [], [], {}, [])
        word_counts = [r.word_count for r in reports if r.word_count > 0]
        avg_word_count = int(statistics.median(word_counts) if word_counts else 1500)
        all_h2s = [h2.lower() for r in reports for h2 in r.h2s]
        common_topics = [topic for topic, count in Counter(all_h2s).most_common(10)]
        all_gaps = list(set([gap for r in reports for gap in r.content_gaps]))
        all_opportunities = list(set([opp for r in reports for opp in r.unique_angles]))
        h2_counts = [len(r.h2s) for r in reports if r.h2s]
        median_h2_count = int(statistics.median(h2_counts)) if h2_counts else 5
        optimal_structure = {
            "recommended_word_count": max(avg_word_count + 500, 1500),
            "recommended_h2_count": max(median_h2_count + 2, 5),
            "should_include_images": any(r.content_structure.get("has_images") for r in reports),
        }
        return ContentInsights(
            avg_word_count, 
            common_topics, 
            all_gaps[:10], 
            all_opportunities[:10], 
            optimal_structure, 
            []
        )

class EnhancedNewsAndDataResearcher:
    """Gathers categorized research data."""
    def __init__(self, search_api: GoogleSearchAPI, query_generator: QueryGenerator):
        self.google_search = search_api
        self.query_generator = query_generator

    def gather_news_and_facts(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        query_dict = self.query_generator.generate_queries(blog_input)
        research_data = {"news_snippets": [], "fact_snippets": [], "unique_snippets": []}
        query_configs = [
            ("news_queries", True, "news_snippets"),
            ("fact_queries", False, "fact_snippets"), 
            ("unique_queries", False, "unique_snippets")
        ]
        for i, (query_key, sort_by_date, result_key) in enumerate(query_configs):
            logger.info(f"--- Starting Research Phase: {result_key.replace('_', ' ').title()} ---")
            queries = query_dict.get(query_key, [])[:3-i]
            for query in queries:
                try:
                    results = self.google_search.search(query, sort_by_date=sort_by_date)
                    snippets = [r['snippet'].strip() for r in results if r.get('snippet')]
                    research_data[result_key].extend(snippets)
                    sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Research query failed: {query} - {e}")
        # Remove duplicates
        for key in research_data: 
            research_data[key] = list(set(research_data[key]))
        return research_data

class OriginalContentGenerator:
    """Specialized class for generating original, high-quality content."""
    def __init__(self, api_key_pool: APIKeyPool, config: SEOBlogConfig):
        self.api_key_pool = api_key_pool
        self.config = config

    def _build_original_content_prompt(self, blog_input: BlogInput, research_data: Dict[str, List[str]], 
                                     competitor_reports: List[EnhancedCompetitorData], insights: ContentInsights) -> str:
        research_summary = "\n".join([
            f"**{cat.replace('_', ' ').title()}:**\n" + "\n".join([f"• {s}" for s in snippets[:5]]) 
            for cat, snippets in research_data.items() if snippets
        ])
        competitor_summary = "\n".join([
            f"**Competitor {i+1}**: {r.title}\n- **Approach**: {r.content_depth} coverage, {r.emotional_tone} tone.\n- **Content Gaps**: {', '.join(r.content_gaps[:3]) or 'N/A'}" 
            for i, r in enumerate(competitor_reports[:3])
        ])
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
{strategy_section}
**RESEARCH DATA:**
{research_summary}
**COMPETITOR STRATEGIES:**
{competitor_summary}
**FINAL TASK:**
Using all the intelligence above, write a complete, original, and highly engaging {blog_input.content_type} titled "{blog_input.title}".
Ensure the primary keywords '{', '.join(blog_input.primary_keywords)}' are naturally integrated.
Target audience: {blog_input.target_audience}.
Tone: {blog_input.tone}.
If applicable, incorporate the unique angle: '{blog_input.unique_angle}'.
The final output must be a well-formatted blog post ready for publication.
"""

    def generate_content(self, blog_input: BlogInput, research_data: Dict[str, List[str]], 
                        competitor_reports: List[EnhancedCompetitorData], insights: ContentInsights) -> str:
        logger.info("Generating original blog content...")
        api_key = self.api_key_pool.get_key()
        if not api_key:
            return "Error: No API key available to generate content."

        try:
            llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key)
            prompt = self._build_original_content_prompt(blog_input, research_data, competitor_reports, insights)
            response = llm.invoke(prompt)
            self.api_key_pool.return_key(api_key, success=True)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            self.api_key_pool.return_key(api_key, success=False)
            return f"Content generation failed: {e}"

# --- Main Orchestrator ---
class SEOBlogWriter:
    """Orchestrates the entire SEO blog writing process."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_keys = APIKeyManager.load_api_keys(config.api_key_file)
        if not self.api_keys:
            logger.error("No API keys available. Cannot proceed.")
            raise SystemExit(1)
        self.api_key_pool = APIKeyPool(self.api_keys, config)
        self.search_api = GoogleSearchAPI(config)
        self.query_generator = QueryGenerator(config, self.api_key_pool)
        self.competitor_analyzer = EnhancedCompetitorAnalyzer(config, self.search_api, self.api_key_pool)
        self.researcher = EnhancedNewsAndDataResearcher(self.search_api, self.query_generator)
        self.content_generator = OriginalContentGenerator(self.api_key_pool, config)

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing process for: {blog_input.title}")
        
        # Step 1: Analyze Competitors
        competitor_reports, insights = self.competitor_analyzer.analyze_competitors(blog_input)
        
        # Step 2: Gather News and Facts
        research_data = self.researcher.gather_news_and_facts(blog_input)
        
        # Step 3: Generate Final Content
        final_content = self.content_generator.generate_content(blog_input, research_data, competitor_reports, insights)
        
        # Compile results
        result = {
            "input": blog_input,
            "competitor_insights": insights,
            "research_data": research_data,
            "final_blog": final_content,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "competitor_count": len(competitor_reports)
            }
        }
        
        logger.info("Blog writing process completed.")
        return result

# --- CLI Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="SEO Blog Writer")
    parser.add_argument("--config", required=True, help="Path to JSON config file containing an array of blog inputs")
    parser.add_argument("--output_dir", default="output_blogs", help="Directory to save generated blog posts")

    args = parser.parse_args()

    # Load config file
    try:
        config_path = Path(args.config)
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read config file {args.config}: {e}")
        raise SystemExit(1)

    # Validate it's a list
    if not isinstance(config_data, list):
        logger.error("Config file must contain a JSON array of blog objects.")
        raise SystemExit(1)

    # Convert to BlogInput objects
    try:
        blog_inputs = [BlogInput(**item) for item in config_data]
    except Exception as e:
        logger.error(f"Error parsing blog input from config: {e}")
        raise SystemExit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize the writer
    config = SEOBlogConfig()
    writer = SEOBlogWriter(config)

    # Process each blog input
    for i, blog_input in enumerate(blog_inputs, 1):
        logger.info(f"--- PROCESSING BLOG {i}/{len(blog_inputs)} ---")
        result = writer.write_blog(blog_input)
        
        # Generate a safe filename
        safe_title = re.sub(r'[^\w\-_\. ]', '_', blog_input.title)[:50]
        output_file = output_dir / f"blog_{i}_{safe_title}.json"
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Blog {i} generated and saved to {output_file}")

if __name__ == "__main__":
    main()
