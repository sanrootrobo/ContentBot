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

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enhanced Data Classes and Config ---
@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.5-pro" # Using a powerful model for better JSON output and reasoning
    timeout: int = 20 # Increased timeout for potentially longer API calls
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    originality_boost: bool = True
    humanization_level: str = "high"
    use_duckduckgo_fallback: bool = True

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

# --- Utility Classes (Previously Omitted) ---
class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists(): 
                logger.error(f"API key file not found: '{filepath}'")
                return None
            with open(key_path, 'r', encoding='utf-8') as f: 
                api_key = f.read().strip()
            if not api_key or len(api_key) < 10: 
                logger.error(f"Invalid API key in {filepath}")
                return None
            return api_key
        except Exception as e: 
            logger.error(f"Error reading API key: {e}")
            return None

class DuckDuckGoSearchAPI:
    """DuckDuckGo search implementation as fallback."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        
    def search(self, query: str, num_results: int = 5, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo's web scraping (HTML version)."""
        try:
            search_url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            if self.config.verbose:
                date_info = " (sorted by date)" if sort_by_date else ""
                logger.info(f"Executing DuckDuckGo Search{date_info} for: {query}")

            response = self.session.get(search_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            result_divs = soup.find_all('div', class_='result')[:num_results]
            
            for div in result_divs:
                try:
                    link_elem = div.find('a', class_='result__a')
                    if not link_elem or not link_elem.get('href'): continue
                    
                    # Clean up the link
                    raw_link = link_elem['href']
                    parsed_link = urlparse(raw_link)
                    cleaned_link = parsed_link.query.split('uddg=')[1].split('&')[0]
                    link = requests.utils.unquote(cleaned_link)

                    title = link_elem.get_text(strip=True)
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if link and title:
                        results.append({'link': link, 'title': title, 'snippet': snippet})
                        
                except Exception as e:
                    logger.debug(f"Error parsing individual DDG result: {e}")
                    continue
            
            return results
            
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
                log_msg = f"Executing Google Search for: {query}"
                if sort_by_date:
                    params['sort'] = 'date'
                    log_msg = f"Executing Google News Search (sorted by date) for: {query}"
                if self.config.verbose: logger.info(log_msg)
                
                response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()
                results = data.get('items', [])
                
                if results:
                    logger.info(f"Google Search returned {len(results)} results")
                    return results
                else:
                    logger.warning("Google Search returned no results, falling back to DuckDuckGo")
            except Exception as e: 
                logger.warning(f"Google Search API error for '{query}': {e}. Falling back to DuckDuckGo.")
        
        if self.duckduckgo_search:
            logger.info("Using DuckDuckGo search as fallback")
            return self.duckduckgo_search.search(query, num_results, sort_by_date)
        else:
            logger.error("No search engines available.")
            return []

class ContentAnalyzer:
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        if not text.strip(): return 0.0
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences: return 0.0
        words = re.findall(r'\w+', text)
        if not words: return 0.0
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        score = max(0, 100 - (avg_sentence_length * 2) - (avg_word_length * 10))
        return min(100, score)

    @staticmethod
    def analyze_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
        text_lower = text.lower()
        word_count = len(re.findall(r'\w+', text))
        if word_count == 0: return {}
        density = {kw: (text_lower.count(kw.lower()) / word_count) * 100 for kw in keywords}
        return density

    @staticmethod
    def assess_content_depth(text: str, h2_count: int, h3_count: int) -> str:
        word_count = len(re.findall(r'\w+', text))
        depth_score = 0
        if word_count > 2000: depth_score += 3
        elif word_count > 1000: depth_score += 2
        elif word_count > 500: depth_score += 1
        if h2_count > 5: depth_score += 2
        elif h2_count > 3: depth_score += 1
        if h3_count > 3: depth_score += 1
        if depth_score >= 5: return "deep"
        elif depth_score >= 3: return "moderate"
        else: return "surface"

class QueryGenerator:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=APIKeyManager.get_api_key(config.api_key_file))

    def generate_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        logger.info("Generating search queries with LLM...")
        prompt = f"""Generate diverse search queries for a blog post titled "{blog_input.title}" with primary keywords: {blog_input.primary_keywords}. Create three types: 'news_queries' for recent trends, 'fact_queries' for foundational info, and 'unique_queries' for alternative angles. Respond with ONLY a valid JSON object like: {{"news_queries": [], "fact_queries": [], "unique_queries": []}}"""
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            queries = json.loads(cleaned_response)
            return queries
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}. Using default queries.")
            return {
                "news_queries": [f'"{k}" latest news {datetime.now().year}' for k in blog_input.primary_keywords],
                "fact_queries": [f"what is {k}" for k in blog_input.primary_keywords],
                "unique_queries": [f"disadvantages of {k}" for k in blog_input.primary_keywords]
            }

class EnhancedCompetitorAnalyzer:
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI):
        self.config = config
        self.google_search = search_api
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=APIKeyManager.get_api_key(config.api_key_file))
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        self.content_analyzer = ContentAnalyzer()

    def _fetch_and_parse_url(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser') if 'text/html' in response.headers.get('Content-Type', '') else None
        except requests.RequestException as e:
            logger.warning(f"Analysis failed for {url}. Reason: {e}")
            return None

    def _identify_content_gaps_and_opportunities(self, content: str, title: str) -> Tuple[List[str], List[str]]:
        prompt = f"""Analyze this competitor content. Identify content gaps (missing topics) and unique opportunities (fresh angles). Title: {title}\nContent (first 4000 chars): {content[:4000]}\nRespond in JSON: {{"content_gaps": [], "unique_opportunities": []}}"""
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            analysis = json.loads(cleaned_response)
            return analysis.get("content_gaps", []), analysis.get("unique_opportunities", [])
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return [], []

    def _analyze_single_competitor(self, url: str, keywords: List[str]) -> Optional[EnhancedCompetitorData]:
        logger.info(f"Analyzing competitor: {url}")
        soup = self._fetch_and_parse_url(url)
        if not soup: return None

        title = soup.title.string.strip() if soup.title else "No Title"
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        main_content_element = soup.find('main') or soup.find('article') or soup.body
        content_text = ' '.join(main_content_element.get_text(strip=True).split())
        word_count = len(re.findall(r'\w+', content_text))
        h2s = [h2.get_text(strip=True) for h2 in soup.find_all('h2')]
        content_gaps, unique_opportunities = self._identify_content_gaps_and_opportunities(content_text, title)
        
        return EnhancedCompetitorData(
            url=url, title=title,
            meta_description=meta_desc['content'].strip() if meta_desc and meta_desc.get('content') else "",
            h1=soup.find('h1').get_text(strip=True) if soup.find('h1') else "",
            h2s=h2s, h3s=[h3.get_text(strip=True) for h3 in soup.find_all('h3')],
            word_count=word_count,
            summary=self._create_enhanced_summary(content_text, title),
            content_gaps=content_gaps, unique_angles=unique_opportunities,
            content_depth=self.content_analyzer.assess_content_depth(content_text, len(h2s), 0)
        )

    def _create_enhanced_summary(self, text: str, title: str) -> str:
        prompt = f"""Create a strategic analysis summary (80-100 words) of this competitor content. Focus on their main strategy, strengths, and weaknesses. Title: {title}\nContent (first 6000 chars): {text[:6000]}\nStrategic Summary:"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.error(f"Enhanced summarization failed: {e}")
            return "Could not analyze content due to an API error."

    def analyze_competitors(self, blog_input: BlogInput) -> Tuple[List[EnhancedCompetitorData], ContentInsights]:
        search_query = f'"{blog_input.primary_keywords[0]}"'
        logger.info(f"Finding top competitors for query: {search_query}")
        search_results = self.google_search.search(search_query, num_results=5)
        urls = [result['link'] for result in search_results if 'link' in result]
        if not urls: return [], self._generate_content_insights([])
        
        reports = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._analyze_single_competitor, url, blog_input.primary_keywords): url for url in urls}
            for future in as_completed(future_to_url):
                if result := future.result(): reports.append(result)
        
        insights = self._generate_content_insights(reports)
        logger.info(f"Completed analysis for {len(reports)} competitors.")
        return reports, insights

    def _generate_content_insights(self, reports: List[EnhancedCompetitorData]) -> ContentInsights:
        if not reports: return ContentInsights(0, [], [], [], {}, [])
        word_counts = [r.word_count for r in reports]
        all_h2s = [h2.lower() for r in reports for h2 in r.h2s]
        all_gaps = [gap for r in reports for gap in r.content_gaps]
        all_opportunities = [opp for r in reports for opp in r.unique_angles]
        
        return ContentInsights(
            avg_word_count=int(statistics.median(word_counts)) if word_counts else 1200,
            common_topics=[topic for topic, count in Counter(all_h2s).most_common(10)],
            content_gaps=list(set(all_gaps))[:10],
            unique_opportunities=list(set(all_opportunities))[:10],
            optimal_structure={
                "recommended_word_count": max(int(statistics.median(word_counts)) + 300, 1500) if word_counts else 1500,
                "recommended_h2_count": max(int(statistics.median([len(r.h2s) for r in reports])) + 2, 5),
            },
            tone_recommendations=[]
        )

class EnhancedNewsAndDataResearcher:
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI):
        self.config = config
        self.google_search = search_api
        self.query_generator = QueryGenerator(config)

    def gather_news_and_facts(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        query_dict = self.query_generator.generate_queries(blog_input)
        research_data = {"news_snippets": [], "fact_snippets": [], "unique_snippets": []}
        
        for category, queries in [("news_snippets", query_dict.get("news_queries", [])), ("fact_snippets", query_dict.get("fact_queries", [])), ("unique_snippets", query_dict.get("unique_queries", []))]:
            for query in queries[:2]:
                results = self.google_search.search(query, sort_by_date=(category=="news_snippets"))
                for result in results:
                    if snippet := result.get('snippet'): research_data[category].append(snippet.strip())
                sleep(0.5)
        
        for key in research_data: research_data[key] = list(set(research_data[key]))
        logger.info(f"Collected {sum(len(v) for v in research_data.values())} unique research snippets.")
        return research_data

# --- Enhanced Content Generation (With SEO & JSON output) ---
class HumanizedContentGenerator:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=APIKeyManager.get_api_key(config.api_key_file))
    
    def _build_humanized_prompt(self, blog_input: BlogInput, research_data: Dict[str, List[str]], 
                               competitor_reports: List[EnhancedCompetitorData], 
                               insights: ContentInsights) -> str:
        research_summary = "\n".join([f"**{k.replace('_', ' ').title()}:**\n" + "\n".join([f"• {s}" for s in v[:5]]) for k, v in research_data.items() if v])
        competitor_summary = "\n".join([f"**Competitor {i+1} ({r.url})**:\n- **Strategy**: {r.summary[:150]}...\n- **Content Gaps**: {', '.join(r.content_gaps[:3]) if r.content_gaps else 'None'}" for i, r in enumerate(competitor_reports[:3])])
        strategy_section = f"""**STRATEGIC INTELLIGENCE:**\n- **Target Word Count**: {insights.optimal_structure.get('recommended_word_count', 1500)}+ words.\n- **Content Gaps to Fill**: {', '.join(insights.content_gaps[:5]) if insights.content_gaps else 'None'}.\n- **Unique Opportunities**: {', '.join(insights.unique_opportunities[:3]) if insights.unique_opportunities else 'None'}.\n- **Recommended Structure**: {insights.optimal_structure.get('recommended_h2_count', 5)}+ H2 sections."""
        seo_instructions = f"""**ON-PAGE SEO & E-E-A-T REQUIREMENTS:**\n1. **Keyword Strategy:**\n    *   **Primary Keyword (`{blog_input.primary_keywords[0]}`):** MUST appear in the SEO Title, Meta Description, the first H1, and within the first 100 words. Use it naturally in 2-3 H2 headings.\n    *   **Secondary Keywords ({', '.join(blog_input.secondary_keywords)}):** Integrate naturally into subheadings and body content.\n2. **E-E-A-T:**\n    *   Write from a position of authority.\n    *   Include placeholders for internal links: `[Internal Link: A related topic post]`.\n    *   Suggest 1-2 external links to high-authority, non-competing sources (e.g., Wikipedia, .gov sites, research papers).\n3. **Structure:**\n    *   Use short paragraphs. End with a clear conclusion, a Call-to-Action (CTA), and a 3-4 point FAQ section."""
        humanization_instructions = f"""**WRITING STYLE:**\n*   **Originality:** Generate content with a unique perspective. Use research to find fresh angles.\n*   **Human Tone:** Write in a conversational, engaging, `{blog_input.tone}` tone. Use active voice.\n*   **Forbidden Phrases:** DO NOT use cliche AI phrases like "delve into," "in conclusion," "the digital landscape," or "it's crucial to." """

        return f"""You are an expert SEO content strategist. Your mission is to create a comprehensive, original, and highly optimized blog post.
**1. SPECS:**
- **Topic**: {blog_input.title}
- **Primary Keyword**: {blog_input.primary_keywords[0]}
- **Audience**: {blog_input.target_audience}
- **Brand Voice**: {blog_input.brand}

**2. ANALYSIS & RESEARCH:**
{strategy_section}
**Competitor Insights:**
{competitor_summary}
**Research Data:**
{research_summary}

**3. EXECUTION:**
{seo_instructions}
{humanization_instructions}

**4. REQUIRED OUTPUT FORMAT:**
You MUST respond with a single, valid JSON object and nothing else.
{{
  "seo_title": "A 50-60 character meta title with the primary keyword.",
  "meta_description": "A 150-160 character meta description that is compelling and includes the primary keyword.",
  "slug": "a-url-friendly-slug-for-the-post",
  "blog_content": "The full blog post content in Markdown format, following all instructions. This includes headings, paragraphs, lists, FAQ, CTA, etc."
}}

Generate the JSON output now.
"""

# --- Main Writer Class ---
class EnhancedSEOBlogWriter:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search_api = GoogleSearchAPI(config)
        self.researcher = EnhancedNewsAndDataResearcher(config, self.google_search_api)
        self.competitor_analyzer = EnhancedCompetitorAnalyzer(config, self.google_search_api)
        self.content_generator = HumanizedContentGenerator(config)

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing process for '{blog_input.title}'")
        try:
            logger.info("[Phase 1/4] Competitor Analysis...")
            competitor_reports, insights = self.competitor_analyzer.analyze_competitors(blog_input)
            
            logger.info("[Phase 2/4] Research Gathering...")
            research_data = self.researcher.gather_news_and_facts(blog_input)
            
            logger.info("[Phase 3/4] Building Content Prompt...")
            prompt = self.content_generator._build_humanized_prompt(blog_input, research_data, competitor_reports, insights)
            
            logger.info("[Phase 4/4] Generating Content...")
            response = self.content_generator.llm.invoke(prompt)
            
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            content_data = json.loads(cleaned_response)

            metadata = self._generate_content_metadata(competitor_reports, insights, research_data)
            
            return {"success": True, "content_data": content_data, "metadata": metadata}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: {e}")
            logger.debug(f"Raw response was: {response.content}")
            return {"success": False, "error": "Failed to parse LLM JSON response."}
        except Exception as e:
            logger.error(f"Error in blog writing process: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _generate_content_metadata(self, reports, insights, research) -> Dict[str, Any]:
        return {
            "competitors_analyzed": len(reports),
            "avg_competitor_word_count": insights.avg_word_count,
            "research_snippets_used": sum(len(s) for s in research.values()),
            "generation_timestamp": datetime.now().isoformat(),
            "model_used": self.config.model_name
        }

# --- Argument Parsing and Main Functions (Previously Omitted) ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate enhanced SEO-optimized blog posts.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file with blog input data.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file.')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
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

def save_enhanced_blog_to_file(content_data: Dict[str, str], blog_input: BlogInput, metadata: Dict[str, Any]):
    slug = content_data.get('slug', re.sub(r'[\s\W]+', '-', blog_input.title.lower()).strip('-'))
    safe_title = slug[:100]
    
    # Save markdown content
    content_filename = Path(f"{safe_title}.md")
    content_filename.write_text(content_data['blog_content'], encoding='utf-8')
    
    # Save SEO data and metadata
    output_data = {
        "title": blog_input.title,
        "seo_title": content_data.get('seo_title'),
        "meta_description": content_data.get('meta_description'),
        "slug": slug,
        "generation_metadata": metadata,
        "blog_content_markdown_file": str(content_filename)
    }
    metadata_filename = Path(f"{safe_title}_seo_data.json")
    metadata_filename.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
    
    logger.info(f"Blog post saved to: {content_filename.resolve()}")
    logger.info(f"SEO data saved to: {metadata_filename.resolve()}")

def main():
    args = parse_arguments()
    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            verbose=args.verbose,
            use_duckduckgo_fallback=not args.no_duckduckgo_fallback
        )
        writer = EnhancedSEOBlogWriter(config)
        
        for i, blog_input in enumerate(blog_inputs, 1):
            logger.info(f"\n{'='*60}\nProcessing blog {i}/{len(blog_inputs)}: {blog_input.title}\n{'='*60}")
            result = writer.write_blog(blog_input)
            
            if result and result.get('success'):
                save_enhanced_blog_to_file(result['content_data'], blog_input, result.get('metadata', {}))
                metadata = result.get('metadata', {})
                print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' completed!")
                print(f"   📊 Competitors analyzed: {metadata.get('competitors_analyzed', 0)}")
                print(f"   📝 Avg competitor word count: {metadata.get('avg_competitor_word_count', 0)}")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                print(f"\n❌ ERROR: Could not generate blog post for '{blog_input.title}'.\n   💥 Details: {error_msg}")
            
            if i < len(blog_inputs):
                logger.info("Pausing before next blog post...")
                sleep(5) # Increased delay to be respectful to APIs
    
    except Exception as e:
        logger.error(f"Critical error in main process: {e}", exc_info=True)
        print(f"\n💥 CRITICAL ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
