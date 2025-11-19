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

import pypandoc
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.file']

# --- Enhanced Data Classes and Config ---
@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-1.5-flash-latest"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    originality_boost: bool = True
    humanization_level: str = "low"
    use_duckduckgo_fallback: bool = True
    # --- New Config Option ---
    prompt_file: str = "inputprompt.md"

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
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        
    def search(self, query: str, num_results: int = 5, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        try:
            api_url = "https://api.duckduckgo.com/"
            params = {'q': query, 'format': 'json', 'no_html': '1', 'skip_disambig': '1'}
            if self.config.verbose:
                logger.info(f"Executing DuckDuckGo Search for: {query}")
            response = self.session.get(api_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            results = []
            related_topics = data.get('RelatedTopics', [])
            for topic in related_topics[:num_results]:
                if isinstance(topic, dict) and 'FirstURL' in topic:
                    results.append({'link': topic['FirstURL'], 'snippet': topic.get('Text', ''), 'title': topic.get('Text', '')[:100]})
            if len(results) < num_results:
                results.extend(self._scrape_duckduckgo_results(query, num_results - len(results)))
            return results[:num_results]
        except Exception as e:
            logger.error(f"DuckDuckGo search error for query '{query}': {e}")
            return []
    
    def _scrape_duckduckgo_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        try:
            search_url = "https://duckduckgo.com/html/"
            params = {'q': query}
            response = self.session.get(search_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            result_divs = soup.find_all('div', class_='result')[:num_results]
            for div in result_divs:
                try:
                    link_elem = div.find('a', class_='result__a')
                    if not link_elem or not link_elem.get('href'): continue
                    link = link_elem['href']
                    title = link_elem.get_text(strip=True)
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    if link and title: results.append({'link': link, 'title': title, 'snippet': snippet})
                except Exception as e:
                    logger.debug(f"Error parsing individual result: {e}")
                    continue
            return results
        except Exception as e:
            logger.error(f"Error scraping DuckDuckGo results: {e}")
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
            logger.warning("Google Search API not properly configured. Will use DuckDuckGo search only.")

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
                if results:
                    logger.info(f"Google Search returned {len(results)} results")
                    return results
                else:
                    logger.warning("Google Search returned no results, falling back to DuckDuckGo")
            except Exception as e: 
                logger.warning(f"Google Search API error for query '{query}': {e}. Falling back to DuckDuckGo.")
        
        if self.duckduckgo_search:
            logger.info("Using DuckDuckGo search as fallback")
            return self.duckduckgo_search.search(query, num_results, sort_by_date)
        else:
            logger.error("No search engines available")
            return []

# (ContentAnalyzer, QueryGenerator, EnhancedCompetitorAnalyzer, and EnhancedNewsAndDataResearcher classes remain unchanged)
# ...
class ContentAnalyzer:
    """Advanced content analysis for competitor insights."""
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Simple readability estimation based on sentence and word length."""
        if not text.strip():
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
            
        words = re.findall(r'\w+', text)
        if not words:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple scoring: lower is more readable
        score = max(0, 100 - (avg_sentence_length * 2) - (avg_word_length * 10))
        return min(100, score)
    
    @staticmethod
    def analyze_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword density for given keywords."""
        text_lower = text.lower()
        word_count = len(re.findall(r'\w+', text))
        
        if word_count == 0:
            return {}
            
        density = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            occurrences = text_lower.count(keyword_lower)
            density[keyword] = (occurrences / word_count) * 100
            
        return density
    
    @staticmethod
    def detect_emotional_tone(text: str) -> str:
        """Simple emotional tone detection."""
        positive_words = ['amazing', 'excellent', 'great', 'wonderful', 'fantastic', 'love', 'best', 'perfect']
        negative_words = ['terrible', 'awful', 'bad', 'worst', 'hate', 'horrible', 'disappointing']
        neutral_words = ['information', 'data', 'analysis', 'research', 'study', 'report']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)
        
        total = positive_count + negative_count + neutral_count
        if total == 0:
            return "neutral"
            
        if positive_count / total > 0.4:
            return "positive"
        elif negative_count / total > 0.3:
            return "negative"
        else:
            return "neutral"
    
    @staticmethod
    def assess_content_depth(text: str, h2_count: int, h3_count: int) -> str:
        """Assess the depth of content coverage."""
        word_count = len(re.findall(r'\w+', text))
        
        # Calculate depth score based on word count and structure
        depth_score = 0
        
        if word_count > 2000:
            depth_score += 3
        elif word_count > 1000:
            depth_score += 2
        elif word_count > 500:
            depth_score += 1
            
        if h2_count > 5:
            depth_score += 2
        elif h2_count > 3:
            depth_score += 1
            
        if h3_count > 3:
            depth_score += 1
            
        if depth_score >= 5:
            return "deep"
        elif depth_score >= 3:
            return "moderate"
        else:
            return "surface"

class QueryGenerator:
    """Enhanced query generation with more sophisticated prompting."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name, 
            google_api_key=APIKeyManager.get_api_key(config.api_key_file)
        )

    def _build_prompt(self, blog_input: BlogInput) -> str:
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"""
You are an expert SEO content writer and digital marketing strategist. Your task is to write a comprehensive, high-quality, SEO-optimized blog post based on the provided requirements and competitor analysis.

**CRITICAL INSTRUCTIONS:**
- Output MUST be in valid Markdown format.
- Use proper heading hierarchy (e.g., `# Title`, `## Section`, `### Subsection`).
- Naturally integrate primary and secondary keywords. Do NOT stuff keywords.
- The content must be original, insightful, and provide significant value to the reader.
- The final output should ONLY be the markdown content of the blog post itself, starting with the main title (`#`). Do not include any preliminary text or explanation.
- Output Must contain four relevant FAQ before conclusion.

---
**BLOG POST REQUIREMENTS:**
- **Title**: {blog_input.title}
- **Brand**: {blog_input.brand}
- **Primary Keywords**: {', '.join(blog_input.primary_keywords)}
- **Secondary Keywords**: {', '.join(blog_input.secondary_keywords)}
- **Target Audience**: {blog_input.target_audience}
- **Specified Tone of Voice**: {blog_input.tone}
- **Target Word Count**: Between {self.config.min_word_count} and {self.config.max_word_count} words.

---
**TONE AND STYLE INSTRUCTIONS (OVERRIDE)**
You must follow these specific instructions for the tone and style, which are more important than the general tone listed above.

**1. NLP-Friendly Content Prompt (60% Focus):**
Create content strictly adhering to an NLP-friendly format, emphasizing clarity and simplicity in structure and language. Ensure sentences follow a straightforward subject-verb-object order, selecting words for their precision and avoiding any ambiguity. Exclude filler content, focusing on delivering information succinctly. Do not use complex or abstract terms such as 'meticulous,' 'navigating,' 'complexities,' 'realm,' 'bespoke,' 'tailored,' 'towards,' 'underpins,' 'ever-changing,' 'ever-evolving,' 'the world of,' 'not only,' 'seeking more than just,' 'designed to enhance,' 'it’s not merely,' 'our suite,' 'it is advisable,' 'daunting,' 'in the heart of,' 'when it comes to,' 'in the realm of,' 'amongst,' 'unlock the secrets,' 'unveil the secrets,' and 'robust.' This approach aims to streamline content production for enhanced NLP algorithm comprehension, ensuring the output is direct, accessible, and easily interpretable.

**2. Humanizing Prompt (40% Focus):**
While prioritizing NLP-friendly content creation (60%), also dedicate 40% of your focus to making the content engaging and enjoyable for readers, balancing technical NLP-optimization with reader satisfaction to produce content that not only ranks well on search engines well is also compelling and valuable to a readership.

---
**COMPETITOR ANALYSIS INSIGHTS:**
Here is a summary of the top-ranking competitor content. Use these insights to create a superior article that covers the topic more comprehensively and fills any identified gaps.

{analysis_summary}

---
**YOUR TASK:**
Write the full blog post now. Adhere strictly to all instructions, especially the tone and style guidelines. Ensure the post starts with a compelling introduction, has a well-structured body with actionable advice and examples, and ends with a strong conclusion. Use the competitor analysis to guide your structure, depth, and topics, but ensure your content is unique and more valuable.

**BEGIN BLOG POST:**

"""

    def generate_queries(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        logger.info("Generating enhanced search queries with LLM...")
        prompt = self._build_prompt(blog_input)
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            queries = json.loads(cleaned_response)
            logger.info(f"Successfully generated queries: {len(queries.get('news_queries',[]))} news, {len(queries.get('fact_queries',[]))} fact, {len(queries.get('unique_queries',[]))} unique")
            return queries
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}. Falling back to enhanced default queries.")
            return {
                "news_queries": [f'"{k}" latest news {datetime.now().year}' for k in blog_input.primary_keywords],
                "fact_queries": [f"what is {k}" for k in blog_input.primary_keywords],
                "unique_queries": [f"problems with {k}" for k in blog_input.primary_keywords]
            }

class EnhancedCompetitorAnalyzer:
    """Advanced competitor analysis with deeper insights."""
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI):
        self.config = config
        self.google_search = search_api
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name, 
            google_api_key=APIKeyManager.get_api_key(config.api_key_file)
        )
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
        self.content_analyzer = ContentAnalyzer()

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

    def _advanced_content_analysis(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """Perform advanced analysis on competitor content."""
        return {
            "readability_score": self.content_analyzer.calculate_readability_score(text),
            "keyword_density": self.content_analyzer.analyze_keyword_density(text, keywords),
            "emotional_tone": self.content_analyzer.detect_emotional_tone(text),
            "content_depth": self.content_analyzer.assess_content_depth(text, 0, 0)  # Will be updated with actual counts
        }

    def _identify_content_gaps_and_opportunities(self, content: str, title: str) -> Tuple[List[str], List[str]]:
        """Use LLM to identify content gaps and unique opportunities."""
        prompt = f"""
Analyze this competitor content and identify:
1. Content gaps (what important topics are missing or underexplored)
2. Unique opportunities (angles or perspectives not covered)

Title: {title}
Content (first 4000 characters): {content[:4000]}

Respond in JSON format:
{{
  "content_gaps": ["gap1", "gap2", ...],
  "unique_opportunities": ["opportunity1", "opportunity2", ...]
}}
"""
        try:
            response = self.llm.invoke(prompt)
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            analysis = json.loads(cleaned_response)
            return analysis.get("content_gaps", []), analysis.get("unique_opportunities", [])
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return [], []

    def _analyze_single_competitor(self, url: str, keywords: List[str]) -> Optional[EnhancedCompetitorData]:
        """Enhanced analysis of a single competitor URL."""
        logger.info(f"Analyzing competitor: {url}")
        soup = self._fetch_and_parse_url(url)
        if not soup: return None

        # Basic extraction
        title = soup.title.string.strip() if soup.title else "No Title Found"
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_desc['content'].strip() if meta_desc and meta_desc.get('content') else "No Meta Description Found"
        
        h1 = soup.find('h1')
        h1_text = h1.get_text(strip=True) if h1 else "No H1 Found"
        
        h2s = [h2.get_text(strip=True) for h2 in soup.find_all('h2')]
        h3s = [h3.get_text(strip=True) for h3 in soup.find_all('h3')]
        
        # Content extraction and analysis
        main_content_element = soup.find('main') or soup.find('article') or soup.body
        content_text = ' '.join(main_content_element.get_text(strip=True).split())
        word_count = len(re.findall(r'\w+', content_text))
        
        # Advanced analysis
        advanced_analysis = self._advanced_content_analysis(content_text, keywords)
        advanced_analysis["content_depth"] = self.content_analyzer.assess_content_depth(
            content_text, len(h2s), len(h3s)
        )
        
        # Content gaps and opportunities
        content_gaps, unique_opportunities = self._identify_content_gaps_and_opportunities(content_text, title)
        
        # Content structure analysis
        content_structure = {
            "has_introduction": bool(re.search(r'\b(introduction|overview|what is)\b', content_text.lower())),
            "has_conclusion": bool(re.search(r'\b(conclusion|summary|final thoughts)\b', content_text.lower())),
            "has_examples": bool(re.search(r'\b(example|for instance|such as)\b', content_text.lower())),
            "has_statistics": bool(re.search(r'\b(\d+%|\d+ percent|statistics|data shows)\b', content_text.lower())),
            "has_lists": len(soup.find_all(['ul', 'ol'])) > 0,
            "has_images": len(soup.find_all('img')) > 0
        }

        return EnhancedCompetitorData(
            url=url,
            title=title,
            meta_description=meta_description,
            h1=h1_text,
            h2s=h2s[:10],  # Limit for readability
            h3s=h3s[:15],
            word_count=word_count,
            summary=self._create_enhanced_summary(content_text, title),
            content_gaps=content_gaps,
            unique_angles=unique_opportunities,
            readability_score=advanced_analysis["readability_score"],
            keyword_density=advanced_analysis["keyword_density"],
            content_structure=content_structure,
            emotional_tone=advanced_analysis["emotional_tone"],
            content_depth=advanced_analysis["content_depth"]
        )

    def _create_enhanced_summary(self, text: str, title: str) -> str:
        """Create a more insightful summary of competitor content."""
        if not text.strip(): return "Content was empty or could not be read."
        
        prompt = f"""
Create a strategic analysis summary (80-100 words) of this competitor content.
Focus on:
1. Main content approach and strategy
2. Key strengths and weaknesses
3. Content structure and organization
4. Unique value propositions

Title: {title}
Content (first 6000 characters): {text[:6000]}

Strategic Summary:
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Enhanced summarization failed: {e}")
            return "Could not analyze content due to an API error."

    def analyze_competitors(self, blog_input: BlogInput) -> Tuple[List[EnhancedCompetitorData], ContentInsights]:
        """Enhanced competitor analysis with aggregated insights."""
        all_keywords = blog_input.primary_keywords + blog_input.secondary_keywords
        search_query = f'"{blog_input.primary_keywords[0]}"'
        logger.info(f"Finding top competitors for query: {search_query}")
        
        try:
            search_results = self.google_search.search(search_query, num_results=7)  # Get more for better analysis
            urls = [result['link'] for result in search_results if 'link' in result]
            if not urls: 
                logger.warning("No competitor URLs found.")
                return [], ContentInsights(0, [], [], [], {}, [])
                
            logger.info(f"Found {len(urls)} competitor URLs to analyze.")
            reports = []
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {
                    executor.submit(self._analyze_single_competitor, url, all_keywords): url 
                    for url in urls
                }
                for future in as_completed(future_to_url):
                    result = future.result()
                    if result: reports.append(result)

            # Generate insights
            insights = self._generate_content_insights(reports)
            
            logger.info(f"Successfully completed enhanced analysis for {len(reports)} competitors.")
            return reports, insights
            
        except Exception as e:
            logger.error(f"Error in enhanced competitor analysis: {e}", exc_info=True)
            return [], ContentInsights(0, [], [], [], {}, [])

    def _generate_content_insights(self, reports: List[EnhancedCompetitorData]) -> ContentInsights:
        """Generate aggregated insights from competitor analysis."""
        if not reports:
            return ContentInsights(0, [], [], [], {}, [])
        
        # Calculate averages
        word_counts = [r.word_count for r in reports]
        avg_word_count = int(statistics.median(word_counts)) if word_counts else 0
        
        # Identify common topics from H2s
        all_h2s = []
        for report in reports:
            all_h2s.extend([h2.lower() for h2 in report.h2s])
        
        common_topics = [topic for topic, count in Counter(all_h2s).most_common(10)]
        
        # Aggregate content gaps and opportunities
        all_gaps = []
        all_opportunities = []
        for report in reports:
            all_gaps.extend(report.content_gaps)
            all_opportunities.extend(report.unique_angles)
        
        unique_gaps = list(set(all_gaps))
        unique_opportunities = list(set(all_opportunities))
        
        # Optimal structure recommendations
        optimal_structure = {
            "recommended_word_count": max(avg_word_count + 500, 1500),  # Beat the average
            "recommended_h2_count": max(int(statistics.median([len(r.h2s) for r in reports])) + 2, 5),
            "should_include_examples": any(r.content_structure.get("has_examples", False) for r in reports),
            "should_include_statistics": any(r.content_structure.get("has_statistics", False) for r in reports),
            "should_include_images": any(r.content_structure.get("has_images", False) for r in reports)
        }
        
        # Tone recommendations
        tones = [r.emotional_tone for r in reports]
        tone_counter = Counter(tones)
        tone_recommendations = [
            f"Competitors mostly use {tone_counter.most_common(1)[0][0]} tone",
            "Consider using a more engaging/personal tone for differentiation" if tone_counter.most_common(1)[0][0] == "neutral" else "Match the emotional engagement level"
        ]
        
        return ContentInsights(
            avg_word_count=avg_word_count,
            common_topics=common_topics,
            content_gaps=unique_gaps[:10],  # Limit for usability
            unique_opportunities=unique_opportunities[:10],
            optimal_structure=optimal_structure,
            tone_recommendations=tone_recommendations
        )

class EnhancedNewsAndDataResearcher:
    """Enhanced researcher with more sophisticated query handling."""
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI):
        self.config = config
        self.google_search = search_api
        self.query_generator = QueryGenerator(config)

    def gather_news_and_facts(self, blog_input: BlogInput) -> Dict[str, List[str]]:
        """Enhanced research gathering with categorized results."""
        query_dict = self.query_generator.generate_queries(blog_input)
        news_queries = query_dict.get("news_queries", [])
        fact_queries = query_dict.get("fact_queries", [])
        unique_queries = query_dict.get("unique_queries", [])
        
        research_data = {
            "news_snippets": [],
            "fact_snippets": [],
            "unique_snippets": []
        }

        logger.info("--- Starting Enhanced News-Focused Search Phase ---")
        for query in news_queries[:3]:
            results = self.google_search.search(query, sort_by_date=True)
            for result in results:
                if result.get('snippet'):
                    research_data["news_snippets"].append(result['snippet'].strip())
            sleep(0.5)

        logger.info("--- Starting Enhanced Fact-Check Phase ---")
        for query in fact_queries[:3]:
            results = self.google_search.search(query, sort_by_date=False)
            for result in results:
                if result.get('snippet'):
                    research_data["fact_snippets"].append(result['snippet'].strip())
            sleep(0.5)

        logger.info("--- Starting Unique Perspective Phase ---")
        for query in unique_queries[:2]:
            results = self.google_search.search(query, sort_by_date=False)
            for result in results:
                if result.get('snippet'):
                    research_data["unique_snippets"].append(result['snippet'].strip())
            sleep(0.5)

        # Deduplicate within each category
        for key in research_data:
            research_data[key] = list(set(research_data[key]))

        total_snippets = sum(len(snippets) for snippets in research_data.values())
        logger.info(f"Collected {total_snippets} unique research snippets across all categories.")
        return research_data


# --- REFACTORED CONTENT GENERATION ---
class HumanizedContentGenerator:
    """Generates content by populating a customizable prompt template."""
    
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name, 
            google_api_key=APIKeyManager.get_api_key(config.api_key_file)
        )
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Loads the prompt template from the file specified in the config."""
        try:
            prompt_path = Path(self.config.prompt_file)
            logger.info(f"Loading prompt template from: {prompt_path.resolve()}")
            if not prompt_path.exists():
                logger.error(f"Prompt file not found: '{self.config.prompt_file}'. Halting execution.")
                raise FileNotFoundError(f"The specified prompt file was not found: {self.config.prompt_file}")
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Fatal error reading prompt file '{self.config.prompt_file}': {e}")
            raise

    def build_final_prompt(self, blog_input: BlogInput, research_data: Dict[str, List[str]], 
                               competitor_reports: List[EnhancedCompetitorData], 
                               insights: ContentInsights) -> str:
        """Builds the final prompt by populating the template with dynamic data."""
        
        # Format research data
        research_sections = []
        for category, snippets in research_data.items():
            if snippets:
                category_name = category.replace('_', ' ').title()
                formatted_snippets = "\n".join([f"• {snippet}" for snippet in snippets[:5]])
                research_sections.append(f"**{category_name}:**\n{formatted_snippets}")
        research_summary = "\n\n".join(research_sections) if research_sections else "No research data available."
        
        # Format competitor insights
        competitor_summary = "No competitor analysis available."
        if competitor_reports:
            competitor_insights = []
            for i, report in enumerate(competitor_reports[:3]): # Limit to top 3 for brevity
                insight = f"""
**Competitor {i+1}**: {report.title} ({report.url})
- **Approach**: {report.content_depth.title()} coverage, {report.emotional_tone} tone.
- **Structure**: {len(report.h2s)} main sections, ~{report.word_count} words.
- **Identified Gaps**: {', '.join(report.content_gaps[:2]) if report.content_gaps else 'None noted.'}
- **Summary**: {report.summary[:120]}...
"""
                competitor_insights.append(insight.strip())
            competitor_summary = "\n".join(competitor_insights)
        
        # Format strategic insights
        strategy_section = f"""
**STRATEGIC INTELLIGENCE:**
- **Target Word Count**: {insights.optimal_structure.get('recommended_word_count', 1500)}+ words
- **Identified Content Gaps to Fill**: {', '.join(insights.content_gaps[:5]) if insights.content_gaps else 'None identified'}
- **Unique Angle Opportunities**: {', '.join(insights.unique_opportunities[:3]) if insights.unique_opportunities else 'None identified'}
- **Structural Advice**: Aim for {insights.optimal_structure.get('recommended_h2_count', 5)}+ H2 sections. Consider including examples and statistics.
"""
        
        # Populate the template using .format()
        return self.prompt_template.format(
            title=blog_input.title,
            brand=blog_input.brand,
            primary_keywords=blog_input.primary_keywords,
            secondary_keywords=blog_input.secondary_keywords,
            target_audience=blog_input.target_audience,
            tone=blog_input.tone,
            unique_angle=blog_input.unique_angle or "Find a fresh, unaddressed perspective.",
            strategy_section=strategy_section,
            competitor_summary=competitor_summary,
            research_summary=research_summary
        )

# --- Enhanced Main Writer Class ---
class EnhancedSEOBlogWriter:
    """Orchestrates the blog writing process."""
    
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search_api = GoogleSearchAPI(config)
        self.researcher = EnhancedNewsAndDataResearcher(config, self.google_search_api)
        self.competitor_analyzer = EnhancedCompetitorAnalyzer(config, self.google_search_api)
        self.content_generator = HumanizedContentGenerator(config)

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """Enhanced blog writing process with external prompt template."""
        logger.info(f"Starting enhanced blog writing process for '{blog_input.title}'")
        
        try:
            logger.info("[Phase 1/5] Enhanced Competitor Analysis...")
            competitor_reports, insights = self.competitor_analyzer.analyze_competitors(blog_input)
            
            logger.info("[Phase 2/5] Multi-dimensional Research Gathering...")
            research_data = self.researcher.gather_news_and_facts(blog_input)
            
            logger.info("[Phase 3/5] Building Humanized Content Strategy...")
            sleep(random.uniform(1, 3))
            
            logger.info("[Phase 4/5] Building final prompt from template...")
            prompt = self.content_generator.build_final_prompt(
                blog_input, research_data, competitor_reports, insights
            )
            
            logger.info("[Phase 5/5] Generating content with LLM...")
            response = self.content_generator.llm.invoke(prompt)
            blog_content = response.content
            
            blog_content = self._post_process_content(blog_content, blog_input)
            metadata = self._generate_content_metadata(competitor_reports, insights, research_data)
            
            return {
                "success": True, 
                "blog_content": blog_content,
                "metadata": metadata,
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced blog writing process: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _post_process_content(self, content: str, blog_input: BlogInput) -> str:
        """Post-process content to enhance humanization and originality."""
        transitions = [
            ("However,", "But here's the thing,"), ("Additionally,", "What's more,"),
            ("Furthermore,", "On top of that,"), ("In conclusion,", "So, what's the bottom line?"),
            ("It is important to note", "Worth mentioning"), ("It should be noted", "Keep in mind")
        ]
        processed_content = content
        for formal, casual in transitions:
            processed_content = processed_content.replace(formal, casual)
        return processed_content

    def _generate_content_metadata(self, reports: List[EnhancedCompetitorData], insights: ContentInsights, data: Dict) -> Dict:
        """Generate metadata about the content creation process."""
        return {
            "competitors_analyzed": len(reports),
            "avg_competitor_word_count": insights.avg_word_count,
            "research_snippets_used": sum(len(v) for v in data.values()),
            "content_gaps_addressed": len(insights.content_gaps),
            "unique_opportunities_leveraged": len(insights.unique_opportunities),
            "generation_timestamp": datetime.now().isoformat(),
            "search_engine_used": "Google + DuckDuckGo" if self.google_search_api.google_available else "DuckDuckGo only"
        }

# --- Google Drive and File Conversion Utilities ---
def authenticate_google_api():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if not os.path.exists('credentials.json'):
            logger.error("Google Drive upload requires 'credentials.json'. Get it from Google Cloud Console.")
            return None
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def convert_md_to_odt(md_file_path: Path) -> Path:
    odt_file = md_file_path.with_suffix('.odt')
    try:
        pypandoc.convert_file(str(md_file_path), 'odt', outputfile=str(odt_file))
        logger.info(f"Successfully converted {md_file_path.name} to {odt_file.name}")
        return odt_file
    except Exception as e:
        logger.error(f"Failed to convert markdown to ODT: {e}")
        raise

def upload_to_drive(odt_file_path: Path, title: str, drive_service):
    file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
    media = MediaFileUpload(str(odt_file_path), mimetype='application/vnd.oasis.opendocument.text')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
    logger.info(f"✅ Successfully uploaded to Google Drive as '{title}'.")
    logger.info(f"   View online: {file.get('webViewLink')}")

# --- Enhanced Argument Parsing and Main Functions ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts using a customizable prompt.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file with blog input data.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file.')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--drive', action='store_true', help='Upload the final blog post to Google Drive.')
    # --- New Argument ---
    parser.add_argument('--prompt-file', type=str, default='inputprompt.md', help='Path to the markdown file for the main prompt template.')
    return parser.parse_args()

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        blog_inputs = [BlogInput(**item) for item in data]
        return blog_inputs
    except Exception as e:
        logger.error(f"Error loading config file {filepath}: {e}")
        raise

def save_enhanced_blog_to_file(blog_content: str, blog_input: BlogInput, metadata: Dict[str, Any]) -> Path:
    safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')
    content_filename = Path(f"{safe_title[:100]}.md")
    content_filename.write_text(blog_content, encoding='utf-8')
    metadata_filename = Path(f"{safe_title[:100]}_metadata.json")
    metadata_filename.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info(f"Blog post saved to: {content_filename.resolve()}")
    logger.info(f"Metadata saved to: {metadata_filename.resolve()}")
    return content_filename

def main():
    args = parse_arguments()
    drive_service = None
    if args.drive:
        try:
            logger.info("Authenticating with Google Drive...")
            creds = authenticate_google_api()
            if creds:
                drive_service = build('drive', 'v3', credentials=creds)
                logger.info("Google Drive authentication successful.")
            else:
                logger.error("Could not build Google Drive service. Uploads will be skipped.")
        except Exception as e:
            logger.error(f"An error during Google Drive authentication: {e}")

    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            verbose=args.verbose,
            prompt_file=args.prompt_file
        )
        writer = EnhancedSEOBlogWriter(config)
        
        if writer.google_search_api.google_available:
            print("🔍 Search Engine Status: Google Search API (with DuckDuckGo fallback)")
        else:
            print("🔍 Search Engine Status: DuckDuckGo Search only")
        
        for i, blog_input in enumerate(blog_inputs, 1):
            logger.info(f"\n{'='*60}\nProcessing blog {i}/{len(blog_inputs)}: {blog_input.title}\n{'='*60}")
            result = writer.write_blog(blog_input)
            
            if result and result.get('success'):
                md_file_path = save_enhanced_blog_to_file(result['blog_content'], blog_input, result.get('metadata', {}))
                metadata = result.get('metadata', {})
                print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' completed!")
                print(f"   - Competitors analyzed: {metadata.get('competitors_analyzed', 0)}")
                print(f"   - Research snippets used: {metadata.get('research_snippets_used', 0)}")
                print(f"   - Content gaps addressed: {metadata.get('content_gaps_addressed', 0)}")
                
                if args.drive and drive_service:
                    try:
                        logger.info("Starting Google Drive upload process...")
                        odt_file_path = convert_md_to_odt(md_file_path)
                        upload_to_drive(odt_file_path, blog_input.title, drive_service)
                        os.remove(odt_file_path)
                    except Exception as e:
                        logger.error(f"Google Drive upload failed for '{blog_input.title}': {e}")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                print(f"\n❌ ERROR: Could not generate blog post for '{blog_input.title}'. Details: {error_msg}")
            
            if i < len(blog_inputs):
                logger.info("Pausing before next blog post...")
                sleep(2)
    except Exception as e:
        logger.error(f"Critical error in main process: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
