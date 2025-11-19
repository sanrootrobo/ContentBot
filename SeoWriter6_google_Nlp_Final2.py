from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class SEOBlogConfig:
    """Configuration for the SEO blog writer."""
    model_name: str = "gemini-1.5-pro-latest"
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 2.0  # Initial delay for backoff
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )

@dataclass
class APIKeys:
    """Container for API keys."""
    gemini: Optional[str] = None
    Google Search: Optional[str] = None
    google_cx: Optional[str] = None

@dataclass
class BlogInput:
    """Input data for generating a blog post."""
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
            raise ValueError("Brand and title are required.")
        if not self.primary_keywords:
            raise ValueError("At least one primary keyword is required.")

@dataclass
class CompetitorData:
    """Structured data for a single competitor's analysis."""
    url: str
    title: str
    h1: Optional[str]
    h2s: List[str]
    word_count: int
    readability_score: float  # Flesch Reading Ease
    summary: str
    analysis: Dict[str, Any] = field(default_factory=dict) # Holds gaps, tone, etc.

@dataclass
class ContentInsights:
    """Aggregated insights from all competitor analyses."""
    median_word_count: int
    common_topics: List[str]
    aggregated_gaps: List[str]
    aggregated_opportunities: List[str]
    strategic_recommendations: Dict[str, Any]
```

-----

### `utils.py`

Contains helper functions, including the improved API key manager and the new retry decorator.

```python
# utils.py
"""Utility functions for API key management, retries, and file operations."""

import os
import time
import logging
import functools
from pathlib import Path
from typing import Optional, Callable, Any

from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
import requests
import textstat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_api_key(env_var: str, file_path: str) -> Optional[str]:
    """
    Get an API key from an environment variable or a file.
    The environment variable is prioritized.
    """
    key = os.getenv(env_var)
    if key:
        logger.info(f"Loaded API key from environment variable '{env_var}'.")
        return key

    try:
        key_path = Path(file_path)
        if key_path.exists():
            key = key_path.read_text().strip()
            if key:
                logger.info(f"Loaded API key from file: '{file_path}'.")
                return key
    except Exception as e:
        logger.error(f"Error reading API key from {file_path}: {e}")

    logger.warning(f"API key not found in environment variable '{env_var}' or file '{file_path}'.")
    return None

def retry_with_backoff(
    retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0
) -> Callable:
    """
    A decorator for retrying a function with exponential backoff.
    Handles common transient API errors.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except (ResourceExhausted, GoogleAPICallError, requests.exceptions.RequestException) as e:
                    logger.warning(f"API error in '{func.__name__}': {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            logger.error(f"Function '{func.__name__}' failed after {retries} retries.")
            raise  # Re-raise the last exception
        return wrapper
    return decorator

def calculate_readability(text: str) -> float:
    """Calculates Flesch Reading Ease score using textstat."""
    if not text or not text.strip():
        return 0.0
    return textstat.flesch_reading_ease(text)

def sanitize_filename(name: str) -> str:
    """Removes invalid characters from a string to make it a valid filename."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
```

-----

### `services.py`

This file contains the core logic of the application, now organized into focused classes.

```python
# services.py
"""Core service classes for search, analysis, and content generation."""

import logging
import json
from datetime import datetime
from collections import Counter
import statistics
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    SEOBlogConfig, BlogInput, APIKeys, CompetitorData, ContentInsights
)
from utils import retry_with_backoff, calculate_readability, logger

class GoogleSearchAPI:
    """Wrapper for the Google Custom Search API."""
    BASE_URL = "[https://www.googleapis.com/customsearch/v1](https://www.googleapis.com/customsearch/v1)"

    def __init__(self, api_keys: APIKeys, config: SEOBlogConfig):
        if not api_keys.Google Search or not api_keys.google_cx:
            raise ValueError("Google Search API key and CX ID are required.")
        self.api_key = api_keys.Google Search
        self.cx_id = api_keys.google_cx
        self.config = config

    @retry_with_backoff()
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        params = {'key': self.api_key, 'cx': self.cx_id, 'q': query, 'num': num_results}
        logger.info(f"Executing search for: '{query}'")
        response = requests.get(self.BASE_URL, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json().get('items', [])

class CompetitorAnalyzer:
    """Analyzes competitor content from URLs."""

    def __init__(self, config: SEOBlogConfig, llm: ChatGoogleGenerativeAI):
        self.config = config
        self.llm = llm
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.config.user_agent})

    @retry_with_backoff()
    def _fetch_page(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return response.text
            logger.warning(f"Skipping non-HTML content at {url}")
            return None
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    @retry_with_backoff()
    def _analyze_content_with_llm(self, text: str, title: str) -> Dict[str, Any]:
        """Uses LLM to perform a consolidated analysis of content."""
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the following article content based on its title.
            Provide a strategic analysis covering the following points:
            1.  **summary**: A concise 100-word summary of the main arguments and approach.
            2.  **emotional_tone**: The primary emotional tone (e.g., 'Informative', 'Urgent', 'Inspirational').
            3.  **content_gaps**: List 2-3 specific topics or questions the article failed to address.
            4.  **unique_angles**: List 1-2 unique perspectives or insights the article offers.

            **Title**: {title}
            **Content (first 4000 chars)**: {content}

            {format_instructions}
            """,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.llm | parser
        return chain.invoke({"title": title, "content": text[:4000]})

    def analyze_single_competitor(self, url: str) -> Optional[CompetitorData]:
        logger.info(f"Analyzing competitor: {url}")
        html = self._fetch_page(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        content_text = ' '.join(main_content.get_text(strip=True).split()) if main_content else ""

        if not content_text:
            logger.warning(f"No content found for {url}")
            return None

        title = soup.title.string.strip() if soup.title else "No Title"
        word_count = len(content_text.split())

        try:
            llm_analysis = self._analyze_content_with_llm(content_text, title)
        except Exception as e:
            logger.error(f"LLM analysis failed for {url}: {e}")
            llm_analysis = {"summary": "Analysis failed.", "emotional_tone": "Unknown", "content_gaps": [], "unique_angles": []}

        return CompetitorData(
            url=url,
            title=title,
            h1=soup.h1.get_text(strip=True) if soup.h1 else None,
            h2s=[h.get_text(strip=True) for h in soup.find_all('h2', limit=10)],
            word_count=word_count,
            readability_score=calculate_readability(content_text),
            summary=llm_analysis.get("summary", ""),
            analysis=llm_analysis
        )

    def run_analysis(self, blog_input: BlogInput, search_api: GoogleSearchAPI) -> Tuple[List[CompetitorData], ContentInsights]:
        """Orchestrates the full competitor analysis process."""
        query = f'"{blog_input.primary_keywords[0]}"'
        results = search_api.search(query, num_results=7)
        urls = [r['link'] for r in results if 'link' in r]

        if not urls:
            logger.warning("No competitor URLs found.")
            return [], self._generate_insights([])

        reports = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.analyze_single_competitor, url): url for url in urls}
            for future in as_completed(future_to_url):
                report = future.result()
                if report:
                    reports.append(report)

        logger.info(f"Successfully analyzed {len(reports)} competitors.")
        insights = self._generate_insights(reports)
        return reports, insights

    def _generate_insights(self, reports: List[CompetitorData]) -> ContentInsights:
        if not reports:
            return ContentInsights(0, [], [], [], {})

        word_counts = [r.word_count for r in reports]
        all_h2s = [h2.lower() for r in reports for h2 in r.h2s]
        all_gaps = [gap for r in reports for gap in r.analysis.get("content_gaps", [])]
        all_opportunities = [angle for r in reports for angle in r.analysis.get("unique_angles", [])]

        strategic_recommendations = {
            "target_word_count": int(statistics.median(word_counts) + 300) if word_counts else 1500,
            "target_readability_score": "Aim for a Flesch score of 60+ (easily understood by 13-15 year olds).",
            "common_competitor_tones": list(Counter(r.analysis.get("emotional_tone") for r in reports).keys()),
        }

        return ContentInsights(
            median_word_count=int(statistics.median(word_counts)) if word_counts else 0,
            common_topics=[topic for topic, count in Counter(all_h2s).most_common(8)],
            aggregated_gaps=list(set(all_gaps))[:8],
            aggregated_opportunities=list(set(all_opportunities))[:5],
            strategic_recommendations=strategic_recommendations
        )

class ContentGenerator:
    """Generates the final blog post content."""

    def __init__(self, config: SEOBlogConfig, llm: ChatGoogleGenerativeAI):
        self.config = config
        self.llm = llm

    @retry_with_backoff(retries=2)
    def generate_blog_post(self, blog_input: BlogInput, insights: ContentInsights, competitor_reports: List[CompetitorData]) -> str:
        logger.info("Generating final blog post...")
        prompt_template = self._build_prompt()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        parser = StrOutputParser()
        chain = prompt | self.llm | parser

        # Prepare context strings
        competitor_summary = "\n".join(
            [f"- **{r.title}**: {r.summary}" for r in competitor_reports[:3]]
        )

        return chain.invoke({
            "blog_input": blog_input,
            "insights": insights,
            "competitor_summary": competitor_summary,
            "current_date": datetime.now().strftime("%B %d, %Y")
        })

    def _build_prompt(self) -> str:
        """Constructs the master prompt for content generation."""
        return """
        You are an expert SEO content strategist and world-class writer. Your mission is to create a blog post that is more insightful, original, and valuable than any existing content.

        **CRITICAL WRITING INSTRUCTIONS:**
        1.  **Originality is Paramount**: Do NOT rephrase competitor content. Synthesize ideas and introduce a truly unique perspective based on the provided "Unique Angle" and "Aggregated Opportunities".
        2.  **Human-Like Tone**: Write in a natural, conversational style. Use contractions (it's, you're). Vary sentence length. Ask rhetorical questions. Use personal anecdotes if `personal_experience` is true.
        3.  **Avoid AI Clichés**: DO NOT use phrases like "In today's digital age...", "delve into", "the world of", "it's crucial to", "unlock the potential", "a deep dive".
        4.  **Value-Driven Content**: Focus on providing actionable advice, clear explanations, and real-world examples.
        5.  **Structure**: Start with a strong hook. Logically structure the article with Markdown headings (H2s and H3s). End with a concise conclusion and a relevant FAQ section.

        ---
        **CONTEXT & STRATEGY BRIEF**

        **Blog Post Details**:
        - Title: {blog_input.title}
        - Brand: {blog_input.brand}
        - Target Audience: {blog_input.target_audience}
        - Desired Tone: {blog_input.tone}
        - Unique Angle to Emphasize: {blog_input.unique_angle}
        - Include Personal Experience: {blog_input.personal_experience}
        - Keywords: {blog_input.primary_keywords}, {blog_input.secondary_keywords}

        **Strategic Insights from Competitor Analysis**:
        - **Target Word Count**: ~{insights.strategic_recommendations.target_word_count} words.
        - **Content Gaps to Fill**: {insights.aggregated_gaps}
        - **Unique Opportunities to Seize**: {insights.aggregated_opportunities}
        - **Common Topics to Cover (but better)**: {insights.common_topics}

        **Summary of Top Competitors**:
        {competitor_summary}

        ---
        **YOUR TASK**

        Write the complete, final blog post in Markdown format. Ensure it meets all the criteria above.
        The post should be ready for publication. Start directly with the content.

        BEGIN BLOG POST:
        """

class SEOBlogWriter:
    """Orchestrator for the entire blog writing process."""

    def __init__(self, config: SEOBlogConfig, api_keys: APIKeys):
        self.config = config
        self.api_keys = api_keys
        if not self.api_keys.gemini:
            raise ValueError("Gemini API key is required.")

        self.llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=api_keys.gemini)
        self.search_api = GoogleSearchAPI(api_keys, config)
        self.competitor_analyzer = CompetitorAnalyzer(config, self.llm)
        self.content_generator = ContentGenerator(config, self.llm)

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """Executes the full pipeline from analysis to content generation."""
        logger.info(f"Starting blog post generation for: '{blog_input.title}'")

        # Phase 1: Competitor Analysis
        logger.info("[Phase 1/2] Analyzing top competitors...")
        competitor_reports, insights = self.competitor_analyzer.run_analysis(blog_input, self.search_api)

        # Phase 2: Content Generation
        logger.info("[Phase 2/2] Generating content based on strategic insights...")
        blog_post_md = self.content_generator.generate_blog_post(blog_input, insights, competitor_reports)

        logger.info("Blog post generation complete.")
        return {
            "blog_post": blog_post_md,
            "competitor_reports": [r.__dict__ for r in competitor_reports],
            "content_insights": insights.__dict__
        }
```

-----

### `main.py`

The main entrypoint script that ties everything together.

```python
# main.py
"""Main execution script for the AI-Powered SEO Blog Writer."""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from config import SEOBlogConfig, APIKeys, BlogInput
from services import SEOBlogWriter
from utils import get_api_key, sanitize_filename, logger

def main():
    """Main function to run the blog generation process."""
    load_dotenv() # Load .env file for environment variables

    parser = argparse.ArgumentParser(description="Generate an SEO-optimized blog post.")
    parser.add_argument("input_file", type=str, help="Path to the JSON file with blog input data.")
    args = parser.parse_args()

    # --- Load Inputs ---
    try:
        input_path = Path(args.input_file)
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        blog_input = BlogInput(**input_data)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Failed to load or parse input file '{args.input_file}': {e}")
        return

    # --- Configuration ---
    config = SEOBlogConfig()
    api_keys = APIKeys(
        gemini=get_api_key("GEMINI_API_KEY", "gemini.key"),
        Google Search=get_api_key("Google Search_API_KEY", "gsearch_api.key"),
        google_cx=get_api_key("GOOGLE_CX", "gsearch_cx.key")
    )

    # --- Run Process ---
    try:
        writer = SEOBlogWriter(config, api_keys)
        result = writer.write_blog(blog_input)
    except (ValueError, Exception) as e:
        logger.error(f"A critical error occurred during the blog writing process: {e}", exc_info=True)
        return

    # --- Save Outputs ---
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    base_filename = sanitize_filename(f"{timestamp}_{blog_input.title}")

    # Save blog post
    md_path = output_dir / f"{base_filename}.md"
    md_path.write_text(result["blog_post"], encoding="utf-8")
    logger.info(f"Blog post saved to: {md_path}")

    # Save analysis report
    report_path = output_dir / f"{base_filename}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "content_insights": result["content_insights"],
                "competitor_reports": result["competitor_reports"],
            },
            f,
            indent=4
        )
    logger.info(f"Analysis report saved to: {report_path}")

if __name__ == "__main__":
    main()
