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
# --- FINAL CORRECTED OLLAMA IMPORT AND CLASS NAME ---
from langchain_ollama.llms import OllamaLLM

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
    model_name: str = "gemini-2.5-flash" # For the initial SEO draft
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

@dataclass
class CompetitorData:
    """Structured data for a single competitor."""
    url: str
    title: str
    meta_description: str
    h1: Optional[str]
    h2s: List[str]
    word_count: int
    summary: str

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

    def search(self, query: str, num_results: int = 5, sort_by_date: bool = False) -> List[Dict[str, Any]]:
        try:
            params = {'key': self.api_key, 'cx': self.search_engine_id, 'q': query, 'num': num_results}
            log_msg = f"Executing Search for: {query}"
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
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"""
You are a world-class research assistant. Your task is to generate a list of Google search queries to gather information for a blog post.

**Research Context:**
- **Current Date:** {current_date}
- **Blog Post Title:** "{blog_input.title}"
- **Primary Keywords:** {blog_input.primary_keywords}
- **Target Audience:** {blog_input.target_audience}

Generate two types of queries based on the current date and blog topic:
1.  **news_queries:** To find the absolute latest news, announcements, and trends.
2.  **fact_queries:** To find foundational, evergreen information and data.

**CRITICAL:** Respond with ONLY a valid JSON object containing the two keys "news_queries" and "fact_queries".

Example format:
{{
  "news_queries": ["latest advancements in {blog_input.primary_keywords[0]}", "{blog_input.primary_keywords[0]} announcements {datetime.now().year}"],
  "fact_queries": ["what is {blog_input.primary_keywords[0]}", "how does {blog_input.primary_keywords[0]} work"]
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

# --- Competitor Analysis Phase ---
class CompetitorAnalyzer:
    """Fetches and analyzes top 5 ranking URLs for key SEO parameters."""
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI):
        self.config = config
        self.google_search = search_api
        self.llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=APIKeyManager.get_api_key(config.api_key_file))
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})

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

    def _summarize_text_with_llm(self, text: str) -> str:
        """Uses an LLM to create a concise summary of the competitor's content."""
        if not text.strip(): return "Content was empty or could not be read."
        prompt = f"""
        Analyze the following article text and provide a very concise summary (under 80 words).
        Focus on the main arguments, key data points, and overall structure.

        Article Text (first 8000 characters):
        ---
        {text[:8000]}
        ---
        Concise Summary:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return "Could not summarize content due to an API error."

    def _analyze_single_competitor(self, url: str) -> Optional[CompetitorData]:
        """Scrapes and analyzes a single URL for SEO data."""
        logger.info(f"Analyzing competitor: {url}")
        soup = self._fetch_and_parse_url(url)
        if not soup: return None

        title = soup.title.string.strip() if soup.title else "No Title Found"
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_desc['content'].strip() if meta_desc and meta_desc.get('content') else "No Meta Description Found"
        h1 = soup.find('h1')
        h1_text = h1.get_text(strip=True) if h1 else "No H1 Found"
        h2s = [h2.get_text(strip=True) for h2 in soup.find_all('h2')]

        main_content_element = soup.find('main') or soup.find('article') or soup.body
        content_text = ' '.join(main_content_element.get_text(strip=True).split())
        word_count = len(re.findall(r'\w+', content_text))
        summary = self._summarize_text_with_llm(content_text)

        return CompetitorData(
            url=url, title=title, meta_description=meta_description,
            h1=h1_text, h2s=h2s[:8], word_count=word_count, summary=summary
        )

    def analyze_competitors(self, blog_input: BlogInput) -> List[CompetitorData]:
        """Main method to orchestrate the competitor analysis."""
        search_query = f'"{blog_input.primary_keywords[0]}"'
        logger.info(f"Finding top 5 competitors for query: {search_query}")

        try:
            search_results = self.google_search.search(search_query, num_results=5)
            urls = [result['link'] for result in search_results if 'link' in result]
            if not urls: logger.warning("No competitor URLs found."); return []

            logger.info(f"Found {len(urls)} competitor URLs to analyze.")
            reports = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(self._analyze_single_competitor, url): url for url in urls}
                for future in as_completed(future_to_url):
                    result = future.result()
                    if result: reports.append(result)

            logger.info(f"Successfully completed analysis for {len(reports)} competitors.")
            return reports
        except Exception as e:
            logger.error(f"An error occurred during competitor analysis: {e}", exc_info=True)
            return []

# --- Researcher using the QueryGenerator ---
class NewsAndDataResearcher:
    """Gathers news and facts using LLM-generated queries."""
    def __init__(self, config: SEOBlogConfig, search_api: GoogleSearchAPI):
        self.config = config
        self.google_search = search_api
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
            sleep(0.5)

        logger.info("--- Starting Foundational Fact-Check Phase ---")
        for query in fact_queries[:3]:
            results = self.google_search.search(query, sort_by_date=False)
            for result in results:
                if result.get('snippet'): all_snippets.append(result['snippet'].strip())
            sleep(0.5)

        unique_snippets = list(set(all_snippets))
        logger.info(f"Collected {len(unique_snippets)} unique research snippets.")
        return unique_snippets

# --- Final Stage Content Humanizer ---
class ContentHumanizer:
    """Uses Ollama to rephrase content to be more engaging and natural."""
    def __init__(self):
        logger.info("Initializing Ollama LLM for humanization stage...")
        # --- Using the corrected class name 'OllamaLLM' ---
        self.llm = OllamaLLM(
            model="openhermes",
            temperature=1,
            top_p=1,
            top_k=40,
            mirostat=1,
            mirostat_eta=0.1,
            mirostat_tau=10,
            num_ctx=4096,
            num_thread=6,
            system="Use active voice, goal is to create human written content"
        )

    def _build_prompt(self, content: str) -> str:
        return f"""Rephrase the following blog post to make it sound more natural, engaging, and human-written.

**CRITICAL INSTRUCTIONS:**
1.  **Preserve Formatting:** Keep the original markdown formatting perfectly intact (e.g., `#`, `##`, `*`, `**`).
2.  **Do Not Add Sections:** Do not add new sections, headers, or an introduction/conclusion if they don't already exist.
3.  **Improve Flow:** Focus on improving the sentence structure, tone, and flow of the existing text.
4.  **No Commentary:** Your output must ONLY be the rephrased article. Do not add any extra text like "Here is the rephrased version:".

Here is the content to rephrase:
---
{content}
---
**REPHRASED BLOG POST (MARKDOWN):**
"""

    def humanize(self, content: str) -> str:
        logger.info("Humanizing content...")
        prompt = self._build_prompt(content)
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Ollama humanization failed: {e}. Returning original draft.")
            return content

# --- Main SEOBlogWriter and supporting classes ---
class SEOBlogWriter:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search_api = GoogleSearchAPI(config)
        self.researcher = NewsAndDataResearcher(config, self.google_search_api)
        self.competitor_analyzer = CompetitorAnalyzer(config, self.google_search_api)
        self.humanizer = ContentHumanizer()

    def _setup_seo_llm(self):
        """Sets up the initial LLM for SEO draft generation."""
        return ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=APIKeyManager.get_api_key(self.config.api_key_file))

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing process for '{blog_input.title}'")
        try:
            logger.info("[Phase 1/4] Analyzing Top 5 Competitors...")
            competitor_reports = self.competitor_analyzer.analyze_competitors(blog_input)

            logger.info("[Phase 2/4] Generating search queries and gathering research...")
            news_and_facts = self.researcher.gather_news_and_facts(blog_input)

            logger.info("[Phase 3/4] Building prompt for initial SEO content generation...")
            prompt = self._build_generation_prompt(blog_input, news_and_facts, competitor_reports)

            logger.info("[Phase 4/4] Generating initial SEO draft...")
            seo_llm = self._setup_seo_llm()
            seo_draft = seo_llm.invoke(prompt).content

            logger.info("[Phase 5/5] Humanizing content with Ollama...")
            humanized_content = self.humanizer.humanize(seo_draft)

            return {"success": True, "blog_content": humanized_content}
        except Exception as e:
            logger.error(f"Error in blog writing process: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_generation_prompt(self, blog_input: BlogInput, research_snippets: List[str], competitor_reports: List[CompetitorData]) -> str:
        research_summary = "\n".join([f'- "{snippet}"' for snippet in research_snippets]) if research_snippets else "No research gathered."

        competitor_summary = "No competitor analysis was available."
        if competitor_reports:
            formatted_reports = []
            for i, report in enumerate(competitor_reports):
                report_str = f"""
### Competitor {i+1}: {report.title}
- **URL**: {report.url}
- **H1**: {report.h1}
- **Word Count**: ~{report.word_count} words
- **Outline (H2s)**: {', '.join(report.h2s) if report.h2s else 'N/A'}
- **AI Summary**: {report.summary}
"""
                formatted_reports.append(report_str.strip())
            competitor_summary = "\n".join(formatted_reports)

        return f"""
You are an expert SEO strategist and content creator. Your goal is to write a blog post that outranks competition and engages readers. You must balance two priorities:

1.  **NLP-Friendly Formatting (60% Focus):** Create content that is clear and simple for algorithms to understand.
2.  **Humanizing & Engagement (40% Focus):** Make the content enjoyable and valuable for human readers.

**--- CRITICAL WRITING RULES ---**

**A. NLP-FRIENDLY FORMATTING (60% FOCUS):**
*   **Structure:** Use simple, direct sentences. Follow a Subject-Verb-Object order where possible, always include atleast four faq after conclusion.
*   **Word Choice:** Select words for their precise meaning. Avoid ambiguity.
*   **Conciseness:** Exclude all filler content. Deliver information succinctly.
*   **FORBIDDEN WORDS & PHRASES:** You are strictly prohibited from using the following terms: 'meticulous,' 'navigating,' 'complexities,' 'realm,' 'bespoke,' 'tailored,' 'towards,' 'underpins,' 'ever-changing,' 'ever-evolving,' 'the world of,' 'not only,' 'seeking more than just,' 'designed to enhance,' 'it’s not merely,' 'our suite,' 'it is advisable,' 'daunting,' 'in the heart of,' 'when it comes to,' 'in the realm of,' 'amongst,' 'unlock the secrets,' 'unveil the secrets,' 'robust.'

**B. HUMANIZING & ENGAGEMENT (40% FOCUS):**
*   While following the NLP rules, make the content compelling.
*   The article should be valuable and satisfying for the target audience.
*   Balance technical optimization with a good reader experience.

**C. STRATEGIC EXECUTION:**
*   **Analyze Competition:** Review the `COMPETITOR ANALYSIS` to identify content gaps. Create a superior article that is more comprehensive.
*   **Use Fresh Facts:** Base your article on the `LATEST NEWS & FACTUAL RESEARCH`. Ensure content is current and data-driven.
*   **Output Format:** The entire output must be in valid Markdown.

---
**BLOG POST REQUIREMENTS:**
- **Title**: {blog_input.title}
- **Tone**: {blog_input.tone}
- **Target Audience**: {blog_input.target_audience}

---
**COMPETITOR ANALYSIS (TOP 5 GOOGLE RESULTS):**
{competitor_summary}

---
**LATEST NEWS & FACTUAL RESEARCH (Your Factual Foundation):**
{research_summary}

---
**YOUR TASK:**
Write the full blog post now. Adhere strictly to all rules. Use the competitor analysis to create a better structure and the research for fresh details. Your final article must be better than the competition, optimized for NLP, and engaging for people.

**BEGIN BLOG POST (MARKDOWN):**
"""

# --- Main execution block ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts using LLM-generated queries and competitor analysis.")
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
