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

# --- Imports for Google Drive functionality ---
import pypandoc
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request # <-- CORRECTED IMPORT
# --- End of new imports ---

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Added for Google Drive API ---
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# ---

@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.5-pro" # Using a valid and current model name
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
    @staticmethod
    def get_api_key(filepath: str = "geminaikey") -> Optional[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'")
                return None
            
            with open(key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
                
            if not api_key or len(api_key) < 10:
                logger.error("Invalid API key")
                return None
                
            if 'geminaikey' in filepath:
                 logger.info(f"Gemini API key loaded successfully.")
            return api_key
            
        except Exception as e:
            logger.error(f"Error reading API key: {e}")
            return None

class GoogleSearchAPI:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        if not self.api_key:
            raise ValueError(f"Google Search API key not found in {config.google_search_api_file}")
        if not self.search_engine_id:
            raise ValueError(f"Google Search Engine ID not found in {config.google_cx_file}")
        
        logger.info("Google Custom Search API initialized successfully")
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        try:
            params = {
                'key': self.api_key, 'cx': self.search_engine_id, 'q': query,
                'num': min(num_results, 10), 'fields': 'items(title,link,snippet,displayLink)'
            }
            if self.config.verbose: logger.info(f"Executing Google Search for query: {query}")
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            results = [{'title': i.get('title',''), 'url': i.get('link',''), 'snippet': i.get('snippet',''), 'domain': i.get('displayLink','')} for i in data.get('items', [])]
            if self.config.verbose: logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Google Search API error: {e}")
            return []

class CompetitorAnalyzer:
    """Analyzes competitor content for SEO insights."""
    
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent, 'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def search_competitors(self, keywords: List[str]) -> List[Dict[str, Any]]:
        competitor_data = []
        for keyword in keywords[:3]:
            try:
                results = self.google_search.search(f"{keyword} blog article", num_results=10)
                for i, result in enumerate(results[:5]):
                    if not any(exc in result['domain'] for exc in ['google.com', 'facebook.com', 'youtube.com', 'wikipedia.org']):
                        result.update({'keyword': keyword, 'search_position': i + 1})
                        competitor_data.append(result)
                sleep(1)
            except Exception as e:
                logger.error(f"Error searching for keyword '{keyword}': {e}")
        
        unique_competitors = {comp['url']: comp for comp in competitor_data}.values()
        logger.info(f"Found {len(unique_competitors)} unique competitor pages to analyze.")
        return list(unique_competitors)[:self.config.competitor_analysis_count]
    
    def analyze_competitor_content(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.config.verbose: logger.info(f"Analyzing content from {len(competitors)} competitor pages...")
        analysis_results = {'total_analyzed': 0, 'successful_analyses': 0, 'search_snippets': [c.get('snippet') for c in competitors if c.get('snippet')]}
        successful_analyses = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            future_to_comp = {executor.submit(self._analyze_single_competitor, c): c for c in competitors}
            for future in as_completed(future_to_comp):
                try:
                    result = future.result()
                    if result.get('success'):
                        successful_analyses.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing competitor {future_to_comp[future]['url']}: {e}")

        analysis_results['successful_analyses'] = len(successful_analyses)
        analysis_results['total_analyzed'] = len(competitors)
        if successful_analyses:
            analysis_results.update(self._process_competitor_analyses(successful_analyses))
        
        logger.info(f"Completed competitor analysis: {analysis_results['successful_analyses']}/{analysis_results['total_analyzed']} pages analyzed.")
        return analysis_results
    
    def _analyze_single_competitor(self, competitor: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.get(competitor['url'], timeout=self.config.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for el in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']): el.decompose()
            title = soup.find('title').get_text().strip() if soup.find('title') else competitor.get('title', '')
            main_content = soup.find('main') or soup.find('article') or soup.body
            content_text = main_content.get_text(separator=' ', strip=True) if main_content else ""
            return {
                'success': True, 'url': competitor['url'], 'title': title, 
                'content': content_text[:2000], 'word_count': len(content_text.split()),
                'headings': [h.get_text().strip() for i in range(1, 4) for h in soup.find_all(f'h{i}')]
            }
        except Exception as e:
            logger.warning(f"Failed to analyze competitor {competitor['url']}: {e}")
            return {'success': False, 'url': competitor['url'], 'error': str(e)}

    def _process_competitor_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        word_counts = [a['word_count'] for a in analyses if a.get('word_count', 0) > 0]
        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
        all_headings = [h.lower() for a in analyses for h in a.get('headings', [])]
        common_topics = {k: v for k, v in sorted(__import__('collections').Counter(all_headings).items(), key=lambda item: item[1], reverse=True)[:20]}
        return {'avg_word_count': avg_word_count, 'common_topics': common_topics}

class MarkdownFormatter:
    @staticmethod
    def format_blog_output(result: Dict[str, Any], blog_input: BlogInput) -> str:
        # This function can be expanded as needed.
        return result["blog_content"]

class SEOBlogWriter:
    """Main SEO blog writing agent."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.formatter = MarkdownFormatter()
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        api_key = APIKeyManager.get_api_key(self.config.api_key_file)
        if not api_key: raise ValueError("Gemini API key is required")
        try:
            llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key)
            logger.info(f"LLM initialized with model: {self.config.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}"); raise

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        logger.info(f"Starting blog writing for: '{blog_input.title}'")
        try:
            logger.info(f"[1/3] Analyzing Competitors...")
            analysis_results = self.analyzer.analyze_competitor_content(self.analyzer.search_competitors(blog_input.primary_keywords))
            
            logger.info(f"[2/3] Building prompt...")
            prompt = self._build_generation_prompt(blog_input, analysis_results)
            
            logger.info(f"[3/3] Generating content with LLM...")
            response = self.llm.invoke(prompt)
            blog_content = response.content
            word_count = len(blog_content.split())
            logger.info(f"Finished writing '{blog_input.title}' ({word_count} words)")
            return {"success": True, "blog_content": blog_content}
        except Exception as e:
            logger.error(f"Error writing '{blog_input.title}': {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_generation_prompt(self, blog_input: BlogInput, analysis: Dict[str, Any]) -> str:
        analysis_summary = json.dumps({k: v for k, v in analysis.items() if k in ['avg_word_count', 'common_topics']}, indent=2)
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

    def save_blog_to_file(self, result: Dict[str, Any], blog_input: BlogInput) -> Path:
        safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')
        filename = Path(f"{safe_title[:100]}.md")
        markdown_content = self.formatter.format_blog_output(result, blog_input)
        filename.write_text(markdown_content, encoding='utf-8')
        logger.info(f"Content for '{blog_input.title}' saved to: {filename.resolve()}")
        return filename

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [BlogInput(**item) for item in data]
    except Exception as e:
        logger.error(f"Error loading or parsing config file {filepath}: {e}"); raise

# --- Google Drive Utility Functions ---
def authenticate_google_api():
    """Handles Google API authentication flow."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request()) # <-- CORRECTED THIS LINE
        else:
            if not os.path.exists('credentials.json'):
                logger.error("Google Drive upload requires 'credentials.json'. Get it from Google Cloud Console.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def convert_md_to_odt(md_file_path: Path) -> Path:
    """Converts a Markdown file to an ODT file using Pandoc."""
    odt_file = md_file_path.with_suffix('.odt')
    try:
        pypandoc.convert_file(str(md_file_path), 'odt', outputfile=str(odt_file))
        logger.info(f"Successfully converted {md_file_path.name} to {odt_file.name}")
        return odt_file
    except Exception as e:
        logger.error(f"Failed to convert markdown to ODT: {e}. Make sure Pandoc is installed.")
        raise

def upload_to_drive(odt_file_path: Path, title: str, drive_service):
    """Uploads a file to Google Drive as a Google Doc."""
    file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
    media = MediaFileUpload(str(odt_file_path), mimetype='application/vnd.oasis.opendocument.text')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
    logger.info(f"✅ Successfully uploaded to Google Drive as '{title}'.")
    logger.info(f"   View online: {file.get('webViewLink')}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file with blog input data.')
    parser.add_argument('--api-key-file', type=str, default='geminaikey', help='Path to Gemini API key file.')
    parser.add_argument('--google-search-api', type=str, default='googlesearchapi', help='Path to Google Search API key file.')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Path to Google Custom Search Engine ID file.')
    parser.add_argument('--min-words', type=int, default=1500, help='Minimum word count.')
    parser.add_argument('--max-words', type=int, default=3000, help='Maximum word count.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--drive', action='store_true', help='Upload the final blog post to Google Drive.')
    return parser.parse_args()

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
            logger.error(f"An error occurred during Google Drive authentication: {e}")

    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        config = SEOBlogConfig(
            api_key_file=args.api_key_file, google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx, min_word_count=args.min_words,
            max_word_count=args.max_words, verbose=args.verbose
        )
        writer = SEOBlogWriter(config)

        for i, blog_input in enumerate(blog_inputs, 1):
            logger.info(f"--- Starting job {i}/{len(blog_inputs)}: '{blog_input.title}' ---")
            result = writer.write_blog(blog_input)
            
            if result and result.get('success'):
                md_file_path = writer.save_blog_to_file(result, blog_input)
                print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' complete. File saved to: {md_file_path}\n")

                if args.drive and drive_service:
                    try:
                        logger.info("Starting Google Drive upload process...")
                        odt_file_path = convert_md_to_odt(md_file_path)
                        upload_to_drive(odt_file_path, blog_input.title, drive_service)
                        os.remove(odt_file_path) # Clean up intermediate .odt file
                    except Exception as e:
                        logger.error(f"Google Drive upload failed for '{blog_input.title}': {e}")

            else:
                logger.error(f"Failed to generate blog post for '{blog_input.title}'. Reason: {result.get('error', 'Unknown')}")
            
            if i < len(blog_inputs): sleep(5)

    except Exception as e:
        logger.critical(f"A critical error occurred in the main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()
