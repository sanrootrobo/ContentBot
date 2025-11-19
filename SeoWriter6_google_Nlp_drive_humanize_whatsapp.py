# main.py  (modified to integrate whatsapp parsed JSON & Ollama target-audience inference)
import os
import requests
import logging
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from datetime import datetime
from threading import Lock
import sys

# Third-party imports for new features
import pypandoc
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Imports for the --humanize / Ollama feature
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoogleDriveUploader:
    """Handles authentication and uploading files to Google Drive."""
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    def __init__(self, client_secrets_file: str = 'credentials.json', token_file: str = 'token.json'):
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self.drive_service = self._authenticate()

    def _authenticate(self):
        """Authenticates with Google API and returns a Drive service object."""
        creds = None
        if not os.path.exists(self.client_secrets_file):
            logger.error(f"Google Drive credentials file not found at '{self.client_secrets_file}'. Drive upload is disabled.")
            return None
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
            except Exception as e:
                logger.warning(f"Could not load credentials from token file, will re-authenticate. Error: {e}")
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Credentials expired. Refreshing token...")
                try:
                    creds.refresh(GoogleAuthRequest())
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}. A new login will be required.")
                    creds = None
            else:
                logger.info("No valid credentials. Starting new authentication flow.")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.error(f"Authentication flow failed: {e}")
                    return None
            if creds:
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
        if creds:
            logger.info("Google Drive authentication successful.")
            return build('drive', 'v3', credentials=creds)
        else:
            logger.error("Failed to obtain Google Drive credentials.")
            return None

    def upload_blog(self, md_file_path: str, title: str) -> Optional[str]:
        """Converts Markdown to ODT and uploads it as a Google Doc."""
        if not self.drive_service:
            logger.error(f"Drive service not available. Cannot upload '{title}'.")
            return None
        odt_file = Path(md_file_path).with_suffix('.odt')
        try:
            logger.info(f"Converting '{md_file_path}' to ODT for Google Drive...")
            pypandoc.convert_file(md_file_path, 'odt', outputfile=str(odt_file), extra_args=['--standalone'])
            logger.info(f"Uploading '{title}' to Google Drive...")
            file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
            media = MediaFileUpload(str(odt_file), mimetype='application/vnd.oasis.opendocument.text', resumable=True)
            file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
            doc_link = file.get('webViewLink')
            logger.info(f"✅ Successfully uploaded '{title}' to Google Drive. Link: {doc_link}")
            return doc_link
        except Exception as e:
            logger.error(f"Google Drive upload for '{title}' failed: {e}")
            return None
        finally:
            if os.path.exists(odt_file):
                os.remove(odt_file)


class OllamaHumanizer:
    """
    Uses a local Ollama model to rephrase markdown content paragraph by paragraph
    and also infer target audience when required.
    """
    def __init__(self):
        # Keep the original large system prompt (as before) for rephrasing/humanizing
        self.system_prompt = """
        Remember this writing style and response in it : Tittle: Best Online Platforms for Buying and Renting Property in India

        Primary keyword: property rental websites

        Secondary keywords: free rental property listing websites, best rental property websites

        ... (kept long prompt as in original script) ...
        """
        self.llm = self._initialize_ollama()

    def _initialize_ollama(self):
        """Initializes and configures the Ollama chat model."""
        try:
            llm = ChatOllama(
                model="openhermes", system=self.system_prompt, seed=42,
                stop=["\n\n", "Conclusion:"], temperature=1, mirostat=0,
                mirostat_eta=0.1, mirostat_tau=8, top_k=40, top_p=1, tfs_z=1.0,
                repeat_penalty=1.6, repeat_last_n=64, num_ctx=4096, num_predict=1024,
                num_gpu=1, num_thread=8, use_mmap=True, use_mlock=False
            )
            logger.info("Ollama 'openhermes' model initialized for humanizing & inference.")
            return llm
        except Exception as e:
            logger.error(f"Error initializing Ollama. Is the server running and 'openhermes' installed? Error: {e}")
            return None

    def rephrase_content(self, original_content: str, title: str, verbose: bool = False) -> str:
        """Rephrases markdown text, showing a progress bar and optional verbose logs."""
        if not self.llm:
            return original_content

        content_parts = re.split(r'(?m)^(#{1,3} .*)$', original_content)
        final_document_parts = []
        parts_to_process = [p for p in content_parts if p and not p.isspace()]
        total_parts = len(parts_to_process)
        if total_parts == 0:
            return ""

        processed_count = 0
        for part in parts_to_process:
            processed_count += 1
            if re.match(r'^(#{1,3} .*)$', part.strip()):
                if verbose: logger.info(f"--- Found Heading: {part.strip()} ---")
                final_document_parts.append(part.strip())
            else:
                if verbose: logger.info(f"--- Rephrasing paragraph: '{part.strip()[:60]}...' ---")
                messages = [HumanMessage(content=f"rephrase this paragraph, keeping bold keywords intact: \n\n---\n\n{part.strip()}")]
                try:
                    rephrased_paragraph = "".join([chunk.content for chunk in self.llm.stream(messages)])
                    final_document_parts.append(rephrased_paragraph)
                except Exception as e:
                    logger.error(f"\nOllama error for '{title}': {e}")
                    final_document_parts.append(part.strip())

            progress = (processed_count / total_parts) * 100
            with Lock():
                sys.stdout.write(f"\r--- [Humanizing '{title[:25]}...'] Rephrasing progress: {progress:.2f}% ---")
                sys.stdout.flush()

        with Lock():
            sys.stdout.write('\n')

        return "\n\n".join(final_document_parts)

    def infer_target_audience(self, title: str, primary_keywords: List[str], secondary_keywords: List[str], max_chars: int = 240) -> str:
        """
        Infer a concise target audience string using Ollama LLM.
        Falls back to a heuristic string if the model isn't available.
        """
        if not self.llm:
            # Heuristic fallback: join keywords into a short audience hint
            pk = ', '.join(primary_keywords[:2]) if primary_keywords else ''
            sk = ', '.join(secondary_keywords[:2]) if secondary_keywords else ''
            fallback = f"Audience interested in {pk}" + (f" and {sk}" if sk else "")
            logger.warning("Ollama not available. Using heuristic target audience: %s", fallback)
            return fallback[:max_chars]

        # Build a concise prompt for audience inference
        prompt = (
            f"Given the blog title and keywords below, produce a concise single-line target audience (very short, <= 100 chars) "
            f"that best describes who the blog should be written for.\n\n"
            f"Title: {title}\n"
            f"Primary keywords: {', '.join(primary_keywords)}\n"
            f"Secondary keywords: {', '.join(secondary_keywords)}\n\n"
            f"Return ONLY the audience string (no explanation)."
        )
        try:
            messages = [HumanMessage(content=prompt)]
            # use the same streaming interface and join
            response = "".join([chunk.content for chunk in self.llm.stream(messages)])
            if not response:
                raise ValueError("Empty response from Ollama")
            audience = response.strip().splitlines()[0].strip()
            return audience[:max_chars]
        except Exception as e:
            logger.error("Ollama inference failed: %s. Falling back to heuristic.", e)
            pk = ', '.join(primary_keywords[:2]) if primary_keywords else ''
            sk = ', '.join(secondary_keywords[:2]) if secondary_keywords else ''
            fallback = f"Audience interested in {pk}" + (f" and {sk}" if sk else "")
            return fallback[:max_chars]


@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.5-pro"
    timeout: int = 20
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    max_concurrent_requests: int = 3
    competitor_analysis_count: int = 5
    min_word_count: int = 1500
    max_word_count: int = 3000
    drive_upload: bool = False
    humanize: bool = False


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
        if not self.brand or not self.title or not self.primary_keywords:
            raise ValueError("Brand, title, and primary_keywords are required.")


class APIKeyManager:
    """Manages a pool of API keys from a file."""
    @staticmethod
    def get_api_keys(filepath: str = "geminaikey") -> List[str]:
        try:
            keys = [line.strip() for line in Path(filepath).read_text().splitlines() if line.strip() and len(line.strip()) > 10]
            if not keys:
                logger.error(f"No valid API keys found in {filepath}")
                return []
            logger.info(f"{len(keys)} API key(s) loaded from {filepath}")
            return keys
        except FileNotFoundError:
            logger.error(f"API key file not found: {filepath}")
            return []

    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        keys = APIKeyManager.get_api_keys(filepath)
        return keys[0] if keys else None


class GoogleSearchAPI:
    """Google Custom Search API client."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        if not self.api_key or not self.search_engine_id:
            raise ValueError("Google Search API key or CX not found.")

    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        try:
            if self.config.verbose: logger.info(f"Executing Google Search for query: {query}")
            params = {'key': self.api_key, 'cx': self.search_engine_id, 'q': query, 'num': min(num_results, 10)}
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            results = response.json().get('items', [])
            if self.config.verbose: logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except requests.RequestException as e:
            logger.error(f"Google Search request failed: {e}")
            return []


class CompetitorAnalyzer:
    """Analyzes competitor content."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.google_search = GoogleSearchAPI(config)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})

    def search_competitors(self, keywords: List[str]) -> List[Dict[str, Any]]:
        competitor_urls = set()
        for keyword in keywords[:3]:
            results = self.google_search.search(f"{keyword} blog article", num_results=5)
            for result in results:
                competitor_urls.add(result['link'])
            sleep(1)

        unique_competitors = [{'url': url} for url in list(competitor_urls)]
        if self.config.verbose: logger.info(f"Found {len(unique_competitors)} unique competitor pages to analyze.")
        return unique_competitors[:self.config.competitor_analysis_count]

    def analyze_competitor_content(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.config.verbose: logger.info(f"Analyzing content from {len(competitors)} competitor pages...")
        successful_analyses = []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests, thread_name_prefix="Analyzer") as executor:
            futures = {executor.submit(self._analyze_single_competitor, c['url']): c for c in competitors}
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    successful_analyses.append(result)

        if self.config.verbose: logger.info(f"Completed competitor analysis: {len(successful_analyses)}/{len(competitors)} pages analyzed successfully.")
        return self._process_competitor_analyses(successful_analyses) if successful_analyses else {}

    def _analyze_single_competitor(self, url: str) -> Dict[str, Any]:
        if self.config.verbose: logger.info(f"Analyzing competitor URL: {url}")
        try:
            r = self.session.get(url, timeout=self.config.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            content = soup.get_text(separator=' ', strip=True)
            return {'success': True, 'content': content[:5000], 'word_count': len(content.split())}
        except Exception as e:
            logger.warning(f"Failed to analyze competitor {url}: {e}")
            return {'success': False, 'error': str(e)}

    def _process_competitor_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        word_counts = [a['word_count'] for a in analyses]
        avg_wc = sum(word_counts) // len(word_counts) if word_counts else 0
        if self.config.verbose: logger.info(f"Average competitor word count: {avg_wc}")
        return {'avg_word_count': avg_wc}


class MarkdownFormatter:
    @staticmethod
    def format_blog_output(result: Dict[str, Any], blog_input: BlogInput) -> str:
        return f"# {blog_input.title}\n\n{result['blog_content']}"


class SEOBlogWriter:
    """Main SEO blog writing agent with API key rotation."""
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.formatter = MarkdownFormatter()
        self.api_keys = APIKeyManager.get_api_keys(self.config.api_key_file)
        if not self.api_keys:
            raise ValueError("No Gemini API keys found.")
        self.key_lock = Lock()
        self.current_key_index = 0
        self.llm = self._get_llm_with_current_key()

    def _get_llm_with_current_key(self):
        """Gets an LLM instance with the current API key."""
        with self.key_lock:
            key_to_use = self.api_keys[self.current_key_index]
        if self.config.verbose: logger.info(f"Initializing LLM with key index: {self.current_key_index}")
        return ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=key_to_use)

    def _rotate_key_and_get_llm(self) -> bool:
        """Rotates to the next API key and returns True if successful."""
        with self.key_lock:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.warning(f"API Quota likely hit. Rotating to key index: {self.current_key_index}")
            if self.current_key_index == 0:
                logger.error("All available API keys have been tried and failed.")
                return False
        self.llm = self._get_llm_with_current_key()
        return True

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """Writes a blog post, with retries and key rotation on quota errors."""
        title = blog_input.title
        if self.config.verbose:
            logger.info(f"[Phase 1/4 - {title}] Analyzing Competitors...")
        analysis = self.analyzer.analyze_competitor_content(
            self.analyzer.search_competitors(blog_input.primary_keywords)
        )
        if self.config.verbose:
            logger.info(f"[Phase 1/4 - {title}] Competitor analysis complete.")
            logger.info(f"[Phase 2/4 - {title}] Building prompt for content generation...")

        prompt = self._build_generation_prompt(blog_input, analysis)

        if self.config.verbose:
            logger.info(f"[Phase 3/4 - {title}] Generating blog content with LLM...")

        for i in range(len(self.api_keys)):
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                if self.config.verbose:
                    logger.info(f"[Phase 3/4 - {title}] Content generation complete.")
                    logger.info(f"[Phase 4/4 - {title}] Finalizing content.")
                return {"success": True, "blog_content": content}
            except ResourceExhausted:
                logger.warning(f"ResourceExhausted error on '{title}'.")
                if not self._rotate_key_and_get_llm():
                    break
                sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error for '{title}': {e}")
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "All API keys exhausted their quotas."}

    def _build_generation_prompt(self, blog_input: BlogInput, analysis: Dict[str, Any]) -> str:
        return (
            f"Write a long, detailed, SEO-optimized blog post titled '{blog_input.title}'. "
            f"The target audience is {blog_input.target_audience}. "
            f"Competitors have an average word count of around {analysis.get('avg_word_count', 2000)} words, so aim for a word count between {self.config.min_word_count} and {self.config.max_word_count}. "
            f"Integrate the primary keywords '{', '.join(blog_input.primary_keywords)}' naturally throughout the text. "
            f"Also include these secondary keywords where appropriate: '{', '.join(blog_input.secondary_keywords)}'. "
            f"The tone of voice should be strictly {blog_input.tone}. "
            "Structure the article with a compelling introduction, a well-organized body with multiple H2 and H3 subheadings, and a strong concluding summary. The output must be in valid markdown format."
        )

    def save_blog_to_file(self, result: Dict[str, Any], blog_input: BlogInput) -> str:
        safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')[:100]
        filename = f"{safe_title}.md"
        content = self.formatter.format_blog_output(result, blog_input)
        Path(filename).write_text(content, encoding='utf-8')
        logger.info(f"Content for '{blog_input.title}' saved to: {Path(filename).resolve()}")
        return str(Path(filename).resolve())


def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [BlogInput(**item) for item in json.load(f)]


def load_blog_inputs_from_whatsapp_parsed(filepath: str) -> List[BlogInput]:
    """Load the JSON saved by the whatsapp-web.js parser and convert to BlogInput list."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WhatsApp parsed file not found: {filepath}")
    raw = json.load(open(filepath, 'r', encoding='utf-8'))
    if not isinstance(raw, list):
        raise ValueError("WhatsApp parsed JSON must be a list of parsed message objects.")
    blog_inputs: List[BlogInput] = []
    for item in raw:
        # expected keys: brand, title, primary_keywords, secondary_keywords, target_audience, tone, content_type
        try:
            blog = BlogInput(
                brand=item.get('brand', 'Propacity'),
                title=item['title'],
                primary_keywords=item.get('primary_keywords', []),
                secondary_keywords=item.get('secondary_keywords', []),
                target_audience=item.get('target_audience') or "General Audience",
                tone=item.get('tone') or 'Informative',
                content_type=item.get('content_type') or 'Blog Post'
            )
            blog_inputs.append(blog)
        except Exception as e:
            logger.warning(f"Skipping malformed whatsapp parsed entry: {e} -- {item}")
    return blog_inputs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts in parallel.")
    parser.add_argument('-c', '--config-file', type=str, required=not '--whatsapp' in sys.argv and '-w' not in sys.argv and '--wa-file' not in sys.argv, help='Path to the JSON file with blog inputs.')
    parser.add_argument('--drive', action='store_true', help='Upload generated blog posts to Google Drive.')
    parser.add_argument('--humanize', action='store_true', help='(Optional) Rephrase generated content using a local Ollama model for a specific style.')
    parser.add_argument('--min-words', type=int, default=1500, help='Minimum word count.')
    parser.add_argument('--max-words', type=int, default=3000, help='Maximum word count.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output to track processing stages.')

    # New WhatsApp & audience options
    parser.add_argument('--whatsapp', '-w', action='store_true', help='Use WhatsApp parsed JSON (from whatsapp listener) as input source instead of -c config-file.')
    parser.add_argument('--wa-file', type=str, default='parsed_messages.json', help='Path to whatsapp listener parsed JSON file (default parsed_messages.json).')
    parser.add_argument('--target-audience', type=str, default=None, help='Override / set a global target audience. If not provided, Ollama will infer per-item when needed.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        config = SEOBlogConfig(
            min_word_count=args.min_words,
            max_word_count=args.max_words,
            verbose=args.verbose,
            drive_upload=args.drive,
            humanize=args.humanize
        )

        # Initialize humanizer (Ollama) only if needed for inference or --humanize
        need_ollama_for_inference = (args.target_audience is None)
        humanizer_for_infer = None
        if need_ollama_for_inference or config.humanize:
            humanizer_for_infer = OllamaHumanizer()
            if humanizer_for_infer and not humanizer_for_infer.llm:
                logger.warning("Ollama initialization failed. Inference or humanize steps will fall back to heuristics or be skipped.")

        # Source blog inputs: either WhatsApp parsed JSON or provided config file
        if args.whatsapp:
            try:
                blog_inputs = load_blog_inputs_from_whatsapp_parsed(args.wa_file)
                if not blog_inputs:
                    logger.error("No blog inputs loaded from WhatsApp parsed file.")
                    return
            except Exception as e:
                logger.error(f"Failed to load WhatsApp parsed file: {e}")
                return
        else:
            try:
                blog_inputs = load_blog_inputs_from_file(args.config_file)
            except Exception as e:
                logger.error(f"Failed to load config file '{args.config_file}': {e}")
                return

        writer = SEOBlogWriter(config)
        uploader = GoogleDriveUploader() if config.drive_upload else None

        # If user provided a global --target-audience override, use it for all inputs
        global_target_override = args.target_audience.strip() if args.target_audience else None

        def process_blog_task(blog_input: BlogInput):
            """Defines the complete task for one blog post."""
            # Infer target audience per-item if needed
            if global_target_override:
                blog_input.target_audience = global_target_override
            elif (not blog_input.target_audience or blog_input.target_audience.strip() == "" or blog_input.target_audience == "General Audience"):
                # infer using OllamaHumanizer if available
                if humanizer_for_infer and humanizer_for_infer.llm:
                    try:
                        inferred = humanizer_for_infer.infer_target_audience(
                            blog_input.title,
                            blog_input.primary_keywords,
                            blog_input.secondary_keywords
                        )
                        blog_input.target_audience = inferred
                        logger.info(f"Inferred target audience for '{blog_input.title}': {inferred}")
                    except Exception as e:
                        logger.warning(f"Failed to infer audience for '{blog_input.title}': {e}")
                else:
                    # fallback heuristic
                    pk = ', '.join(blog_input.primary_keywords[:2]) if blog_input.primary_keywords else ''
                    sk = ', '.join(blog_input.secondary_keywords[:2]) if blog_input.secondary_keywords else ''
                    blog_input.target_audience = f"Audience interested in {pk}" + (f" and {sk}" if sk else "")
                    logger.info(f"Using heuristic target audience for '{blog_input.title}': {blog_input.target_audience}")

            # Proceed with regular generation
            result = writer.write_blog(blog_input)

            if result and result.get('success'):
                # --- Humanizer Step (rephrase) ---
                if config.humanize and humanizer_for_infer and humanizer_for_infer.llm:
                    logger.info(f"Starting humanizing step for '{blog_input.title}'...")
                    original_content = result['blog_content']
                    rephrased_content = humanizer_for_infer.rephrase_content(original_content, blog_input.title, verbose=config.verbose)
                    result['blog_content'] = rephrased_content
                    logger.info(f"Successfully completed humanizing step for '{blog_input.title}'.")

                file_path = writer.save_blog_to_file(result, blog_input)
                upload_msg = ""
                if uploader:
                    drive_link = uploader.upload_blog(file_path, blog_input.title)
                    upload_msg = f" | GDrive Link: {drive_link}" if drive_link else " | GDrive Upload Failed"
                return f"✅ SUCCESS: '{blog_input.title}' -> {file_path}{upload_msg}"
            else:
                error = result.get('error', 'Unknown error') if result else 'Unknown error'
                return f"❌ FAILED: '{blog_input.title}'. Reason: {error}"

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=config.max_concurrent_requests, thread_name_prefix='BlogBot') as executor:
            for i, blog_input in enumerate(blog_inputs):
                logger.info(f"--- Starting job {i+1} of {len(blog_inputs)}: '{blog_input.title}' ---")

            future_results = [executor.submit(process_blog_task, blog) for blog in blog_inputs]
            for future in as_completed(future_results):
                try:
                    message = future.result()
                    print(f"\n{message}\n")
                except Exception as e:
                    logger.error(f"A blog processing task failed unexpectedly: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"The program encountered a critical error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

