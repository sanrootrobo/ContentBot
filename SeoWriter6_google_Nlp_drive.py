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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- NEW: Google Drive Uploader (with corrected authentication) ---
class GoogleDriveUploader:
    """Handles authentication and uploading files to Google Drive."""
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    def __init__(self, client_secrets_file: str = 'credentials.json', token_file: str = 'token.json'):
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self.drive_service = self._authenticate()

    def _authenticate(self):
        """
        Authenticates with Google API. Handles token loading, refreshing,
        and new user authentication. Returns a Drive service object.
        """
        creds = None
        
        if not os.path.exists(self.client_secrets_file):
            logger.error(f"Google Drive credentials file not found at '{self.client_secrets_file}'. Drive upload is disabled.")
            return None

        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time.
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
            except Exception as e:
                logger.warning(f"Could not load credentials from token file, will re-authenticate. Error: {e}")

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Credentials have expired. Attempting to refresh token...")
                try:
                    creds.refresh(GoogleAuthRequest())
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}. A new login will be required.")
                    creds = None  # Force re-authentication by nullifying creds
            else:
                logger.info("No valid credentials found. Starting new authentication flow.")
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.error(f"Authentication flow failed: {e}")
                    return None
            
            # Save the credentials for the next run, only if we have valid credentials
            if creds:
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Credentials saved to {self.token_file}")

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
            logger.info(f"Converting '{md_file_path}' to ODT format for Google Drive upload.")
            pypandoc.convert_file(md_file_path, 'odt', outputfile=str(odt_file), extra_args=['--standalone'])
        except Exception as e:
            logger.error(f"Failed to convert markdown to ODT for '{title}': {e}")
            return None

        try:
            logger.info(f"Uploading '{title}' to Google Drive...")
            file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
            media = MediaFileUpload(str(odt_file), mimetype='application/vnd.oasis.opendocument.text', resumable=True)
            
            file = self.drive_service.files().create(
                body=file_metadata, media_body=media, fields='id,webViewLink'
            ).execute()
            
            doc_link = file.get('webViewLink')
            logger.info(f"✅ Successfully uploaded '{title}' to Google Drive. Link: {doc_link}")
            return doc_link
        except HttpError as e:
            logger.error(f"An HTTP error occurred during Google Drive upload for '{title}': {e}")
            return None
        finally:
            if os.path.exists(odt_file):
                os.remove(odt_file)

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

# --- MODIFIED: APIKeyManager for Multiple Keys ---
class APIKeyManager:
    """Manages a pool of API keys from a file."""
    
    @staticmethod
    def get_api_keys(filepath: str = "geminaikey") -> List[str]:
        """Retrieve and validate multiple API keys from a newline-separated file."""
        try:
            key_path = Path(filepath)
            if not key_path.exists():
                logger.error(f"API key file not found at '{filepath}'")
                return []
            
            with open(key_path, 'r', encoding='utf-8') as f:
                keys = [line.strip() for line in f if line.strip()]
            
            valid_keys = [key for key in keys if len(key) > 10]
            if not valid_keys:
                logger.error("No valid API keys found in file.")
                return []
                
            logger.info(f"{len(valid_keys)} API key(s) loaded successfully from {filepath}")
            return valid_keys
            
        except Exception as e:
            logger.error(f"Error reading API keys: {e}")
            return []
    
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        """Static method to read a single key, for Google Search compatibility."""
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
        # Logic from original script
        try:
            params = {'key': self.api_key,'cx': self.search_engine_id,'q': query,'num': min(num_results, 10)}
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json().get('items', [])
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
        # Logic from original script
        competitor_data = []
        for keyword in keywords[:3]:
            results = self.google_search.search(f"{keyword} blog", num_results=5)
            for result in results:
                competitor_data.append({'url': result['link'], 'title': result['title']})
            sleep(1)
        return competitor_data[:self.config.competitor_analysis_count]

    def analyze_competitor_content(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Logic from original script
        successful_analyses = []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            futures = [executor.submit(self._analyze_single_competitor, c['url']) for c in competitors]
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    successful_analyses.append(result)
        return self._process_competitor_analyses(successful_analyses) if successful_analyses else {}

    def _analyze_single_competitor(self, url: str) -> Dict[str, Any]:
        # Logic from original script
        try:
            r = self.session.get(url, timeout=self.config.timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            content = soup.get_text(separator=' ', strip=True)
            return {'success': True, 'content': content[:5000], 'word_count': len(content.split())}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _process_competitor_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Logic from original script
        word_counts = [a['word_count'] for a in analyses]
        avg_wc = sum(word_counts) // len(word_counts) if word_counts else 0
        return {'avg_word_count': avg_wc}

class MarkdownFormatter:
    @staticmethod
    def format_blog_output(result: Dict[str, Any], blog_input: BlogInput) -> str:
        # Logic from original script
        return f"# {blog_input.title}\n\n{result['blog_content']}"

# --- MODIFIED: SEOBlogWriter for Key Rotation ---
class SEOBlogWriter:
    """Main SEO blog writing agent with API key rotation."""
    
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.formatter = MarkdownFormatter()
        
        self.api_keys = APIKeyManager.get_api_keys(self.config.api_key_file)
        if not self.api_keys:
            raise ValueError("No Gemini API keys found. Please check your geminaikey file.")
        
        self.key_lock = Lock()
        self.current_key_index = 0
        self.llm = self._get_llm_with_current_key()

    def _get_llm_with_current_key(self):
        """Gets an LLM instance with the current API key."""
        with self.key_lock:
            key_to_use = self.api_keys[self.current_key_index]
        logger.info(f"Initializing LLM with key index: {self.current_key_index}")
        return ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=key_to_use)

    def _rotate_key_and_get_llm(self) -> bool:
        """Rotates to the next API key and returns True if successful."""
        with self.key_lock:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.warning(f"API Quota likely hit. Rotating to key index: {self.current_key_index}")
            
            if self.current_key_index == 0:
                logger.error("All available API keys have been tried and failed. Aborting task.")
                return False
        
        self.llm = self._get_llm_with_current_key()
        return True

    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """Writes a blog post, with retries and key rotation on quota errors."""
        logger.info(f"Starting blog process for: '{blog_input.title}'")
        
        analysis = self.analyzer.analyze_competitor_content(
            self.analyzer.search_competitors(blog_input.primary_keywords)
        )
        prompt = self._build_generation_prompt(blog_input, analysis)
        
        # Retry loop for API keys
        for i in range(len(self.api_keys)):
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                logger.info(f"Content generated for '{blog_input.title}'.")
                return {"success": True, "blog_content": content}
            except ResourceExhausted:
                logger.warning(f"ResourceExhausted error for '{blog_input.title}'.")
                if not self._rotate_key_and_get_llm():
                    break # Stop if all keys have been tried
                sleep(2) # Wait a moment before retrying with the new key
            except Exception as e:
                logger.error(f"Unexpected error for '{blog_input.title}': {e}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "All API keys exhausted their quotas."}

    def _build_generation_prompt(self, blog_input: BlogInput, analysis: Dict[str, Any]) -> str:
        # Logic from original script
        return f"Write a blog post titled '{blog_input.title}' targeting '{blog_input.target_audience}'..."

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts in parallel.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the JSON file with blog inputs.')
    parser.add_argument('--drive', action='store_true', help='Upload generated blog posts to Google Drive.')
    parser.add_argument('--min-words', type=int, default=1500, help='Minimum word count.')
    parser.add_argument('--max-words', type=int, default=3000, help='Maximum word count.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    return parser.parse_args()

# --- MODIFIED: Main function for parallel execution ---
def main():
    args = parse_arguments()
    
    try:
        config = SEOBlogConfig(
            min_word_count=args.min_words,
            max_word_count=args.max_words,
            verbose=args.verbose,
            drive_upload=args.drive
        )
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        writer = SEOBlogWriter(config)
        uploader = GoogleDriveUploader() if config.drive_upload else None

        def process_blog_task(blog_input: BlogInput):
            """Defines the complete task for one blog post."""
            result = writer.write_blog(blog_input)
            
            if result and result.get('success'):
                file_path = writer.save_blog_to_file(result, blog_input)
                upload_msg = ""
                if uploader:
                    drive_link = uploader.upload_blog(file_path, blog_input.title)
                    upload_msg = f" | GDrive Link: {drive_link}" if drive_link else " | GDrive Upload Failed"
                return f"✅ SUCCESS: '{blog_input.title}' -> {file_path}{upload_msg}"
            else:
                error = result.get('error', 'Unknown error')
                return f"❌ FAILED: '{blog_input.title}'. Reason: {error}"

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=config.max_concurrent_requests, thread_name_prefix='BlogBot') as executor:
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
