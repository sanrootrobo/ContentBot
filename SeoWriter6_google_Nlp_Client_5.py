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

# --- Third Party Imports ---
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# --- Google Drive API Imports ---
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration & Data Classes ---

@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    drive_credentials_file: str = "credentials.json"
    drive_token_file: str = "token.json"
    model_name: str = "gemini-2.5-pro"
    timeout: int = 30
    max_content_length: int = 10000
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    max_concurrent_requests: int = 5
    competitor_analysis_count: int = 10
    min_word_count: int = 1500
    max_word_count: int = 3000
    client_context: Optional[str] = None
    client_focus: float = 0.5

@dataclass
class BlogInput:
    """Input data structure for blog writing."""
    brand: str
    title: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    target_audience: str = "General Audience"
    tone: str = "Informative"
    
    def __post_init__(self):
        if not self.brand or not self.title:
            raise ValueError("Brand and title are required")
        # Ensure keywords are lists (handles both JSON list and CLI string input)
        if isinstance(self.primary_keywords, str):
            self.primary_keywords = [k.strip() for k in self.primary_keywords.split(',')]
        if isinstance(self.secondary_keywords, str):
            self.secondary_keywords = [k.strip() for k in self.secondary_keywords.split(',')]

# --- Google Drive Logic ---

class GoogleDriveManager:
    """Manages Google Drive authentication and uploads."""
    SCOPES = ['https://www.googleapis.com/auth/drive']

    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.service = None

    def authenticate(self):
        creds = None
        if os.path.exists(self.config.drive_token_file):
            creds = Credentials.from_authorized_user_file(self.config.drive_token_file, self.SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                except Exception:
                    creds = None

            if not creds:
                if not os.path.exists(self.config.drive_credentials_file):
                    logger.error(f"❌ Drive credentials file not found: {self.config.drive_credentials_file}")
                    return False
                flow = InstalledAppFlow.from_client_secrets_file(self.config.drive_credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.config.drive_token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        return True

    def remove_seo_metadata(self, md_file: str) -> str:
        """Creates a clean version of the file without YAML front matter/SEO logs for Doc upload."""
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove YAML
        content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
        # Remove SEO Logs section
        seo_patterns = [
            r'## SEO Metadata.*?(?=\n#|\Z)', 
            r'SEO\s*Metadata.*?(?=\n#|\Z)', 
            r'Target\s*Word\s*Count:.*?(?=\n#|\Z)',
            r'Primary\s*Keywords:.*?(?=\n#|\Z)'
        ]
        for pattern in seo_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        temp_file = f"temp_{os.path.basename(md_file)}"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n")
        return temp_file

    def upload_md_to_gdoc(self, md_file: str, title: str) -> Optional[str]:
        if not self.service:
            if not self.authenticate(): return None

        temp_file = self.remove_seo_metadata(md_file)
        file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
        media = MediaFileUpload(temp_file, mimetype='text/markdown', resumable=True)
        
        try:
            logger.info(f"Uploading '{title}' to Drive...")
            file = self.service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
            
            # Set permissions to 'anyone with link can edit'
            self.service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'writer'}).execute()
            
            if os.path.exists(temp_file): os.remove(temp_file)
            return file.get('webViewLink')
        except Exception as e:
            logger.error(f"❌ Upload Error: {e}")
            if os.path.exists(temp_file): os.remove(temp_file)
            return None
    
    @staticmethod
    def update_upload_report(title: str, link: str, report_file: str):
        """Updates the text file with Title -> Link mapping."""
        try:
            # Append mode 'a' ensures we add to the list, not overwrite it
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write(f"{title} -> {link}\n")
            print(f"📊 Report updated: {report_file}")
        except Exception as e:
            print(f"❌ Error updating report: {e}")

# --- Search & Analysis ---

class APIKeyManager:
    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        if not os.path.exists(filepath): return None
        with open(filepath, 'r', encoding='utf-8') as f: return f.read().strip()

class GoogleSearchAPI:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.cx = APIKeyManager.get_api_key(config.google_cx_file)
        if not self.api_key or not self.cx: raise ValueError("Missing Google Search API keys")

    def search(self, query: str) -> List[Dict]:
        try:
            params = {'key': self.api_key, 'cx': self.cx, 'q': query, 'num': 10}
            res = requests.get("https://www.googleapis.com/customsearch/v1", params=params).json()
            return [{'title': i.get('title'), 'url': i.get('link'), 'snippet': i.get('snippet')} for i in res.get('items', [])]
        except Exception: return []

class CompetitorAnalyzer:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.search = GoogleSearchAPI(config)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})

    def analyze(self, keywords: List[str]) -> Dict[str, Any]:
        logger.info(f"🔍 Analyzing competitors for: {keywords[0]}...")
        urls = []
        for kw in keywords[:2]:
            results = self.search.search(f"{kw} blog article")
            valid_results = [r['url'] for r in results if not any(x in r['url'] for x in ['youtube', 'facebook', 'instagram', 'twitter'])]
            urls.extend(valid_results[:3])
        
        analyses = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(self._scrape, url): url for url in set(urls)}
            for f in as_completed(futures):
                if res := f.result(): analyses.append(res)
        
        if not analyses:
            return {}

        return {
            "avg_word_count": sum(a['word_count'] for a in analyses) / len(analyses),
            "top_competitors": [{'title': a['title'], 'url': a['url']} for a in analyses[:3]]
        }

    def _scrape(self, url):
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200: return None
            soup = BeautifulSoup(resp.text, 'html.parser')
            for x in soup(['script', 'style', 'nav', 'footer']): x.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return {'url': url, 'title': soup.title.string if soup.title else url, 'word_count': len(text.split())}
        except: return None

# --- Content Generation ---

class SEOBlogWriter:
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name, 
            google_api_key=APIKeyManager.get_api_key(config.api_key_file)
        )

    def generate(self, inp: BlogInput) -> Dict:
        try:
            analysis = self.analyzer.analyze(inp.primary_keywords)
            prompt = self._build_prompt(inp, analysis)
            logger.info(f"✍️ Generating content for: {inp.title}")
            response = self.llm.invoke(prompt)
            
            word_count = len(response.content.split())
            return {
                "success": True, 
                "content": response.content,
                "metadata": {
                    "title": inp.title, "word_count": word_count,
                    "keywords": inp.primary_keywords
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _build_prompt(self, inp: BlogInput, analysis: Dict) -> str:
        comp_summary = json.dumps(analysis.get('top_competitors', []), indent=2)
        return f"""
        You are an expert SEO writer for the brand '{inp.brand}'.
        Write a blog post titled: "{inp.title}"
        
        **Requirements:**
        - Primary Keywords: {', '.join(inp.primary_keywords)}
        - Secondary Keywords: {', '.join(inp.secondary_keywords)}
        - Audience: {inp.target_audience}
        - Tone: {inp.tone}
        - Length: {self.config.min_word_count}-{self.config.max_word_count} words
        - Format: Markdown (Use H1, H2, H3)
        - Include 4 FAQs at the end.

        **Competitor Context:**
        The top ranking articles are: {comp_summary}
        Make this article better and more comprehensive than them.

        **Content Style:**
        - 60% NLP Friendly (Clear, simple sentences).
        - 40% Humanizing (Engaging, no fluff).
        
        Output ONLY the markdown content.
        """

class MarkdownFormatter:
    @staticmethod
    def format(result: Dict, inp: BlogInput) -> str:
        md = [
            "---",
            f"title: \"{inp.title}\"",
            f"brand: \"{inp.brand}\"",
            f"keywords: {json.dumps(inp.primary_keywords)}",
            "---",
            "",
            result['content'],
            "\n---\n## SEO Metadata",
            f"**Word Count:** {result['metadata']['word_count']}",
            f"**Primary Keywords:** {', '.join(inp.primary_keywords)}"
        ]
        return "\n".join(md)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Batch SEO Blog Generator")
    parser.add_argument('--batch_file', type=str, default='topics.json', help='JSON file containing blog topics')
    parser.add_argument('--drive', action='store_true', help='Upload to Google Drive')
    parser.add_argument('--dir', type=str, default='output', help='Output folder')
    parser.add_argument('--report', type=str, default='upload_report.txt', help='Report file')
    args = parser.parse_args()

    # Load Configuration
    config = SEOBlogConfig()
    if not os.path.exists(config.api_key_file):
        print("❌ Error: API Key files missing.")
        return

    # Load Batch File
    if not os.path.exists(args.batch_file):
        print(f"❌ Error: Batch file '{args.batch_file}' not found.")
        return

    with open(args.batch_file, 'r', encoding='utf-8') as f:
        try:
            topics = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error: '{args.batch_file}' is not valid JSON.")
            return

    print(f"🚀 Found {len(topics)} topics to process in {args.batch_file}")
    
    writer = SEOBlogWriter(config)
    
    if args.drive:
        drive = GoogleDriveManager(config)

    # Batch Loop
    for index, topic in enumerate(topics, 1):
        print(f"\n--------------------------------------------------")
        print(f"🔄 Processing {index}/{len(topics)}: {topic.get('title')}")
        
        try:
            # Map JSON to Input Object
            blog_input = BlogInput(
                brand=topic.get('brand', 'MyBrand'),
                title=topic.get('title'),
                primary_keywords=topic.get('primary_keywords', []),
                secondary_keywords=topic.get('secondary_keywords', []),
                target_audience=topic.get('target_audience', 'General'),
                tone=topic.get('tone', 'Informative')
            )

            # Generate Content
            result = writer.generate(blog_input)

            if result['success']:
                # Save Locally
                if not os.path.exists(args.dir): os.makedirs(args.dir)
                clean_filename = "".join(x for x in blog_input.title if x.isalnum() or x in " -_").strip().replace(" ", "_")
                filepath = os.path.join(args.dir, f"{clean_filename}.md")
                
                formatted_content = MarkdownFormatter.format(result, blog_input)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
                print(f"✅ Saved locally: {filepath}")

                # Upload to Drive & Update Report
                if args.drive:
                    link = drive.upload_md_to_gdoc(filepath, blog_input.title)
                    if link:
                        # 1. Append link to local file for reference
                        with open(filepath, 'a', encoding='utf-8') as f:
                            f.write(f"\n\n**Google Doc:** {link}")
                        
                        # 2. Append to master report file (The Fix)
                        drive.update_upload_report(blog_input.title, link, args.report)
            else:
                print(f"❌ Failed to generate: {result.get('error')}")

        except Exception as e:
            print(f"❌ Error processing topic: {e}")
        
        # Rate limit safe-guard
        if index < len(topics):
            print("⏳ Sleeping 5 seconds before next topic...")
            sleep(5)

    print("\n🎉 Batch processing complete! Check upload_report.txt for links.")

if __name__ == '__main__':
    main()
