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

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SEOBlogConfig:
    """Configuration for SEO blog writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.5-pro"
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
    client_context: Optional[str] = None
    client_focus: float = 0.5  # 0.0 = TOFU (awareness), 1.0 = BOFU (conversion)

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
    """Enhanced API key management."""
    
    @staticmethod
    def get_api_key(filepath: str = "geminaikey") -> Optional[str]:
        """Retrieve and validate API key from file."""
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
                
            logger.info(f"API key loaded successfully from {filepath}")
            return api_key
            
        except Exception as e:
            logger.error(f"Error reading API key: {e}")
            return None

class GoogleSearchAPI:
    """Google Custom Search API client."""
    
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
        """
        Perform Google Custom Search.
        
        Args:
            query: Search query
            num_results: Number of results to return (max 10 per request)
            
        Returns:
            List of search results
        """
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),  # Google allows max 10 per request
                'fields': 'items(title,link,snippet,displayLink)'
            }
            
            if self.config.verbose:
                logger.info(f"Executing Google Search for query: {query}")
            
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'domain': item.get('displayLink', '')
                    })
            
            if self.config.verbose:
                logger.info(f"Found {len(results)} results for query: {query}")
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
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def search_competitors(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search for competitor content using Google Custom Search API.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of competitor URLs and metadata
        """
        competitor_data = []
        
        for keyword in keywords[:3]:
            try:
                search_query = f"{keyword} blog article"
                results = self.google_search.search(search_query, num_results=10)
                
                for i, result in enumerate(results[:5]):
                    domain = urlparse(result['url']).netloc.lower()
                    exclude_domains = ['google.com', 'facebook.com', 'twitter.com', 'youtube.com', 'instagram.com', 'wikipedia.org']
                    
                    if not any(exc in domain for exc in exclude_domains):
                        competitor_data.append({
                            'url': result['url'], 'title': result['title'], 'snippet': result['snippet'],
                            'domain': result['domain'], 'keyword': keyword, 'search_position': i + 1
                        })
                
                sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching for keyword '{keyword}': {e}")
                continue
        
        seen_urls = set()
        unique_competitors = []
        for comp in competitor_data:
            if comp['url'] not in seen_urls:
                seen_urls.add(comp['url'])
                unique_competitors.append(comp)
        
        logger.info(f"Found {len(unique_competitors)} unique competitor pages to analyze.")
        return unique_competitors[:self.config.competitor_analysis_count]
    
    def analyze_competitor_content(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze competitor content for SEO insights.
        """
        if self.config.verbose:
            logger.info(f"Analyzing content from {len(competitors)} competitor pages...")
        
        analysis_results = {
            'total_analyzed': 0, 'successful_analyses': 0, 'common_topics': {},
            'keyword_usage': {}, 'content_structures': [], 'avg_word_count': 0,
            'title_patterns': [], 'meta_insights': [], 'competitor_strengths': [],
            'content_gaps': [], 'search_snippets': [comp['snippet'] for comp in competitors if comp.get('snippet')]
        }
        
        successful_analyses = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            future_to_competitor = {executor.submit(self._analyze_single_competitor, comp): comp for comp in competitors}
            
            for future in as_completed(future_to_competitor):
                competitor = future_to_competitor[future]
                try:
                    result = future.result()
                    if result['success']:
                        successful_analyses.append(result)
                        analysis_results['successful_analyses'] += 1
                    analysis_results['total_analyzed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error analyzing competitor {competitor['url']}: {e}")
                    analysis_results['total_analyzed'] += 1
        
        if successful_analyses:
            analysis_results.update(self._process_competitor_analyses(successful_analyses))
        
        logger.info(f"Completed competitor analysis: {analysis_results['successful_analyses']}/{analysis_results['total_analyzed']} pages analyzed successfully.")
        
        return analysis_results
    
    def _analyze_single_competitor(self, competitor: Dict[str, Any]) -> Dict[str, Any]:
        url = competitor['url']
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else competitor.get('title', '')
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            content_text = main_content.get_text(separator=' ', strip=True) if main_content else ""
            
            headings = [{'level': i, 'text': h.get_text().strip()} for i in range(1, 7) for h in soup.find_all(f'h{i}')]
            
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '') if meta_desc else ''
            word_count = len(content_text.split())
            
            return {
                'success': True, 'url': url, 'title': title_text, 'meta_description': meta_description,
                'content': content_text[:2000], 'word_count': word_count, 'headings': headings,
                'keyword_from_search': competitor.get('keyword', ''), 'search_position': competitor.get('search_position', 0),
                'search_snippet': competitor.get('snippet', ''), 'domain': competitor.get('domain', '')
            }
        except Exception as e:
            logger.warning(f"Failed to analyze competitor {url}: {e}")
            return {'success': False, 'url': url, 'error': str(e)}

    def _process_competitor_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process competitor analyses to extract insights."""
        word_counts = [a['word_count'] for a in analyses if a['word_count'] > 0]
        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
        
        all_content = ' '.join([a['content'].lower() for a in analyses])
        words = re.findall(r'\b\w+\b', all_content)
        
        word_freq = {}
        for word in words:
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        common_topics = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        
        title_patterns = [a['title'] for a in analyses if a['title']]
        content_structures = [[h['text'] for h in analysis['headings'][:10]] for analysis in analyses if analysis['headings']]
        
        competitor_strengths = []
        top_performers = sorted(analyses, key=lambda x: x.get('search_position', 10))[:3]
        for performer in top_performers:
            competitor_strengths.append({
                'url': performer['url'], 'title': performer['title'], 'word_count': performer['word_count'],
                'position': performer.get('search_position', 'Unknown'), 'domain': performer.get('domain', ''),
                'snippet': performer.get('search_snippet', '')
            })
        
        all_snippets = ' '.join([a.get('search_snippet', '') for a in analyses]).lower()
        snippet_words = re.findall(r'\b\w+\b', all_snippets)

        snippet_freq = {}
        for word in snippet_words:
            if len(word) > 3:
                snippet_freq[word] = snippet_freq.get(word, 0) + 1

        snippet_keywords = dict(sorted(snippet_freq.items(), key=lambda x: x[1], reverse=True)[:15])
        
        return {
            'avg_word_count': avg_word_count, 'common_topics': common_topics, 'title_patterns': title_patterns,
            'content_structures': content_structures, 'competitor_strengths': competitor_strengths,
            'content_gaps': [], 'snippet_keywords': snippet_keywords
        }

class MarkdownFormatter:
    @staticmethod
    def format_blog_output(result: Dict[str, Any], blog_input: BlogInput) -> str:
        markdown_content = ["---"]
        markdown_content.append(f"title: \"{blog_input.title}\"")
        markdown_content.append(f"brand: {blog_input.brand}")
        markdown_content.append(f"created: {result['metadata']['created_at']}")
        markdown_content.append(f"word_count: {result['metadata']['word_count']}")
        markdown_content.append("primary_keywords:")
        for keyword in blog_input.primary_keywords:
            markdown_content.append(f"  - \"{keyword}\"")
        markdown_content.append("secondary_keywords:")
        for keyword in blog_input.secondary_keywords[:10]:
            markdown_content.append(f"  - \"{keyword}\"")
        markdown_content.append(f"target_audience: \"{blog_input.target_audience}\"")
        markdown_content.append(f"tone: \"{blog_input.tone}\"")
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append(result["blog_content"])
        markdown_content.append("\n---\n")
        markdown_content.append("## SEO Metadata\n")
        markdown_content.append(f"**Target Word Count:** {result['metadata']['target_word_count']}")
        markdown_content.append(f"**Actual Word Count:** {result['metadata']['word_count']}\n")
        markdown_content.append("**Primary Keywords:**")
        for keyword in blog_input.primary_keywords:
            markdown_content.append(f"- {keyword}")
        markdown_content.append("\n**Secondary Keywords:**")
        for keyword in blog_input.secondary_keywords:
            markdown_content.append(f"- {keyword}")
        return "\n".join(markdown_content)

class SEOBlogWriter:
    """Main SEO blog writing agent."""
    
    def __init__(self, config: SEOBlogConfig):
        self.config = config
        self.analyzer = CompetitorAnalyzer(config)
        self.formatter = MarkdownFormatter()
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        api_key = APIKeyManager.get_api_key(self.config.api_key_file)
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        try:
            llm = ChatGoogleGenerativeAI(model=self.config.model_name, google_api_key=api_key, convert_system_message_to_human=True)
            logger.info(f"LLM initialized with model: {self.config.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
            
    def write_blog(self, blog_input: BlogInput) -> Dict[str, Any]:
        """
        Write a complete SEO blog post by orchestrating analysis and generation.
        """
        logger.info(f"Starting blog writing process for title: '{blog_input.title}'")
        
        try:
            logger.info(f"[Phase 1/4 - {blog_input.title}] Analyzing Competitors...")
            competitors = self.analyzer.search_competitors(blog_input.primary_keywords)
            analysis_results = self.analyzer.analyze_competitor_content(competitors)
            logger.info(f"[Phase 1/4 - {blog_input.title}] Competitor analysis complete.")

            logger.info(f"[Phase 2/4 - {blog_input.title}] Building prompt for content generation...")
            prompt = self._build_generation_prompt(blog_input, analysis_results)
            
            logger.info(f"[Phase 3/4 - {blog_input.title}] Generating blog content with LLM... (This may take a moment)")
            response = self.llm.invoke(prompt)
            blog_content = response.content
            logger.info(f"[Phase 3/4 - {blog_input.title}] Content generation complete.")

            logger.info(f"[Phase 4/4 - {blog_input.title}] Finalizing and preparing content.")
            word_count = len(blog_content.split())
            
            result = {
                "success": True, "blog_content": blog_content,
                "metadata": {
                    "brand": blog_input.brand, "title": blog_input.title, "word_count": word_count,
                    "primary_keywords": blog_input.primary_keywords, "secondary_keywords": blog_input.secondary_keywords,
                    "created_at": datetime.now().isoformat(),
                    "target_word_count": f"{self.config.min_word_count}-{self.config.max_word_count}",
                    "competitor_analysis": analysis_results
                }
            }
            
            logger.info(f"Successfully finished writing process for '{blog_input.title}' ({word_count} words)")
            return result
            
        except ResourceExhausted as e:
            logger.error(f"API quota exceeded during blog writing for '{blog_input.title}': {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing '{blog_input.title}': {e}", exc_info=True)
            return {"success": False, "error": str(e), "blog_content": "", "metadata": {}}

    def _build_generation_prompt(self, blog_input: BlogInput, analysis: Dict[str, Any]) -> str:
        analysis_summary = json.dumps({
            "avg_word_count": analysis.get('avg_word_count', 0),
            "common_topics": analysis.get('common_topics', {}),
            "top_competitor_strengths": analysis.get('competitor_strengths', []),
            "identified_content_gaps": analysis.get('content_gaps', []),
            "important_snippet_keywords": analysis.get('snippet_keywords', {})
        }, indent=2)

        # Determine funnel stage and client focus based on client_focus parameter
        funnel_stage = self._get_funnel_stage_description(self.config.client_focus)
        client_focus_guidance = self._get_client_focus_guidance(self.config.client_focus)

        # Build client context section if available
        client_context_section = ""
        if self.config.client_context:
            client_context_section = f"""
---
**CLIENT WEBSITE CONTENT:**
The following is a dump of the client's website content. Use this to understand the client's brand voice, products/services, unique value propositions, expertise areas, and target market. Integrate relevant aspects naturally throughout the blog post where appropriate. Ensure the content aligns with the client's messaging and positioning.

{self.config.client_context}

**Important:** {client_focus_guidance}
"""

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
{client_context_section}
---
**TONE AND STYLE INSTRUCTIONS (OVERRIDE)**
You must follow these specific instructions for the tone and style, which are more important than the general tone listed above.

**1. NLP-Friendly Content Prompt (60% Focus):**
Create content strictly adhering to an NLP-friendly format, emphasizing clarity and simplicity in structure and language. Ensure sentences follow a straightforward subject-verb-object order, selecting words for their precision and avoiding any ambiguity. Exclude filler content, focusing on delivering information succinctly. Do not use complex or abstract terms such as 'meticulous,' 'navigating,' 'complexities,' 'realm,' 'bespoke,' 'tailored,' 'towards,' 'underpins,' 'ever-changing,' 'ever-evolving,' 'the world of,' 'not only,' 'seeking more than just,' 'designed to enhance,' 'it's not merely,' 'our suite,' 'it is advisable,' 'daunting,' 'in the heart of,' 'when it comes to,' 'in the realm of,' 'amongst,' 'unlock the secrets,' 'unveil the secrets,' and 'robust.' This approach aims to streamline content production for enhanced NLP algorithm comprehension, ensuring the output is direct, accessible, and easily interpretable.

**2. Humanizing Prompt (40% Focus):**
While prioritizing NLP-friendly content creation (60%), also dedicate 40% of your focus to making the content engaging and enjoyable for readers, balancing technical NLP-optimization with reader satisfaction to produce content that not only ranks well on search engines well is also compelling and valuable to a readership.

---
**COMPETITOR ANALYSIS INSIGHTS:**
Here is a summary of the top-ranking competitor content. Use these insights to create a superior article that covers the topic more comprehensively and fills any identified gaps.

{analysis_summary}

---
**YOUR TASK:**
Write the full blog post now. Adhere strictly to all instructions, especially the tone and style guidelines. Ensure the post starts with a compelling introduction, has a well-structured body with actionable advice and examples, and ends with a strong conclusion. Use the competitor analysis to guide your structure, depth, and topics, but ensure your content is unique and more valuable.

**FUNNEL STAGE FOCUS:** {funnel_stage}

**BEGIN BLOG POST:**
"""
    
    def _get_funnel_stage_description(self, client_focus: float) -> str:
        """Generate funnel stage description based on client_focus value."""
        if client_focus <= 0.2:
            return "TOFU (Top of Funnel - Awareness Stage): Focus on educational content, building awareness, and addressing broad topics. Minimize direct product/service mentions. The goal is to inform and engage readers who are just discovering the problem space."
        elif client_focus <= 0.4:
            return "Upper-MOFU (Middle of Funnel - Early Consideration): Provide educational content with light mentions of solution types. Subtly introduce the client's expertise areas. Balance between information and positioning."
        elif client_focus <= 0.6:
            return "MOFU (Middle of Funnel - Consideration Stage): Balance educational content with clear positioning of the client's solutions. Readers are evaluating options. Show how the client's approach addresses specific needs while still providing valuable general information."
        elif client_focus <= 0.8:
            return "Lower-MOFU to BOFU (Bottom of Funnel - Decision Stage): Emphasize the client's specific solutions, unique value propositions, and competitive advantages. Include clear CTAs and demonstrate why the client is the right choice. Still provide value but with strong commercial intent."
        else:
            return "BOFU (Bottom of Funnel - Conversion Stage): Strongly emphasize the client's products/services, unique differentiators, case studies, and specific solutions. Include multiple CTAs and conversion-focused messaging. The primary goal is to drive action and conversions."
    
    def _get_client_focus_guidance(self, client_focus: float) -> str:
        """Generate client focus guidance based on client_focus value."""
        if client_focus <= 0.2:
            return "Keep client mentions minimal and subtle. Focus on providing valuable, educational content that establishes thought leadership without being promotional. The client's brand should be present but not dominant."
        elif client_focus <= 0.4:
            return "Mention the client's expertise and capabilities naturally but avoid hard selling. Position the client as a knowledgeable resource while keeping the content primarily educational."
        elif client_focus <= 0.6:
            return "Balance educational value with clear positioning of the client's solutions. Reference the client's offerings, expertise, and brand voice naturally throughout the content. The content should feel cohesive with the client's overall brand messaging."
        elif client_focus <= 0.8:
            return "Prominently feature the client's solutions, services, and unique value propositions. Include specific examples of how the client solves problems. The content should clearly guide readers toward the client as a solution provider."
        else:
            return "Heavily emphasize the client's products, services, unique advantages, and specific solutions throughout the content. Include compelling reasons to choose this client over competitors. Use strong, conversion-focused language and multiple calls-to-action."
    
    def save_blog_to_file(self, result: Dict[str, Any], blog_input: BlogInput) -> str:
        """Saves the blog content to a markdown file named after the title."""
        safe_title = re.sub(r'[\s\W]+', '_', blog_input.title.lower()).strip('_')
        base_filename = safe_title
        
        max_len = 100
        if len(base_filename) > max_len:
            base_filename = base_filename[:max_len].rstrip('_')
            
        filename = f"{base_filename}.md"
        markdown_content = self.formatter.format_blog_output(result, blog_input)
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Content for '{blog_input.title}' saved to: {output_path.resolve()}")
        return str(output_path.resolve())

def load_blog_inputs_from_file(filepath: str) -> List[BlogInput]:
    """Loads a list of blog input data from a JSON file."""
    try:
        logger.info(f"Loading blog inputs from config file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise TypeError("The JSON file must contain a list of blog post objects.")
            
        return [BlogInput(**item) for item in data]
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the configuration file: {filepath}")
        raise
    except TypeError as e:
        logger.error(f"Mismatch between JSON keys and BlogInput fields: {e}")
        raise

def load_client_context(filepath: str) -> Optional[str]:
    """Load client website content dump from a text file."""
    try:
        client_path = Path(filepath)
        if not client_path.exists():
            logger.error(f"Client website content file not found at '{filepath}'")
            return None
        
        with open(client_path, 'r', encoding='utf-8') as f:
            client_context = f.read().strip()
        
        if not client_context:
            logger.warning(f"Client website content file '{filepath}' is empty")
            return None
        
        # Truncate if too long to avoid context overflow (keep first 8000 characters)
        #    max_length = 8000
        #    if len(client_context) > max_length:
        #        logger.info(f"Client content is large ({len(client_context)} chars), truncating to {max_length} characters")
        #        client_context = client_context[:max_length] + "\n\n[Content truncated for length...]"
            
        #    logger.info(f"Client website content loaded successfully from {filepath} ({len(client_context)} characters)")
        return client_context
        
    except Exception as e:
        logger.error(f"Error reading client website content file: {e}")
        return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate SEO-optimized blog posts from a JSON file. Each post is saved as 'title_name.md'.")
    
    parser.add_argument(
        '-c', '--config-file', type=str, required=True,
        help='Path to the JSON file containing a list of blog input data.'
    )
    parser.add_argument(
        '--api-key-file', type=str, default='geminaikey',
        help='Path to Gemini API key file (default: geminaikey)'
    )
    parser.add_argument(
        '--google-search-api', type=str, default='googlesearchapi',
        help='Path to Google Search API key file (default: googlesearchapi)'
    )
    parser.add_argument(
        '--google-cx', type=str, default='googlecx',
        help='Path to Google Custom Search Engine ID file (default: googlecx)'
    )
    parser.add_argument(
        '--client', type=str, default=None,
        help='Path to a text file containing client website content dump for content optimization (optional)'
    )
    parser.add_argument(
        '--client-focus', type=float, default=0.5,
        help='Client focus level from 0.0 (TOFU/awareness) to 1.0 (BOFU/conversion). Default: 0.5 (MOFU/balanced)'
    )
    parser.add_argument(
        '--min-words', type=int, default=1500,
        help='Minimum word count (default: 1500)'
    )
    parser.add_argument(
        '--max-words', type=int, default=3000,
        help='Maximum word count (default: 3000)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output to track processing stages.'
    )
    
    return parser.parse_args()

def main():
    """Main function to run the SEO blog writer."""
    args = parse_arguments()
    
    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)
        
        # Validate client_focus parameter
        if args.client_focus < 0.0 or args.client_focus > 1.0:
            logger.error("--client-focus must be between 0.0 and 1.0")
            return
        
        # Load client context if provided
        client_context = None
        if args.client:
            client_context = load_client_context(args.client)
            if client_context:
                focus_description = "TOFU (awareness)" if args.client_focus <= 0.3 else \
                                  "MOFU (consideration)" if args.client_focus <= 0.7 else \
                                  "BOFU (conversion)"
                logger.info(f"Client website content will be used for content optimization with {focus_description} focus (level: {args.client_focus})")
        
        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            min_word_count=args.min_words,
            max_word_count=args.max_words,
            verbose=args.verbose,
            client_context=client_context,
            client_focus=args.client_focus
        )

        writer = SEOBlogWriter(config)

        for i, blog_input in enumerate(blog_inputs):
            logger.info(f"--- Starting job {i+1} of {len(blog_inputs)}: '{blog_input.title}' ---")
            
            for attempt in range(config.max_retries):
                try:
                    result = writer.write_blog(blog_input)
                    
                    if result and result.get('success'):
                        saved_file = writer.save_blog_to_file(result, blog_input)
                        print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' complete. File saved to: {saved_file}\n")
                        break 
                    else:
                        logger.error(f"Failed to generate blog post for '{blog_input.title}'.")
                        if 'error' in result:
                            logger.error(f"Reason: {result['error']}")
                        break

                except ResourceExhausted as e:
                    if attempt < config.max_retries - 1:
                        logger.warning(f"API quota exceeded. Retrying in {config.retry_delay} seconds... (Attempt {attempt + 1}/{config.max_retries})")
                        sleep(config.retry_delay)
                    else:
                        logger.error("API quota exceeded and max retries reached. Please check your billing plan or wait for the quota to reset.")
                        print(f"\n❌ ERROR: Could not generate blog post for '{blog_input.title}' due to API rate limits.\n")
                        break
                
                except Exception as e:
                    logger.error(f"An unexpected error occurred during blog generation for '{blog_input.title}': {e}", exc_info=True)
                    break

    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Could not start blog generation due to a configuration error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main process: {e}", exc_info=True)


if __name__ == "__main__":
    main()
