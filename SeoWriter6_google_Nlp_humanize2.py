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
import sys

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Imports for the --humanize feature
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

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
        """
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),
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

class OllamaHumanizer:
    """
    Uses a local Ollama model to rephrase markdown content paragraph by paragraph
    based on a predefined stylistic system prompt, with a progress bar.
    """
    def __init__(self):
        self.system_prompt = """
        Remember this writing style and response in it : Tittle: Best Online Platforms for Buying and Renting Property in India

        Primary keyword: property rental websites

        Secondary keywords: free rental property listing websites, best rental property websites

        Introduction

        Traditionally, the process of buying and renting properties has been an obstacle when it comes to better quality of transparency and accessibility to the resources or toll that the modern society has or the technology that prevails due to the property rental websites.

        In the Real Estate sector, the transactions have evolved through complex legal procedure  and it also imposes certain kinds of regulations which make it complex and intricate, whether one is purchasing a property for the sake of investment or residential reasons.

        But, due to the emergence of technology, efficiency of the internet and other such tools have made it easier for home buyers and investors to indulge into the transaction.
        These tools include various kinds of online platforms such as property listing sites, which facilitate the buying, selling, renting, investing or leasing of the properties in the real estate.


        Know Why the use of  property rental online websites for buying and rental properties?

        In the modern times, as aforementioned it becomes important to inform home buyers and investors to be aware of the various platforms in the real estate market in India looking at the contemporary need and status quo of the residential industry.

        Best Online Platforms for Buying and Renting Property in India

        This is list of reasons why modern investors and home buyers should look forward to use renting and buying platforms  such as the property rental websites :

        Extensive property listing.

        Property rental websites when it comes Online real estate markets and portals pertaining to the same provide a wide range of property options that allow users to match their choices and preferences thereby increasing the likelihood of the market.


        Virtual Tours.

        Many online websites and platforms provide and offer virtual tours and high quality images, which results in the enabling of the potential buyers and renters to observe the property fluctuations making  property rental websites more accessible   .

        Cost Efficiency.

        Online  and virtual real estate agencies operate with overhead costs to be compared with the traditional rental approach. The monetary savings can reduce the fees for the client
        Making the very acquisition process affordable.

        Time saving search capabilities.

          	These kind of tools help to filter the results which is based on a specific criteria such as
        the location, price range and the property type lets the users easily meet their need which reduces the time which is searching the property.

        Now let us focus on free property rental websites  sites for buying and rental process :

        99 acres.

        This site is a popular real estate platform when it comes to property seekers like sellers and landlords. This site is launched in the year 2025, this site deals with the aspect of the needs of the consumers in the real estate industry.

        In addition to providing an online platform to real estate developers, brokers and property owners, to list their property for sale, purchase a rent, in this case 99acres.com offers advertisement offers a tenure, banners, home page links and projects for better visibility and branding market.

        The features of this site are that it covers major cities across India which enhances the very exploration of the property. Furthermore, enhancing premium options stand out amongst others attracting more attention.

        Magicbrick.

        As the largest platform connecting property buyers and sellers, magic brick renders over 2 core monthly visitors and 15 lack active property listing.

        The site provides a platform for buying, selling and renting property.
        Feature advanced search options for buyers to find properties that detailed descriptions and multiple images.

        Housing.

        This website was founded in the year 2012 and obtained the title of being India's no.1 app serving the demands of the urban professionals, landlords, developers and real estate brokers or commercial spaces.

        Housing.com with its objective in play also has various kinds of features that it renders to the consumers which makes it more transparent to use such as it showcases your property through professional photographs and virtual tours, it ensures that all the listings are verified which enhances the credibility of your property,suitable filters make the narrowing of the search process of the property more proficient and increasing the chances of the match.

        Nobroker.

        Nobroker is an uninterrupted real estate platform that makes the framework of buying, selling and renting any kind of property more transparent.
        This site eliminates the need of intermediaries, making  transactions more cost effective.

        It provides detailed information about your property, helping buyers make their informed decisions. If your property is for rent no broker offers your nobroker offers tenant screening services for added security.

        Communicate directly with buyers with real time chat features,moreover buyers can access detailed insights about neighbourhood, including safety ratings through real time features.

        Buyers can access detailed insights about the neighbourhood, including safety ratings and local services.

        Makaan.

        The sole purpose of this site is to provide everywhere they are searching the home to call their own. So it begins by partnering with our customers from the start and being there when it matters the most- right from the online search to brokers to home loans to paperwork to finally finding that perfect home. At Makaan.com, we help you find joy.

        One of the best features this includes is that it offers tools for sellers /to highlight the unique features of their properties. Thereafter, it offers a convenient mobile app that allows sellers and buyers to browse and inquire about your property on the go.

        Square yards.

        Square yard is the India's largest integrated platform from real estate & mortgages and one of the fastest growing proptech in UAE, Rest of the middle east, which offers an integrated consumer experience and covers
        the full real-estate journey from search and discovery,transactions, home loans, interior, rentals, property management and post-sales services.

        Conclusion.

        Conclusively, it becomes quite clear that the virtual process of buying and renting property in India. However, it becomes quite important to research by using this website and platform to ensure that the process is efficient.
        However, this makes the property rental websites more transparent.

        FAQ’s

        Are these property listing sites free for users?

        Yes, most property listing sites offer basic services for free, but some premium services come with a cost.

         How do I avoid scams when searching for properties online ?
        Stay cautious by verifying property details, dealing directly with property owners, and using secure payment methods.
        """
        self.llm = self._initialize_ollama()

    def _initialize_ollama(self):
        """Initializes and configures the Ollama chat model."""
        try:
            llm = ChatOllama(
                model="openhermes:v2.5",
                system=self.system_prompt,
                seed=42,
                stop=["\n\n", "Conclusion:"],
                temperature=0.6,
                mirostat=0,
                mirostat_eta=0.1,
                mirostat_tau=8,
                top_k=40,
                top_p=1,
                tfs_z=1.0,
                repeat_penalty=1.6,
                repeat_last_n=64,
                num_ctx=4096,
                num_predict=1024,
                num_gpu=1,
                num_thread=8,
                use_mmap=True,
                use_mlock=False,
            )
            logger.info("Ollama model 'openhermes' initialized successfully for humanizing step.")
            return llm
        except Exception as e:
            logger.error(f"Error initializing Ollama. Is the server running and the 'openhermes' model installed? Error: {e}")
            return None

    def rephrase_content(self, original_content: str) -> str:
        """
        Takes markdown text, separates headings, rephrases paragraphs,
        and reconstructs the document, showing a progress bar.
        """
        if not self.llm:
            logger.error("Ollama model not initialized. Skipping rephrasing.")
            return original_content

        content_parts = re.split(r'(?m)^(#{1,3} .*)$', original_content)
        final_document_parts = []
        
        # Filter out empty parts to get an accurate total for the progress bar
        parts_to_process = [p for p in content_parts if p and not p.isspace()]
        total_parts_to_process = len(parts_to_process)

        if total_parts_to_process == 0:
            logger.info("--- No content found to humanize. ---")
            return ""

        processed_count = 0
        for part in content_parts:
            if not part or part.isspace():
                continue

            processed_count += 1

            if re.match(r'^(#{1,3} .*)$', part.strip()):
                final_document_parts.append(part.strip())
            else:
                paragraph_to_rephrase = part.strip()
                messages = [
                    HumanMessage(
                        content=f"rephrase this paragraph, keeping the bold keywords intact: \n\n---\n\n{paragraph_to_rephrase}"
                    ),
                ]

                try:
                    rephrased_paragraph = "".join([chunk.content for chunk in self.llm.stream(messages)])
                    final_document_parts.append(rephrased_paragraph)
                except Exception as e:
                    logger.error(f"\nAn error occurred while invoking the Ollama model for a paragraph: {e}")
                    final_document_parts.append(paragraph_to_rephrase)

            # --- PROGRESS BAR LOGIC ---
            progress = (processed_count / total_parts_to_process) * 100
            sys.stdout.write(f"\r--- [Humanizing] Rephrasing progress: {progress:.2f}% ---")
            sys.stdout.flush()
        
        # Print a newline at the end to move past the progress bar
        sys.stdout.write('\n')

        return "\n\n".join(final_document_parts)


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
    parser.add_argument(
        '--humanize', action='store_true',
        help='(Optional) After generation, rephrase the content using a local Ollama model to apply a specific, fine-tuned writing style. Requires Ollama and the "openhermes" model to be installed and running.'
    )

    return parser.parse_args()

def main():
    """Main function to run the SEO blog writer."""
    args = parse_arguments()

    try:
        blog_inputs = load_blog_inputs_from_file(args.config_file)

        config = SEOBlogConfig(
            api_key_file=args.api_key_file,
            google_search_api_file=args.google_search_api,
            google_cx_file=args.google_cx,
            min_word_count=args.min_words,
            max_word_count=args.max_words,
            verbose=args.verbose
        )

        writer = SEOBlogWriter(config)
        humanizer = None
        if args.humanize:
            logger.info("--- Humanize option enabled. Initializing Ollama Humanizer... ---")
            humanizer = OllamaHumanizer()
            if not humanizer.llm:
                logger.error("Ollama Humanizer could not be initialized. The --humanize step will be skipped.")

        for i, blog_input in enumerate(blog_inputs):
            logger.info(f"--- Starting job {i+1} of {len(blog_inputs)}: '{blog_input.title}' ---")

            for attempt in range(config.max_retries):
                try:
                    result = writer.write_blog(blog_input)

                    if result and result.get('success'):
                        # --- HUMANIZER STEP ---
                        if args.humanize and humanizer and humanizer.llm:
                            logger.info(f"--- [Humanizing] Rephrasing content for '{blog_input.title}' with Ollama... ---")
                            try:
                                original_text = result['blog_content']
                                rephrased_text = humanizer.rephrase_content(original_text)
                                result['blog_content'] = rephrased_text
                                new_word_count = len(rephrased_text.split())
                                result['metadata']['word_count'] = new_word_count
                                logger.info(f"--- [Humanizing] Content successfully rephrased. New word count: {new_word_count} ---")
                            except Exception as e:
                                logger.error(f"An error occurred during the humanizing process. The original content will be saved. Error: {e}")
                        
                        saved_file = writer.save_blog_to_file(result, blog_input)
                        print(f"\n✅ SUCCESS: Blog post '{blog_input.title}' complete. File saved to: {saved_file}\n")
                        break
                    else:
                        logger.error(f"Failed to generate blog post for '{blog_input.title}'.")
                        if 'error' in result:
                            logger.error(f"Reason: {result['error']}")
                        break

                except ResourceExhausted:
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
