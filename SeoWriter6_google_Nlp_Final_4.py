import os
import requests
import logging
import argparse
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import threading

from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class GuestPostConfig:
    """Configuration for guest post writing."""
    api_key_file: str = "geminaikey"
    google_search_api_file: str = "googlesearchapi"
    google_cx_file: str = "googlecx"
    model_name: str = "gemini-2.0-flash-exp"  # Updated to latest model
    timeout: int = 15
    max_retries: int = 3
    retry_delay: float = 60.0
    verbose: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    max_workers: int = 3  # Reduced for guest posts (fewer concurrent)
    target_word_count: Tuple[int, int] = (900, 1300)  # Min, Max
    premium_word_count: Tuple[int, int] = (700, 900)  # For high-authority mags

@dataclass
class GuestPostInput:
    """Input data structure for guest post writing."""
    brand: str
    project_name: str  # New: Specific project/concept to highlight
    title: str
    primary_keyword: str  # Single keyword for focus
    secondary_keywords: List[str]
    target_publication: str  # e.g., "Architectural Digest India"
    publication_style: str = "premium"  # "premium" or "standard"
    tone: str = "Sophisticated, narrative-driven, aspirational"
    unique_angle: str = ""
    backlink_url: str = ""  # URL for anchor text
    backlink_anchor_text: str = ""  # Text for the link
    author_bio: str = ""  # Optional author bio
    
    def __post_init__(self):
        if not self.brand or not self.title:
            raise ValueError("Brand and title are required")
        if not self.primary_keyword:
            raise ValueError("Primary keyword is required")

@dataclass
class CompetitorInsight:
    """Streamlined competitor data for guest posts."""
    url: str
    title: str
    word_count: int
    h2_structure: List[str]
    key_themes: List[str]
    tone: str
    unique_angles: List[str]

@dataclass
class GuestPostResearch:
    """Research data for guest post."""
    industry_trends: List[Dict[str, str]]
    authority_sources: List[Dict[str, str]]  # For external linking
    competitor_insights: List[CompetitorInsight]
    content_gaps: List[str]
    
# --- Utility Classes ---
class RotatingAPIKeyManager:
    """Thread-safe API key rotation manager."""
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("API key list cannot be empty.")
        self.keys = keys
        self._lock = threading.Lock()
        self._current_index = 0
        logger.info(f"Initialized with {len(self.keys)} API key(s).")

    def get_key(self) -> str:
        with self._lock:
            return self.keys[self._current_index]

    def rotate_key(self):
        with self._lock:
            initial_index = self._current_index
            self._current_index = (self._current_index + 1) % len(self.keys)
            logger.warning(f"Rotated API key from index {initial_index} to {self._current_index}.")
            return self.keys[self._current_index]
            
    @property
    def key_count(self):
        return len(self.keys)

class APIKeyManager:
    @staticmethod
    def get_api_keys(filepath: str) -> List[str]:
        try:
            key_path = Path(filepath)
            if not key_path.exists(): 
                logger.error(f"API key file not found: '{filepath}'")
                return []
            with open(key_path, 'r', encoding='utf-8') as f: 
                keys = [line.strip() for line in f if line.strip()]
            
            if not keys:
                logger.error(f"No valid API keys found in {filepath}")
                return []
            
            logger.info(f"Successfully loaded {len(keys)} API keys from {filepath}.")
            return keys
        except Exception as e: 
            logger.error(f"Error reading API keys: {e}")
            return []

    @staticmethod
    def get_api_key(filepath: str) -> Optional[str]:
        return (APIKeyManager.get_api_keys(filepath) or [None])[0]

class GoogleSearchAPI:
    """Google Custom Search implementation."""
    def __init__(self, config: GuestPostConfig):
        self.config = config
        self.api_key = APIKeyManager.get_api_key(config.google_search_api_file)
        self.search_engine_id = APIKeyManager.get_api_key(config.google_cx_file)
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.google_available = bool(self.api_key and self.search_engine_id)
        
        if not self.google_available:
            logger.warning("Google Search API not configured.")

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        if not self.google_available:
            return []
            
        try:
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': num_results
            }
            
            if self.config.verbose:
                logger.info(f"Searching Google for: {query}")
            
            response = requests.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('items', [])
        except Exception as e:
            logger.warning(f"Google Search error for '{query}': {e}")
            return []

# --- LLM Handler Base ---
class BaseLLMHandler:
    """Base class for LLM-powered components."""
    def __init__(self, config: GuestPostConfig, key_manager: RotatingAPIKeyManager, writer_invoker):
        self.config = config
        self.key_manager = key_manager
        self._invoke_llm = writer_invoker

# --- Guest Post Specific Components ---
class GuestPostResearcher(BaseLLMHandler):
    """Specialized researcher for guest post content."""
    
    def __init__(self, config: GuestPostConfig, key_manager: RotatingAPIKeyManager, 
                 writer_invoker, search_api: GoogleSearchAPI):
        super().__init__(config, key_manager, writer_invoker)
        self.search_api = search_api
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.user_agent})
    
    def gather_research(self, post_input: GuestPostInput) -> GuestPostResearch:
        """Gather all research needed for guest post."""
        logger.info("Gathering research for guest post...")
        
        # 1. Find industry trends and recent news
        trends = self._find_industry_trends(post_input)
        
        # 2. Find authority sources for external linking
        authority_sources = self._find_authority_sources(post_input)
        
        # 3. Quick competitor analysis
        competitors = self._analyze_competitors(post_input)
        
        # 4. Identify content gaps
        gaps = self._identify_content_gaps(competitors, post_input)
        
        return GuestPostResearch(
            industry_trends=trends,
            authority_sources=authority_sources,
            competitor_insights=competitors,
            content_gaps=gaps
        )
    
    def _find_industry_trends(self, post_input: GuestPostInput) -> List[Dict[str, str]]:
        """Find recent industry trends and news."""
        current_year = datetime.now().year
        query = f"{post_input.primary_keyword} trends {current_year}"
        
        results = self.search_api.search(query, num_results=3)
        trends = []
        
        for result in results:
            trends.append({
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'url': result.get('link', '')
            })
        
        return trends
    
    def _find_authority_sources(self, post_input: GuestPostInput) -> List[Dict[str, str]]:
        """Find high-authority sources for external linking."""
        # Target specific authority sites
        authority_domains = [
            "site:architecturaldigest.com",
            "site:ndtv.com/property",
            "site:economictimes.indiatimes.com",
            "site:forbes.com"
        ]
        
        sources = []
        for domain in authority_domains[:2]:  # Limit to 2 to save API calls
            query = f"{post_input.primary_keyword} {domain}"
            results = self.search_api.search(query, num_results=1)
            
            if results:
                sources.append({
                    'title': results[0].get('title', ''),
                    'url': results[0].get('link', ''),
                    'domain': domain.replace('site:', '')
                })
        
        return sources
    
    def _analyze_competitors(self, post_input: GuestPostInput) -> List[CompetitorInsight]:
        """Quick competitor analysis for structure insights."""
        query = f'"{post_input.primary_keyword}" guest post OR article'
        results = self.search_api.search(query, num_results=3)
        
        insights = []
        for result in results:
            url = result.get('link', '')
            if url:
                insight = self._analyze_single_page(url, result.get('title', ''))
                if insight:
                    insights.append(insight)
        
        return insights
    
    def _analyze_single_page(self, url: str, title: str) -> Optional[CompetitorInsight]:
        """Analyze a single competitor page."""
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract structure
            h2s = [h2.get_text(strip=True) for h2 in soup.find_all('h2')[:5]]
            word_count = len(soup.get_text().split())
            
            # Use LLM for quick analysis
            content_preview = soup.get_text()[:3000]
            analysis_prompt = f"""
Analyze this content snippet from a guest post. Extract:
1. Key themes (3-5 topics)
2. Overall tone (one word: sophisticated/casual/technical/aspirational)
3. Unique angles used (2-3 points)

Title: {title}
Content: {content_preview}

Respond in JSON:
{{
  "key_themes": ["theme1", "theme2", "theme3"],
  "tone": "tone_word",
  "unique_angles": ["angle1", "angle2"]
}}
"""
            
            try:
                response = self._invoke_llm(analysis_prompt)
                analysis = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
                
                return CompetitorInsight(
                    url=url,
                    title=title,
                    word_count=word_count,
                    h2_structure=h2s,
                    key_themes=analysis.get('key_themes', []),
                    tone=analysis.get('tone', 'neutral'),
                    unique_angles=analysis.get('unique_angles', [])
                )
            except Exception as e:
                logger.warning(f"LLM analysis failed for {url}: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to analyze {url}: {e}")
            return None
    
    def _identify_content_gaps(self, competitors: List[CompetitorInsight], 
                               post_input: GuestPostInput) -> List[str]:
        """Identify content gaps from competitor analysis."""
        if not competitors:
            return []
        
        # Aggregate competitor themes
        all_themes = []
        for comp in competitors:
            all_themes.extend(comp.key_themes)
        
        gap_prompt = f"""
Based on competitor analysis for "{post_input.primary_keyword}", identify 3-5 content gaps or underexplored angles.

Common themes in existing content: {', '.join(set(all_themes))}
Target publication: {post_input.target_publication}
Unique angle to explore: {post_input.unique_angle}

Respond with JSON:
{{
  "content_gaps": ["gap1", "gap2", "gap3"]
}}
"""
        
        try:
            response = self._invoke_llm(gap_prompt)
            analysis = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            return analysis.get('content_gaps', [])
        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return []

class GuestPostGenerator(BaseLLMHandler):
    """Generates magazine-quality guest posts."""
    
    def _build_guest_post_prompt(self, post_input: GuestPostInput, 
                                 research: GuestPostResearch) -> str:
        """Build comprehensive prompt for guest post generation."""
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Determine word count range
        if post_input.publication_style == "premium":
            min_words, max_words = self.config.premium_word_count
        else:
            min_words, max_words = self.config.target_word_count
        
        # Build research context
        research_context = "**Industry Trends & Recent Data:**\n"
        for trend in research.industry_trends[:3]:
            research_context += f"- {trend['title']}: {trend['snippet']}\n"
        
        authority_links = "\n**Authority Sources for External Linking:**\n"
        for source in research.authority_sources:
            authority_links += f"- [{source['domain']}]({source['url']}): {source['title']}\n"
        
        competitor_insights = "\n**Competitor Structure Analysis:**\n"
        for comp in research.competitor_insights[:2]:
            competitor_insights += f"- {comp.title} ({comp.word_count} words, tone: {comp.tone})\n"
            competitor_insights += f"  H2 structure: {', '.join(comp.h2_structure[:3])}\n"
        
        gaps_section = "\n**Content Gaps to Address:**\n"
        gaps_section += "\n".join([f"- {gap}" for gap in research.content_gaps])
        
        # Build the main prompt
        prompt = f"""You are a premium content writer for {post_input.target_publication}. Create a magazine-quality guest post that balances editorial sophistication with SEO best practices.

**DATE:** {current_date}

**ASSIGNMENT BRIEF:**
- **Publication:** {post_input.target_publication}
- **Brand:** {post_input.brand}
- **Project/Concept:** {post_input.project_name}
- **Title:** {post_input.title}
- **Primary Keyword:** {post_input.primary_keyword}
- **Secondary Keywords:** {', '.join(post_input.secondary_keywords)}
- **Tone:** {post_input.tone}
- **Unique Angle:** {post_input.unique_angle}

**BACKLINK REQUIREMENTS:**
- Anchor text: "{post_input.backlink_anchor_text}"
- URL: {post_input.backlink_url}
- Place this link naturally in H2 #3 (Project Spotlight section)

**RESEARCH CONTEXT:**
{research_context}
{authority_links}
{competitor_insights}
{gaps_section}

**STRICT STRUCTURE REQUIREMENTS:**

**Title (H1):** Use the provided title exactly as given: "{post_input.title}"

**Introduction (100-150 words):**
- Open with an emotional or visual hook
- Introduce the core topic naturally
- Weave in primary keyword "{post_input.primary_keyword}" once organically
- Make it magazine-worthy — sophisticated, not generic

**H2 #1 — Context/Background (200-250 words):**
- Set the scene or define the trend
- Example format: "How [Place/Concept] Became [Something Significant]"
- Include primary keyword naturally once
- Use specific data or anecdotes, not platitudes

**H2 #2 — Key Theme/Innovation (250-300 words):**
- Explore the main trend, shift, or movement
- Can include 1-2 H3 subheadings if needed:
  * H3: [Specific aspect like "Sustainable Development Takes Center Stage"]
  * H3: [Another angle like "Community-Centric Living"]
- Reference one of the industry trends from research
- Weave in 1-2 secondary keywords naturally

**H2 #3 — Project/Concept Spotlight (200-250 words):**
- Subtly highlight {post_input.project_name}
- Focus on vision, values, or architectural philosophy
- **CRITICAL:** Include the backlink here using anchor text "{post_input.backlink_anchor_text}" linking to {post_input.backlink_url}
- Keep promotional tone minimal — emphasize design thinking and impact
- Example: "Projects like {post_input.project_name} demonstrate this philosophy..."

**H2 #4 — Future Outlook/Impact (150-200 words):**
- Zoom out to broader industry implications
- Example: "What This Means for India's Urban Future"
- Include one external link to an authority source from the research provided
- Conclude with forward-looking insight

**Closing (80-100 words):**
- Summarize key insights poetically
- Reconnect with the emotional or aspirational tone
- Optional soft CTA like: "As India's cities evolve, projects that combine design, purpose, and empathy will define how we live tomorrow."

**SEO & STYLE GUIDELINES:**
✅ Total word count: {min_words}-{max_words} words
✅ Use primary keyword "{post_input.primary_keyword}" 2-3 times total (naturally distributed)
✅ Include 1 internal link (the backlink specified above)
✅ Include 1-2 external links to authority sources (use the research provided)
✅ Write in short paragraphs (2-4 lines each)
✅ Avoid generic phrases like "in today's world," "it's no secret," "in conclusion"
✅ Use sensory details and specific examples over abstract claims
✅ Maintain sophisticated, narrative-driven tone throughout
✅ Write as if for a discerning reader, not a search engine

**CRITICAL FORMATTING:**
- Use proper markdown: # for H1, ## for H2, ### for H3
- Bold key terms sparingly
- No emojis or excessive punctuation
- Professional, polished prose throughout

**RESPONSE FORMAT:**
Provide the complete guest post in markdown format, ready to publish. Include ONLY the content — no meta-commentary or explanations.

Begin now:
"""
        
        return prompt
    
    def generate(self, post_input: GuestPostInput, research: GuestPostResearch) -> str:
        """Generate the guest post."""
        logger.info(f"Generating guest post: '{post_input.title}'")
        
        prompt = self._build_guest_post_prompt(post_input, research)
        
        try:
            response = self._invoke_llm(prompt)
            content = response.content.strip()
            
            # Clean up any wrapper artifacts
            content = content.replace("```markdown", "").replace("```", "")
            
            return content
        except Exception as e:
            logger.error(f"Guest post generation failed: {e}")
            raise

# --- Main Writer Class ---
class GuestPostWriter:
    """Main orchestrator for guest post generation."""
    
    def __init__(self, config: GuestPostConfig, key_manager: RotatingAPIKeyManager):
        self.config = config
        self.key_manager = key_manager
        self.search_api = GoogleSearchAPI(config)
        
        # Pass the LLM invoker to components
        invoker = self._invoke_llm_with_retry
        
        self.researcher = GuestPostResearcher(config, key_manager, invoker, self.search_api)
        self.generator = GuestPostGenerator(config, key_manager, invoker)
    
    def _invoke_llm_with_retry(self, prompt: str) -> Any:
        """Invoke LLM with automatic key rotation on quota exhaustion."""
        for attempt in range(self.key_manager.key_count + 1):
            try:
                api_key = self.key_manager.get_key()
                llm = ChatGoogleGenerativeAI(
                    model=self.config.model_name,
                    google_api_key=api_key,
                    temperature=0.7  # Slightly creative for magazine writing
                )
                response = llm.invoke(prompt)
                return response
            except ResourceExhausted as e:
                logger.warning(f"Quota exhausted. Rotating key... (Attempt {attempt + 1}/{self.key_manager.key_count})")
                self.key_manager.rotate_key()
                sleep(2)
            except Exception as e:
                logger.error(f"LLM invocation error: {e}", exc_info=True)
                raise
        
        error_msg = "All API keys exhausted."
        logger.critical(error_msg)
        raise ResourceExhausted(error_msg)
    
    def write_guest_post(self, post_input: GuestPostInput) -> Dict[str, Any]:
        """Complete guest post writing workflow."""
        logger.info(f"Starting guest post generation: '{post_input.title}'")
        
        try:
            # Phase 1: Research
            logger.info("[Phase 1/2] Gathering research and competitor insights...")
            research = self.researcher.gather_research(post_input)
            
            # Phase 2: Generate
            logger.info("[Phase 2/2] Generating magazine-quality content...")
            content = self.generator.generate(post_input, research)
            
            metadata = {
                "generated_at": datetime.now().isoformat(),
                "word_count": len(content.split()),
                "primary_keyword": post_input.primary_keyword,
                "target_publication": post_input.target_publication,
                "backlink_included": bool(post_input.backlink_url)
            }
            
            return {
                "success": True,
                "content": content,
                "metadata": metadata,
                "post_input": post_input
            }
            
        except Exception as e:
            logger.error(f"Guest post generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "post_input": post_input
            }

# --- File I/O ---
def load_guest_post_inputs(filepath: str) -> List[GuestPostInput]:
    """Load guest post inputs from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        inputs = []
        for item in data:
            inputs.append(GuestPostInput(**item))
        
        logger.info(f"Loaded {len(inputs)} guest post input(s)")
        return inputs
        
    except Exception as e:
        logger.error(f"Error loading inputs from {filepath}: {e}")
        return []

def save_guest_post(content: str, post_input: GuestPostInput, metadata: Dict[str, Any]):
    """Save generated guest post to file."""
    output_dir = Path("generated_guest_posts")
    output_dir.mkdir(exist_ok=True)
    
    # Create filename from title
    safe_title = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in post_input.title)
    safe_title = safe_title.replace(' ', '_')[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.md"
    
    filepath = output_dir / filename
    
    # Add metadata header
    header = f"""---
Title: {post_input.title}
Brand: {post_input.brand}
Project: {post_input.project_name}
Target Publication: {post_input.target_publication}
Primary Keyword: {post_input.primary_keyword}
Generated: {metadata['generated_at']}
Word Count: {metadata['word_count']}
---

"""
    
    full_content = header + content
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    logger.info(f"Saved guest post to: {filepath}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate magazine-quality guest posts")
    parser.add_argument('--config', type=str, required=True, help='JSON file with guest post configurations')
    parser.add_argument('--api-keys', type=str, default='geminaikey', help='File containing Gemini API keys')
    parser.add_argument('--google-api', type=str, default='googlesearchapi', help='Google Search API key file')
    parser.add_argument('--google-cx', type=str, default='googlecx', help='Google Custom Search Engine ID file')
    parser.add_argument('--workers', type=int, default=3, help='Max parallel workers')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = GuestPostConfig(
            api_key_file=args.api_keys,
            google_search_api_file=args.google_api,
            google_cx_file=args.google_cx,
            verbose=args.verbose,
            max_workers=args.workers
        )
        
        # Load API keys
        gemini_keys = APIKeyManager.get_api_keys(config.api_key_file)
        if not gemini_keys:
            logger.critical("No Gemini API keys found. Exiting.")
            return 1
        
        key_manager = RotatingAPIKeyManager(gemini_keys)
        
        # Load guest post inputs
        post_inputs = load_guest_post_inputs(args.config)
        if not post_inputs:
            logger.error("No valid guest post inputs found.")
            return 1
        
        # Initialize writer
        writer = GuestPostWriter(config, key_manager)
        
        # Process guest posts
        logger.info(f"Processing {len(post_inputs)} guest post(s)...")
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_input = {
                executor.submit(writer.write_guest_post, post_input): post_input 
                for post_input in post_inputs
            }
            
            for future in as_completed(future_to_input):
                result = future.result()
                
                if result['success']:
                    save_guest_post(
                        result['content'],
                        result['post_input'],
                        result['metadata']
                    )
                    logger.info(f"✅ SUCCESS: '{result['post_input'].title}'")
                else:
                    logger.error(f"❌ FAILED: '{result['post_input'].title}' - {result['error']}")
        
        logger.info("All guest posts processed.")
        return 0
        
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
