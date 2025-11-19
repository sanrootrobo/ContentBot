# ContentBot
# SEO Blog Writer 🚀

An intelligent, AI-powered SEO blog content generator that analyzes competitor content and creates optimized, high-quality blog posts using Google's Gemini AI. Perfect for content marketers, SEO specialists, and digital agencies looking to scale their content production while maintaining quality and search engine optimization.

## ✨ Features

- **Competitor Analysis**: Automatically researches and analyzes top-ranking competitor content using Google Custom Search API
- **AI-Powered Writing**: Leverages Google's Gemini 2.5 Pro for generating human-quality, SEO-optimized content
- **Funnel-Aware Content**: Configurable content focus from TOFU (awareness) to BOFU (conversion)
- **Client Context Integration**: Optionally incorporates your client's website content to match brand voice and messaging
- **NLP-Optimized**: Generates content structured for better search engine comprehension
- **Batch Processing**: Process multiple blog posts from a single JSON configuration file
- **Markdown Output**: Clean, well-formatted markdown files with SEO metadata
- **Keyword Intelligence**: Strategic integration of primary and secondary keywords without stuffing
- **Rate Limit Handling**: Built-in retry logic for API quota management

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key
- Google Custom Search API key and Search Engine ID
- Required Python packages (see Installation)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/seo-blog-writer.git
cd seo-blog-writer
```

2. Install required dependencies:
```bash
pip install requests beautifulsoup4 langchain-google-genai google-api-core
```

3. Set up your API keys:

Create three text files in the project directory:
- `geminaikey` - Contains your Google Gemini API key
- `googlesearchapi` - Contains your Google Custom Search API key
- `googlecx` - Contains your Google Custom Search Engine ID

## 🚀 Quick Start

1. Create a JSON configuration file (`blog_config.json`) with your blog post specifications:

```json
[
  {
    "brand": "TechCorp",
    "title": "The Ultimate Guide to Cloud Migration in 2025",
    "primary_keywords": ["cloud migration", "cloud strategy"],
    "secondary_keywords": ["hybrid cloud", "cloud security", "migration planning"],
    "target_audience": "IT Decision Makers",
    "tone": "Professional and Informative",
    "content_type": "Guide"
  }
]
```

2. Run the script:

```bash
python seo_blog_writer.py -c blog_config.json
```

3. Find your generated blog post as a markdown file: `the_ultimate_guide_to_cloud_migration_in_2025.md`

## 📖 Usage Examples

### Basic Usage
```bash
python seo_blog_writer.py -c blog_config.json
```

### With Client Website Context (for brand-aligned content)
```bash
python seo_blog_writer.py -c blog_config.json --client client_website_dump.txt
```

### Custom Word Count Range
```bash
python seo_blog_writer.py -c blog_config.json --min-words 2000 --max-words 4000
```

### Funnel-Stage Focused Content
```bash
# TOFU (awareness-focused, educational)
python seo_blog_writer.py -c blog_config.json --client-focus 0.2

# MOFU (balanced educational + promotional)
python seo_blog_writer.py -c blog_config.json --client-focus 0.5

# BOFU (conversion-focused, product-heavy)
python seo_blog_writer.py -c blog_config.json --client-focus 0.9
```

### Verbose Mode (detailed logging)
```bash
python seo_blog_writer.py -c blog_config.json --verbose
```

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --config-file` | Path to JSON configuration file (required) | - |
| `--api-key-file` | Path to Gemini API key file | `geminaikey` |
| `--google-search-api` | Path to Google Search API key file | `googlesearchapi` |
| `--google-cx` | Path to Google Custom Search Engine ID file | `googlecx` |
| `--client` | Path to client website content dump file | None |
| `--client-focus` | Content focus level (0.0=TOFU to 1.0=BOFU) | 0.5 |
| `--min-words` | Minimum word count | 1500 |
| `--max-words` | Maximum word count | 3000 |
| `--verbose` | Enable detailed logging | False |

## 📝 JSON Configuration Format

```json
[
  {
    "brand": "Your Brand Name",
    "title": "Blog Post Title",
    "primary_keywords": ["keyword1", "keyword2"],
    "secondary_keywords": ["keyword3", "keyword4", "keyword5"],
    "target_audience": "Your Target Audience",
    "tone": "Desired Tone (e.g., Professional, Casual, Technical)",
    "content_type": "Blog Post Type (e.g., Guide, Tutorial, Listicle)"
  }
]
```

### Required Fields
- `brand`: Your brand/company name
- `title`: The blog post title
- `primary_keywords`: Array of main keywords (at least 1 required)

### Optional Fields
- `secondary_keywords`: Array of supporting keywords (default: [])
- `target_audience`: Who the content is for (default: "General Audience")
- `tone`: Writing tone (default: "Informative")
- `content_type`: Type of content (default: "Blog Post")

## 🎯 Client Focus Levels

The `--client-focus` parameter controls how promotional vs. educational your content is:

| Level | Stage | Description | Use Case |
|-------|-------|-------------|----------|
| 0.0 - 0.2 | TOFU | Pure educational, minimal branding | Awareness content, thought leadership |
| 0.2 - 0.4 | Upper MOFU | Educational with subtle positioning | Building trust, establishing expertise |
| 0.4 - 0.6 | MOFU | Balanced education + solutions | Consideration stage, solution comparisons |
| 0.6 - 0.8 | Lower MOFU/BOFU | Solution-focused with CTAs | Decision stage, competitive positioning |
| 0.8 - 1.0 | BOFU | Heavy product focus, conversion-driven | Sales content, product pages |

## 📂 Output Format

Generated markdown files include:

- YAML frontmatter with metadata
- SEO-optimized blog content
- Proper heading hierarchy
- FAQ section
- SEO metadata summary
- Keyword tracking

Example output structure:
```markdown
---
title: "Your Blog Title"
brand: YourBrand
created: 2025-01-15T10:30:00
word_count: 2453
primary_keywords:
  - "keyword1"
  - "keyword2"
---

# Your Blog Title

[Blog content here...]

## FAQ

[4 relevant FAQs...]

## Conclusion

[Conclusion content...]

---

## SEO Metadata
...
```

## 🔍 How It Works

1. **Competitor Research**: Queries Google Custom Search API for top-ranking content based on your keywords
2. **Content Analysis**: Scrapes and analyzes competitor pages for:
   - Word count patterns
   - Common topics and keywords
   - Content structure
   - Title patterns
   - Meta descriptions
3. **Insight Generation**: Identifies content gaps and opportunities
4. **AI Generation**: Uses Gemini AI with competitor insights and your specifications to generate superior content
5. **Quality Assurance**: Ensures NLP-friendly structure, proper keyword integration, and brand alignment
6. **Output**: Saves as formatted markdown with full metadata

## 🛡️ Error Handling

The script includes robust error handling for:
- API rate limits (automatic retry with backoff)
- Network timeouts
- Invalid configurations
- Missing API keys
- Failed competitor analysis


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is designed to assist in content creation, not replace human oversight. Always review and edit generated content before publication to ensure accuracy, brand alignment, and quality.

## 🙋 Support

For issues, questions, or feature requests, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Image suggestion and optimization
- [ ] Internal linking recommendations
- [ ] Readability score analysis
- [ ] Schema markup generation
- [ ] Content refresh recommendations
- [ ] SERP feature optimization

---

Made with ❤️ for content creators and SEO professionals
