# Joplin UX Analysis

AI-powered analysis of 754 user feedback items to identify product improvement priorities for the Joplin note-taking application.

## Overview

This project demonstrates how to transform unstructured user feedback into actionable product insights using large language models and data analysis techniques. By scraping and analyzing feedback from multiple platforms, it identifies the most critical pain points affecting user retention and adoption.

## Key Findings

- **Analyzed 754 feedback items**: 511 pain points and 243 feature requests
- **Primary Issues Identified**: 
  - Sync reliability (86% of sync issues marked high-severity)
  - UI/UX modernization (102 unique interface complaints)
- **Data-Driven Recommendations**: Prioritized roadmap based on frequency and impact analysis
- **Processing Efficiency**: Automated pipeline completed analysis in ~10 minutes for $0.34 in API costs

## Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- Reddit API credentials (for data collection)
- GitHub personal access token (for data collection)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/HanifCarroll/joplin-ux-analysis.git
cd joplin-ux-analysis
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Usage

#### Available pipeline components:
```bash
# 1. Data collection (optional - raw data already included)
python original_data/data_collector.py

# 2. Data cleaning (optional - cleaned data already included)
python cleaned_data/clean_data.py

# 3. AI categorization (requires analyzed_threads.json - pre-processed file included)
cd categorized_data && python categorize_joplin.py

# 4. Generate visualizations
cd categorized_data && python generate_charts.py
```

**Note**: The intermediate processing script that converts cleaned data to `analyzed_threads.json` format is not included in this repository. However, the pre-processed `analyzed_threads.json` file is available, allowing you to run the AI categorization and visualization steps.

#### Analyze existing results:
The repository includes pre-processed results. You can directly explore:
- `categorized_data/categorized_results.json` - Complete analysis results
- `categorized_data/pain_points_distribution.png` - Pain point categories chart
- `categorized_data/priority_matrix.png` - Impact vs frequency matrix

## Methodology

### Data Collection
- **GitHub Issues**: 349 items from Joplin repository
- **Reddit**: 50 threads from r/joplin and related subreddits  
- **Discourse Forums**: 112 discussions from Joplin community forum

### AI-Powered Categorization
- **Model**: OpenAI o4-mini for cost-effective batch processing
- **Categories**: 15 pain point categories (Sync, UI/UX, Performance, etc.)
- **Validation**: High confidence ratings (75% of categorizations marked "High" confidence)

### Analysis Pipeline
1. **Data Extraction**: Multi-platform scraping with rate limiting
2. **Cleaning**: Deduplication and text preprocessing  
3. **Categorization**: LLM-based classification with reasoning
4. **Prioritization**: Impact vs frequency matrix analysis
5. **Visualization**: Automated chart generation

## Project Structure

```
joplin-ux-analysis/
├── original_data/          # Raw scraped data
│   ├── reddit_threads.json
│   ├── github_issues.json
│   ├── discourse_threads.json
│   └── data_collector.py
├── cleaned_data/           # Processed and cleaned data
│   ├── clean_data.py
│   └── *_cleaned.json
├── categorized_data/       # AI analysis results
│   ├── categorize_joplin.py
│   ├── categorized_results.json
│   ├── generate_charts.py
│   └── *.png charts
└── pyproject.toml         # Dependencies and config
```

## Key Results

### Pain Point Distribution
- **UI/UX Issues**: 102 complaints (most frequent)
- **Sync Problems**: 65 complaints (highest severity - 86% high-priority)
- **Editor Issues**: 44 complaints
- **Performance**: 28 complaints

### Critical Insights
1. **Sync reliability is the #1 churn driver** - users abandon Joplin after data loss
2. **UI friction blocks adoption** - interface described as "clunky" and "dated"
3. **Multi-window support highly requested** - 11 specific complaints found

## Technical Details

### Dependencies
- `openai>=1.25.0` - LLM analysis and categorization
- `praw>=7.8.1` - Reddit API integration
- `pygithub>=2.6.1` - GitHub Issues API
- `requests>=2.32.3` - Discourse forum scraping
- `matplotlib` - Data visualization
- `python-dotenv>=1.1.0` - Environment management

### Performance Metrics
- **API Cost**: $0.34 (OpenAI o4-mini)
- **Coverage**: 754 feedback items across 3 platforms

## Business Case Study

For detailed business insights, strategic recommendations, and product roadmap based on this analysis, see the complete case study: **[Data-Driven Product Strategy for Joplin](https://hanifcarroll.com/projects/joplin-product-analysis)**

The case study covers:
- Executive summary and business impact
- Strategic recommendations with phased implementation
- Success metrics and ROI projections
- Product thinking and insights

## Contact

Built by [Hanif Carroll](https://hanifcarroll.com) as a product engineering portfolio project demonstrating data-driven product strategy development. 